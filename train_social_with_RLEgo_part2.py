import os
import copy
import time
import shutil
from collections import deque

import wandb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from rl import utils
from rl.envs import make_vec_envs

from driving_sim.envs import *  # 0. environment
from driving_sim.utils.info import *

from pretext.loss import *  # 1. inference network
from pretext.pretext_models.cvae_model import CVAEIntentPredictor
from pretext.data_loader import makeDataset

from rl.ppo import PPO  # 2. RL network (for ego)
from rl.storage import RolloutStorage
from rl.rl_models.policy import Policy
from configs.ego_config.config import Config

from rl.ppo import PPO_Social  # 3. RL network (for social)
from rl.storage_social import RolloutStorageSocial
from rl.rl_models_social.policy import SocialPolicy
from configs.social_config.config import SocialConfig


def main():

    use_guide_model = True
    guide_model_dir = 'data/rl_social_guide/checkpoints/Social_27776.pt'

    #################################################
    #### 0. Configuration & Environment set up
    #################################################

    # initialize the config instance: main configuration is in ego_config
    config = Config()
    social_config = SocialConfig()
    social_config.training.num_processes = config.training.num_processes
    social_config.ppo.num_mini_batch = config.ppo.num_mini_batch
    social_config.ppo.num_steps = config.ppo.num_steps

    # save policy to output_dir
    if os.path.exists(config.training.output_dir):
        shutil.rmtree(config.training.output_dir)  # delete an entire directory tree
    os.makedirs(config.training.output_dir)
    shutil.copytree('configs', os.path.join(config.training.output_dir, 'configs'))

    # cuda and pytorch settings
    torch.manual_seed(config.env_config.env.seed)
    torch.cuda.manual_seed_all(config.env_config.env.seed)
    if config.training.cuda:
        if config.training.cuda_deterministic:  # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:  # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    torch.set_num_threads(config.training.num_threads)
    device = torch.device("cuda" if config.training.cuda else "cpu")
    print('Using device: ', device)

    if config.training.render:
        config.training.num_processes = 1
        config.ppo.num_mini_batch = 1
        social_config.training.num_processes = 1
        social_config.ppo.num_mini_batch = 1

    # Create a wrapped, monitored VecEnv
    env_num = config.training.num_processes
    human_num = config.env_config.car.max_veh_num
    envs = make_vec_envs(config.env_config.env.env_name, config.env_config.env.seed, config.training.num_processes,
                         config.env_config.reward.gamma, None, device, False, config=config)

    #################################################
    #### 1. Inference network (Ego agent)
    #################################################
    print("###################### 1. Inference network (Ego agent) ######################")
    inference_model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict', decoder_base='lstm', config=config)
    inference_model.load_state_dict(torch.load('trained_models/pretext/public_ours/checkpoints/995.pt', map_location=device))
    envs.pred_model = inference_model.encoder
    envs.pred_model.eval()

    #################################################
    #### 2. RL network (Ego agent)
    #################################################
    print("###################### 2. RL network (Ego agent) ######################")
    actor_critic = Policy(envs.observation_space.spaces, envs.action_space,
                          base_kwargs=config, base=config.env_config.robot.policy)
    actor_critic.load_state_dict(torch.load('trained_models/rl/con40/public_ours_rl/checkpoints/26800.pt', map_location=device))
    eval_recurrent_hidden_states = {'rnn': torch.zeros(config.training.num_processes, 1, config.network.rnn_hidden_size, device=device)}
    eval_masks = torch.zeros(config.training.num_processes, 1, device=device)

    #################################################
    #### 3. RL network (Social agent)
    #################################################
    print("###################### 3. RL network (Social agent) ######################")
    actor_critic_social = SocialPolicy(envs.observation_space.spaces, envs.action_space,
                                       base_kwargs=social_config, base=social_config.env_config.robot.policy,
                                       meta=True)
    if use_guide_model:
        actor_critic_social_target = SocialPolicy(envs.observation_space.spaces, envs.action_space,
                                                  base_kwargs=social_config, base=social_config.env_config.robot.policy)
        actor_critic_social_target.load_state_dict(torch.load(guide_model_dir, map_location=device))
    else:
        actor_critic_social_target = None

    dummy_env = TIntersectionRobustnessSocial()
    dummy_env.configure(config.env_config)
    rl_ob_space_social = dummy_env.observation_space.spaces
    dummy_env.close()
    for key_ in ['pretext_actions', 'pretext_infer_masks', 'pretext_masks', 'pretext_nodes',
                 'pretext_spatial_edges', 'pretext_temporal_edges', 'spatial_edges', 'true_labels']:
        rl_ob_space_social.pop(key_)
    rollouts_social = RolloutStorageSocial(config.ppo.num_steps, config.training.num_processes,
                                           rl_ob_space_social, envs.action_space, social_config.network.rnn_hidden_size, device)

    # ppo optimizer
    agent_social = PPO_Social(actor_critic_social, social_config.ppo.clip_param, social_config.ppo.epoch,
                              social_config.ppo.num_mini_batch, social_config.ppo.value_loss_coef,
                              social_config.ppo.entropy_coef, lr=social_config.training.lr,
                              eps=social_config.training.eps, max_grad_norm=social_config.training.max_grad_norm,
                              actor_critic_backup=actor_critic_social_target)

    #################################################
    #### 4. Initialization for training
    #################################################
    print("###################### 4. Initialization for training ######################")
    # 4.1 initialize the environment
    obs = envs.reset()

    # 4.3. initialize rollout storage (for social)
    if isinstance(obs, dict):
        for key in rl_ob_space_social:
            rollouts_social.obs[key][0].copy_(obs[key])
    else:
        rollouts_social.obs[0].copy_(obs)
    rollouts_social.to(device)

    # 4.4. initialize wandb (if use wandb)
    use_wandb = config.training.use_wandb
    if use_wandb:
        from datetime import date
        exp_name = date.today().strftime("%Y%m%d") + "_Social"
        wandb.init(project="Robustness-MARL", name=exp_name)
        wandb.define_metric("env_step")
        wandb.define_metric("train_step")
        wandb.define_metric("env/*", step_metric="env_step")
        wandb.define_metric("train/*", step_metric="train_step")
        env_iter, train_iter = 0, 0

    # 4.4. initialize some variables (for logging or configuration)
    episode_rewards = deque(maxlen=100)  # just for logging & display purpose
    episode_rewards_env = np.zeros(envs.num_envs)
    start = time.time()
    num_updates = int(config.training.num_env_steps) // config.ppo.num_steps // config.training.num_processes

    #################################################
    #### 5. Main training loop start
    ####   5.1. environment step
    ####   5.2. training inference net (for Ego) evey 10 step.
    ####   5.3. training RL net (for Ego) when inference net is not trained.
    ####   5.4. training RL net (for Social)
    ####   5.5. logging & saving the model
    #################################################
    print("###################### 5. Main training loop start ######################")

    for j in range(num_updates):

        # decrease learning rate linearly
        # j and num_updates_lr_decrease are just used for calculating new lr
        if config.training.use_linear_lr_decay:
            utils.update_linear_schedule(agent_social.optimizer, j, num_updates, social_config.training.lr)

        # 5.1. environment step
        # rollout the current policy for 30 steps, and store {obs, action, reward, rnn_hxs, masks, etc} to memory
        for step in range(config.ppo.num_steps):

            # 5.1.1. Sample actions (Ego)
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = \
                    actor_critic.act(obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

            # 5.1.2. Sample actions (Social)
            with torch.no_grad():
                rollouts_obs = {}
                for key in rollouts_social.obs:
                    rollouts_obs[key] = rollouts_social.obs[key][step]

                rollouts_hidden_s = {}
                for key in rollouts_social.recurrent_hidden_states:
                    rollouts_hidden_s[key] = rollouts_social.recurrent_hidden_states[key][step]

                value_social, action_social, action_log_prob_social, recurrent_hidden_states_social = \
                    actor_critic_social.act(rollouts_obs, rollouts_hidden_s, rollouts_social.masks[step])

            # 5.1.3. Rendering the environment (if render)
            if config.training.render:
                envs.render()

            # 5.1.4. Observe reward and next obs
            action_all = torch.cat((action, action_social), dim=-1)
            obs, reward, done, infos = envs.step(action_all)

            # 5.1.5. save the data for logging & wandb log for terminated environment
            for env_idx in range(envs.num_envs):
                episode_rewards_env[env_idx] += reward[env_idx]
                if done[env_idx]:  # if an episode ends
                    episode_rewards.append(episode_rewards_env[env_idx])
                    if use_wandb:
                        wandb.log({
                            "env_step": env_iter,
                            "env/time_to_terminate": infos[env_idx]['episode']['l'],
                            "env/collision": 1 if isinstance(infos[env_idx]['info'], Collision) else 0,
                            "env/success": 1 if isinstance(infos[env_idx]['info'], ReachGoal) else 0,
                            "env/time_out": 1 if isinstance(infos[env_idx]['info'], Timeout) else 0,
                            "env/reward": episode_rewards_env[env_idx],
                            "env/collision_vehicle_type_beta": infos[env_idx]['collision_vehicle_type'][1],
                        })
                        env_iter += 1
                    episode_rewards_env[env_idx] = 0.0

            # 5.1.7. save the data for RL network (Ego agent)
            obs_rl = {}
            for key_ in ['robot_node', 'spatial_edges', 'temporal_edges']:
                obs_rl[key_] = copy.deepcopy(obs[key_])

            # If done then clean the history of observations.
            eval_masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)

            # 5.1.8. save the data for RL network (Social agents)
            obs_rl_social = {}
            for key_ in ['front_car_information', 'objective_weight', 'robot_node',
                         'social_car_information', 'temporal_edges', 'valid_training']:
                obs_rl_social[key_] = copy.deepcopy(obs[key_])

            reward_social = torch.zeros((envs.num_envs, human_num))
            for env_idx, info in enumerate(infos):
                reward_social[env_idx] = info['social_reward']

            # If done then clean the history of observations.
            masks_social = copy.deepcopy(obs['pretext_masks'])
            masks_social_reward = copy.deepcopy(obs['pretext_masks'])
            bad_masks_social = torch.ones_like(obs['pretext_masks'])
            for env_idx, done_ in enumerate(done):
                if done_:
                    masks_social[env_idx] = 0.0
                    masks_social_reward[env_idx] = 0.0
                    bad_masks_social[env_idx] = 0.0
                    bad_masks_social[env_idx][infos[env_idx]['social_done']] = 1.0

            rollouts_social.insert(obs_rl_social, recurrent_hidden_states_social, action_social, action_log_prob_social,
                                   value_social, reward_social, masks_social, masks_social_reward, bad_masks_social)

        # 5.4. training RL net (for Social)
        # 5.4.1. calculate predicted value from value network
        with torch.no_grad():
            rollouts_obs = {}
            for key in rollouts_social.obs:
                rollouts_obs[key] = rollouts_social.obs[key][-1]

            rollouts_hidden_s = {}
            for key in rollouts_social.recurrent_hidden_states:
                rollouts_hidden_s[key] = rollouts_social.recurrent_hidden_states[key][-1]
            next_value_social = actor_critic_social.get_value(rollouts_obs, rollouts_hidden_s, rollouts_social.masks[-1]).detach()

        # 5.4.2. compute return from next_value
        rollouts_social.compute_returns(next_value_social, social_config.ppo.use_gae,
                                        social_config.env_config.reward.gamma, social_config.ppo.gae_lambda,
                                        social_config.training.use_proper_time_limits)

        # 5.4.3. use ppo loss to do backprop on network parameters
        value_loss_social, actor_loss_social, dist_entropy_social, _, policy_reg = agent_social.update(rollouts_social)
        rollouts_social.after_update()  # clear the rollout storage since ppo is on-policy

        # 5.5. logging & saving the model.
        # 5.5.1. wandb log if use wandb (for training)
        if use_wandb:
            wandb.log({
                "train_step": train_iter,
                "train/social_value_loss": value_loss_social,
                "train/social_actor_loss": actor_loss_social,
                "train/social_dist_entropy": dist_entropy_social,
                "train/social_policy_reg": policy_reg
            })
            train_iter += 1

        # 5.5.2. save the model for every interval-th episode or for the last epoch
        if (j % config.training.save_interval == 0 or j == num_updates - 1):
            save_path = os.path.join(config.training.output_dir, 'checkpoints')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if j == 0 or j == num_updates - 1:
                torch.save(inference_model.state_dict(), os.path.join(save_path, 'Encoder_' + '%.5i' % j + ".pt"))
                torch.save(actor_critic.state_dict(), os.path.join(save_path, 'Ego_' + '%.5i' % j + ".pt"))
            torch.save(actor_critic_social.state_dict(), os.path.join(save_path, 'Social_' + '%.5i' % j + ".pt"))

        # 5.5.3. log some important features
        if j % config.training.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config.training.num_processes * config.ppo.num_steps
            end = time.time()

            # log on terminal
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                  .format(j, total_num_steps,
                          int(total_num_steps / (end - start)), len(episode_rewards), np.mean(episode_rewards),
                          np.median(episode_rewards), np.min(episode_rewards),
                          np.max(episode_rewards)))

            # save as csv
            df = pd.DataFrame({
                'misc/nupdates': [j],
                'misc/total_timesteps': [total_num_steps],
                'fps': int(total_num_steps / (end - start)),
                'eprewmean': [np.mean(episode_rewards)],
            })

            if os.path.exists(os.path.join(config.training.output_dir, 'progress.csv')) and j > 20:
                df.to_csv(os.path.join(config.training.output_dir, 'progress.csv'), mode='a', header=False, index=False)
            else:
                df.to_csv(os.path.join(config.training.output_dir, 'progress.csv'), mode='w', header=True, index=False)


def process_traj(data, save_key_list, nenv, traj_len, human_num):
    new_data = []
    # convert each key in data to np array
    for key in data:
        data[key] = np.array(data[key])

    # calculate the start index for each traj
    humans_masks = np.array(data['pretext_masks']) # [traj_len, nenv, human_num]
    done_masks = np.expand_dims(np.array(data['dones']), axis=-1) # [traj_len, nenv, 1]

    # add a sentinel in the front
    humans_masks = np.concatenate((np.zeros((1, nenv, human_num)), humans_masks), axis=0) # 21, nenv, human_num
    humans_start_idx = np.logical_not(humans_masks).cumsum(axis=0).argmax(axis=0)
    done_masks = np.concatenate((np.zeros((1, nenv, 1), dtype=bool), done_masks), axis=0)
    done_start_idx = done_masks.cumsum(axis=0).argmax(axis=0)
    # if done_masks are all zeros, done_start_idx should be 0

    start_idx = np.maximum(humans_start_idx, done_start_idx)

    # slice the traj and save in return value
    for i in range(nenv):  # for each env
        for j in range(human_num):
            # if traj_len = 20, the largest max index is 18 (so that each traj has at least 2 steps)
            if start_idx[i, j] < traj_len-1:
            # the largest max index is 15 (so that each traj has at least 5 steps)
            # if start_idx[i, j] < traj_len - 4:
                cur_dict = {}
                for key in save_key_list:
                    # only save one label for each traj
                    if key == 'labels':
                        cur_dict[key] = data[key][-1, i, j]
                    else:
                        # data[key]: [traj_len, nenv, human_num, ?]
                        cur_dict[key] = data[key][start_idx[i, j]:, i, j]
                # change the px of pretext_nodes to odometry (displacement since 20 steps ago)
                cur_dict['pretext_nodes'][:, 0] = cur_dict['pretext_nodes'][:, 0] - cur_dict['pretext_nodes'][0, 0]
                # error check: all px must be non-negative
                assert (cur_dict['pretext_nodes'][:, 0] >= 0).all(), cur_dict['pretext_nodes'][:, 0]

                new_data.append(copy.deepcopy(cur_dict))
    return new_data


if __name__ == '__main__':
    main()
