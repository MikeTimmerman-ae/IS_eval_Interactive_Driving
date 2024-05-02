import os
import copy
import time
import shutil
import argparse
from collections import deque

import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim

from rl import utils
from rl.envs import make_vec_envs

from pretext.loss import *  # 1. inference network
from pretext.pretext_models.cvae_model import CVAEIntentPredictor
from pretext.data_loader import makeDataset

from rl.ppo import PPO  # 2. RL network (for ego)
from rl.rl_models.policy import Policy
from rl.storage import RolloutStorage

from rl.ppo import PPO_Social  # 3. RL network (for social)
from rl.rl_models_social.policy import SocialPolicy
from rl.storage_social import RolloutStorageSocial

from configs.ego_config.config import Config
from configs.social_config.config import SocialConfig
from driving_sim.envs import *
from driving_sim.utils.info import *


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--social_agent', default="data/rl_social_gmeta/checkpoints/Social_27776.pt", action='store_true')
    parser.add_argument('--ego_start_baseline', default=True)
    parser.add_argument('--use_pretrained_encoder', default=True)
    parser.add_argument('--encoder', default="data/encoder_idm/checkpoints/49.pt")
    parser.add_argument('--use_idm', default=False)
    ### Training distribution settings
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--gmm', default=None)                      # If specified, use GMM to generate beta's
    parser.add_argument('--mean', default=None)                     # If specified, use normal dist to generate beta's
    parser.add_argument('--std', default=None)
    parser.add_argument('--naturalistic_dist', default=None)        # If specified, reweigh ego rewards
    ################
    test_args = parser.parse_args()

    #################################################
    #### 0. Configuration & Environment set up
    #################################################
    print("###################### 0. Configuration & Environment set up ######################")

    # initialize the config instance
    # main configuration is in ego_config
    config = Config()
    social_config = SocialConfig()
    social_config.training.num_processes = config.training.num_processes
    social_config.ppo.num_mini_batch = config.ppo.num_mini_batch
    social_config.ppo.num_steps = config.ppo.num_steps
    social_config.training.cuda = config.training.cuda

    # Set-up experiment
    if test_args.mean is not None and test_args.std is not None:
        save_path = f'data/{test_args.experiment}/rl_ego_{test_args.mean.replace(".","")}_{test_args.std.replace(".","")}'
        config.training.output_dir = save_path
        print(f"Writing output to {config.training.output_dir}")
    elif test_args.gmm is not None:
        # Train ego policy on gaussian mixture model
        save_path = f'data/{test_args.experiment}/rl_ego_{test_args.gmm}'
        config.training.output_dir = save_path
        print(f"Writing output to {config.training.output_dir}")
        test_args.gmm = {'k': 2, 'mean': [0.5, 1.2], 'std': [0.5, 0.5], 'weights': [0.5, 0.5]}
        # test_args.gmm = {'k': 3, 'mean': [0.5, 1.2, 2.14], 'std': [0.5, 0.5, 0.5], 'weights': [1/3, 1/3, 1/3]}
    elif test_args.experiment is not None:
        save_path = f'data/{test_args.experiment}'
        config.training.output_dir = save_path
        print(f"Writing output to {config.training.output_dir}")

    # save policy to output_dir
    if os.path.exists(config.training.output_dir) and config.training.overwrite:  # if I want to overwrite the directory
        shutil.rmtree(config.training.output_dir)  # delete an entire directory tree

    if not os.path.exists(config.training.output_dir):
        os.makedirs(config.training.output_dir)

    shutil.copytree('configs', os.path.join(config.training.output_dir, 'configs'))

    # cuda and pytorch settings
    torch.manual_seed(config.env_config.env.seed)
    torch.cuda.manual_seed_all(config.env_config.env.seed)
    if config.training.cuda:
        # torch.cuda.set_device(1)
        if config.training.cuda_deterministic:  # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:  # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(config.training.num_threads)
    device = torch.device("cuda:0" if config.training.cuda else "cpu")
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
                         config.env_config.reward.gamma, None, device, False, config=config,
                         mean=test_args.mean, std=test_args.std, nat=test_args.naturalistic_dist, gmm=test_args.gmm)

    #################################################
    #### 1. Inference network (Ego agent)
    #################################################
    print("###################### 1. Inference network (Ego agent) ######################")

    # 1.1. make inference network
    inference_model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                                          decoder_base='lstm', config=config)
    envs.pred_model = inference_model.encoder  # put the encoder network to the environment (for latent inference)

    if test_args.ego_start_baseline:
        inference_model.load_state_dict(torch.load('trained_models/pretext/public_ours/checkpoints/995.pt', map_location=device))
        envs.pred_model = inference_model.encoder  # put the encoder network to the environment (for latent inference)
        envs.pred_model.eval()
    else:
        loss_func = CVAE_loss(config=config, schedule_kl_method='constant')
        optimizer = optim.Adam(inference_model.parameters(), lr=config.pretext.lr, weight_decay=1e-6)

        # 1.2. set up components of dataloader for inference network training
        enc_key = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'labels', 'pretext_masks', 'dones']
        enc_save_key = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'labels']

        data_list = []  # list for all data collected
        data = {}  # buffer for data from env
        for key in enc_key:
            data[key] = []

    #################################################
    #### 2. RL network (Ego agent)
    #################################################
    print("###################### 2. RL network (Ego agent) ######################")

    # 2.1. create RL policy network
    actor_critic = Policy(envs.observation_space.spaces, envs.action_space,
                          base_kwargs=config, base=config.env_config.robot.policy)
    if test_args.ego_start_baseline:
        actor_critic.load_state_dict(torch.load('trained_models/rl/con40/public_ours_rl/checkpoints/26800.pt', map_location=device))

    # 2.2. setting up the replay memory
    # exclude the keys in obs that are only for pretext network to save memory & construct an env without pretext obs
    dummy_env = TIntersection()
    dummy_env.configure(config.env_config)
    rl_ob_space = dummy_env.observation_space.spaces
    dummy_env.close()
    rollouts = RolloutStorage(config.ppo.num_steps, config.training.num_processes, rl_ob_space,
                              envs.action_space, config.network.rnn_hidden_size)

    # 2.3 retrieve the model if resume = True
    if config.training.resume:
        load_path = config.training.load_path
        actor_critic, _ = torch.load(load_path)

    # 2.4 ppo optimizer
    agent = PPO(actor_critic, config.ppo.clip_param, config.ppo.epoch, config.ppo.num_mini_batch,
                config.ppo.value_loss_coef, config.ppo.entropy_coef, lr=config.training.lr,
                eps=config.training.eps, max_grad_norm=config.training.max_grad_norm)

    #################################################
    #### 3. RL network (Social agent)
    #################################################
    print("###################### 3. RL network (Social agent) ######################")

    actor_critic_social = SocialPolicy(envs.observation_space.spaces, envs.action_space,
                                       base_kwargs=social_config, base=social_config.env_config.robot.policy, meta=True)
    if not test_args.use_idm:
        actor_critic_social.load_state_dict(torch.load(test_args.social_agent, map_location=device))
    nn.DataParallel(actor_critic_social).to(device)
    dummy_env = TIntersectionRobustnessSocial()
    dummy_env.configure(config.env_config)
    rl_ob_space_social = dummy_env.observation_space.spaces
    dummy_env.close()
    for key_ in ['pretext_actions', 'pretext_infer_masks', 'pretext_masks', 'pretext_nodes',
                 'pretext_spatial_edges', 'pretext_temporal_edges', 'spatial_edges', 'true_labels']:
        rl_ob_space_social.pop(key_)
    recurrent_hidden_states_social = {'rnn': torch.zeros(env_num, human_num, social_config.network.rnn_hidden_size, device=device)}
    masks_social = torch.zeros(env_num, human_num, device=device)

    #################################################
    #### 4. Initialization for training
    #################################################
    print("###################### 4. Initialization for training ######################")

    # 4.1 initialize the environment
    obs = envs.reset()

    # 4.2. initialize rollout storage (for ego)
    if isinstance(obs, dict):
        for key in rl_ob_space:
            rollouts.obs[key][0].copy_(obs[key])
    else:
        rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # 4.4. initialize wandb (if use wandb)
    use_wandb = config.training.use_wandb
    if use_wandb:
        wandb.login(key='6640757327f8d81d892cf149af65fb5bab1d1df6')
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
            utils.update_linear_schedule(agent.optimizer, j, num_updates, config.training.lr)

        # 5.1. environment step
        # rollout the current policy for 30 steps, and store {obs, action, reward, rnn_hxs, masks, etc} to memory
        for step in range(config.ppo.num_steps):

            # 5.1.1. Sample actions (Ego)
            with torch.no_grad():
                rollouts_obs = {}
                for key in rollouts.obs:
                    rollouts_obs[key] = rollouts.obs[key][step]

                rollouts_hidden_s = {}
                for key in rollouts.recurrent_hidden_states:
                    rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]

                value, action, action_log_prob, recurrent_hidden_states = \
                    actor_critic.act(rollouts_obs, rollouts_hidden_s, rollouts.masks[step])

            # 5.1.2. Sample actions (Social)
            if test_args.use_idm:
                # num of env, num of social agents
                # Note that -1 is invalid action. (It doesn't use at all)
                action_social = - torch.ones([env_num, 12], dtype=torch.int).to(device) * 99
            else:
                with torch.no_grad():
                    _, action_social, _, recurrent_hidden_states_social = \
                        actor_critic_social.act(obs, recurrent_hidden_states_social, masks_social, deterministic=True)
            # 5.1.3. Rendering the environment (if render)
            if config.training.render:
                envs.render()

            # 5.1.4. Observe reward and next obs
            prev_obs = copy.deepcopy(obs)
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
                            "env/collision_vehicle_idm": int(isinstance(infos[env_idx]['info'], Collision)) if infos[env_idx]['idm_or_rl'] else None,
                            "env/collision_vehicle_rl": int(isinstance(infos[env_idx]['info'], Collision)) if not infos[env_idx]['idm_or_rl'] else None
                        })
                        env_iter += 1
                    episode_rewards_env[env_idx] = 0.0

            if not test_args.use_pretrained_encoder:
                # 5.1.6. save the data for encoder
                pretext_nodes = np.concatenate((prev_obs['pretext_nodes'].to('cpu').numpy(), prev_obs['pretext_actions'].to('cpu').numpy()), axis=-1)
                data['pretext_nodes'].append(copy.deepcopy(pretext_nodes))
                data['pretext_spatial_edges'].append(copy.deepcopy(prev_obs['pretext_spatial_edges'].to('cpu').numpy()))
                data['pretext_temporal_edges'].append(copy.deepcopy(prev_obs['pretext_temporal_edges'].to('cpu').numpy()))
                data['labels'].append(copy.deepcopy(prev_obs['true_labels'].to('cpu').numpy()))
                data['pretext_masks'].append(copy.deepcopy(prev_obs['pretext_masks'].to('cpu').numpy()))
                data['dones'].append(copy.deepcopy(done))

                # save data to data_list for every 20 steps
                if len(data['labels']) == config.pretext.num_steps:
                    # process traj, keep the last sub-traj of non-dummy human in each traj
                    processed_data = process_traj(data, enc_save_key, env_num, config.pretext.num_steps, human_num)
                    data_list.extend(copy.deepcopy(processed_data))
                    data.clear()
                    for key in enc_key:
                        data[key] = []

            # 5.1.7. save the data for RL network (Ego agent)
            obs_rl = {}
            for key_ in ['robot_node', 'spatial_edges', 'temporal_edges']:
                obs_rl[key_] = copy.deepcopy(obs[key_])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs_rl, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

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

        # 5.1.9. initialize loss value as None (for dummy of wandb log)
        latent_loss, latent_recon_loss, latent_kl_loss = None, None, None
        value_loss, actor_loss, dist_entropy = None, None, None

        # 5.2. training inference net (for Ego) evey 10 step.
        if (j + 1) % 10 == 0 and (not test_args.use_pretrained_encoder) and (not test_args.ego_start_baseline):
            data_generator = makeDataset(train_data=data_list, batch_size=config.pretext.batch_size)
            loss_ep, act_loss_ep, kl_loss_ep = [0.0], [0.0], [0.0]
            for ep in range(config.pretext.epoch_num):

                for n_iter, (robot_node, spatial_edges, temporal_edges, labels, seq_len) in enumerate(data_generator):
                    robot_node = robot_node.float().to(device)  # [batch_size * seq_len * feat_dim]
                    spatial_edges = spatial_edges.float().to(device)  # [batch_size * seq_len * feat_dim]
                    temporal_edges = temporal_edges.float().to(device)  # [batch_size * seq_len * feat_dim]
                    labels = labels.float().to(device)  # [batch_size]

                    # initialize rnn hidden state: encoder / decoder
                    rnn_hxs_encoder = {'rnn': torch.zeros(config.pretext.batch_size,
                                                          config.network.rnn_hidden_size, device=device)}
                    rnn_hxs_decoder = {'rnn': torch.zeros(config.pretext.batch_size,
                                                          config.network.rnn_hidden_size, device=device)}

                    inference_model.zero_grad()
                    optimizer.zero_grad()

                    # pretext nodes: [px], pretext_spatial_edges: relative [delta_px, delta_vx] with front car & ego car
                    # pretext_temporal_edges: [vx]
                    robot_node = robot_node[:, :, 0, None]  # remove ax since the pretext nodes are [px, ax] in dataset
                    state_dict = {'pretext_nodes': robot_node, 'pretext_spatial_edges': spatial_edges,
                                  'pretext_temporal_edges': temporal_edges}
                    # joint_states = torch.cat((robot_node, temporal_edges, spatial_edges), dim=-1)
                    joint_states = torch.cat((robot_node, spatial_edges[:, :, 0, None]), dim=-1)
                    # z_mean latent variable with length = 2
                    pred_traj, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = \
                        inference_model(state_dict, rnn_hxs_encoder, rnn_hxs_decoder, seq_len)
                    loss, act_loss, kl_loss = loss_func.forward(joint_states, pred_traj, z_mean, z_log_var, seq_len)

                    loss.backward()
                    optimizer.step()
                    loss_ep.append(loss.item())
                    act_loss_ep.append(act_loss.item())
                    kl_loss_ep.append(kl_loss.item())

            latent_loss = sum(loss_ep) / len(loss_ep)
            latent_recon_loss = sum(act_loss_ep) / len(act_loss_ep)
            latent_kl_loss = sum(kl_loss_ep) / len(kl_loss_ep)
            data_list = []

        # 5.3. training RL net (for Ego) when inference net is not trained.
        else:
            # 5.3.1. calculate predicted value from value network
            with torch.no_grad():
                rollouts_obs = {}
                for key in rollouts.obs:
                    rollouts_obs[key] = rollouts.obs[key][-1]

                rollouts_hidden_s = {}
                for key in rollouts.recurrent_hidden_states:
                    rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
                next_value = actor_critic.get_value(rollouts_obs, rollouts_hidden_s, rollouts.masks[-1]).detach()

            # 5.3.2. compute return from next_value
            rollouts.compute_returns(next_value, config.ppo.use_gae, config.env_config.reward.gamma,
                                     config.ppo.gae_lambda, config.training.use_proper_time_limits)

            # 5.3.3. use ppo loss to do backprop on network parameters
            value_loss, actor_loss, dist_entropy, _ = agent.update(rollouts)
            rollouts.after_update()  # clear the rollout storage since ppo is on-policy

        # 5.5. logging & saving the model.
        # 5.5.1. wandb log if use wandb (for training)
        if use_wandb:
            wandb.log({
                "train_step": train_iter,
                "train/latent_loss": latent_loss,
                "train/latent_recon_loss": latent_recon_loss,
                "train/latent_kl_loss": latent_kl_loss,
                "train/ego_value_loss": value_loss,
                "train/ego_actor_loss": actor_loss,
                "train/ego_dist_entropy": dist_entropy,
            })
            train_iter += 1

        # 5.5.2. save the model for every interval-th episode or for the last epoch
        if (j % config.training.save_interval == 0 or j == num_updates - 1):
            save_path = os.path.join(config.training.output_dir, 'checkpoints')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if j == 0 or j == num_updates - 1:
                torch.save(actor_critic_social.state_dict(), os.path.join(save_path, 'Social_' + '%.5i' % j + ".pt"))            
                torch.save(inference_model.state_dict(), os.path.join(save_path, 'Encoder_' + '%.5i' % j + ".pt"))
            torch.save(actor_critic.state_dict(), os.path.join(save_path, 'Ego_' + '%.5i' % j + ".pt"))

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
            df = pd.DataFrame({'misc/nupdates': [j],
                               'misc/total_timesteps': [total_num_steps],
                               'fps': int(total_num_steps / (end - start)),
                               'eprewmean': [np.mean(episode_rewards)],
                               'loss/policy_entropy': dist_entropy,
                               'loss/policy_loss': actor_loss,
                               'loss/value_loss': value_loss})

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
