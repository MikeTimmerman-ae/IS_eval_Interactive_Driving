import pickle
import copy
import os
import numpy as np
import torch
import torch.nn as nn

from driving_sim.envs import *
from rl.envs import make_vec_envs
from rl.ppo import PPO_Social
from rl.rl_models_social.policy import SocialPolicy
from configs.social_config.config import SocialConfig
'''
collect trajectory data of other cars in the env and the label of the cars' traits
make sure each trajectory has and only has one car

device: cpu or cuda0
train_data: True if collect training data, False if collect testing data
config: config object
'''


def collectMany2OneData(device, train_data, config, social_policy_dir):
    # always use 'TIntersectionPredictFrontAct-v0', since the observation is compatible for both our method and Morton baseline
    env_name = 'TIntersectionRobustnessSocial-v0'

    env_num = 1 if config.pretext.render else config.pretext.num_processes
    human_num = config.env_config.car.max_veh_num
    envs = make_vec_envs(env_name, config.env_config.env.seed, env_num,
                         config.env_config.reward.gamma, None, device, allow_early_resets=True, config=config,
                         wrap_pytorch=False)

    social_config = SocialConfig()
    social_config.training.num_processes = config.training.num_processes
    social_config.ppo.num_mini_batch = config.ppo.num_mini_batch
    social_config.ppo.num_steps = config.ppo.num_steps
    actor_critic_social = SocialPolicy(envs.observation_space.spaces, envs.action_space,
                                       base_kwargs=social_config, base=social_config.env_config.robot.policy)
    actor_critic_social.load_state_dict(torch.load(social_policy_dir, map_location=device))

    dummy_env = TIntersectionRobustnessSocial()
    dummy_env.configure(config.env_config)
    rl_ob_space_social = dummy_env.observation_space.spaces
    dummy_env.close()
    for key_ in ['pretext_actions', 'pretext_infer_masks', 'pretext_masks', 'pretext_nodes',
                 'pretext_spatial_edges', 'pretext_temporal_edges', 'spatial_edges', 'true_labels']:
        rl_ob_space_social.pop(key_)
    recurrent_hidden_states_social = {
        'rnn': torch.zeros(env_num, human_num, social_config.network.rnn_hidden_size, device=device)
    }
    masks_social = torch.zeros(env_num, human_num, device=device)
    actor_critic_social.load_state_dict(torch.load(social_policy_dir, map_location=device))
    nn.DataParallel(actor_critic_social).to(device)

    # key list for observation from env
    ob_key_list = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'labels', 'pretext_masks', 'dones', 'objective_weight']
    # key list for saved data
    save_key_list = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'labels', 'objective_weight']

    # collect data for pretext training
    # list of dicts, the value of each key is a list of 30
    data_list = [] # list for all data collected
    data = {} # buffer for data from env
    # initialize buffer to store data from env
    # each data[key] = list of traj_len, each element of the list = array (nenv, human_num, ?)
    for key in ob_key_list:
        data[key] = []

    obs = envs.reset()

    # 1 epoch -> 1 file
    for epoch in range(10):
        print('collect data epoch', epoch)
        # how may traj do we want in one file
        # number of collected traj in a file will be >= config.pretext.num_data_per_file
        while(len(data_list)) < config.pretext.num_data_per_file:

            if config.pretext.render:
                envs.render()

            # NOTE: the robot doesn't move!
            action = np.zeros((env_num, 1), dtype=int)

            # 5.1.2. Sample actions (Social)
            with torch.no_grad():
                obs_rl = dict()
                for key, obs_ in obs.items():
                    obs_rl[key] = torch.tensor(obs_)

                _, action_social, _, recurrent_hidden_states_social = \
                    actor_critic_social.act(obs_rl, recurrent_hidden_states_social, masks_social, deterministic=True)
                action_social = action_social.numpy()

            # save the previous obs before it is overwritten
            prev_obs = copy.deepcopy(obs)

            action_all = np.concatenate((action, action_social), axis=-1)
            obs, reward, done, infos = envs.step(action_all)

            # If done then clean the history of observations.
            masks_social = torch.tensor(copy.deepcopy(obs['pretext_masks']))
            for env_idx, done_ in enumerate(done):
                if done_:
                    masks_social[env_idx] = 0.0

            # pretext node: [px, ax] of other cars
            pretext_nodes = np.concatenate((prev_obs['pretext_nodes'], prev_obs['pretext_actions']), axis=-1)
            data['pretext_nodes'].append(copy.deepcopy(pretext_nodes))
            data['pretext_spatial_edges'].append(copy.deepcopy(prev_obs['pretext_spatial_edges']))
            data['pretext_temporal_edges'].append(copy.deepcopy(prev_obs['pretext_temporal_edges']))
            data['labels'].append(copy.deepcopy(prev_obs['true_labels']))
            data['pretext_masks'].append(copy.deepcopy(prev_obs['pretext_masks']))
            data['dones'].append(copy.deepcopy(done))
            data['objective_weight'].append(copy.deepcopy(prev_obs['objective_weight']))

            # save data to data_list for every 20 steps
            if len(data['labels']) == config.pretext.num_steps:
                # process traj, keep the last sub-traj of non-dummy human in each traj
                processed_data = process_traj(data, save_key_list, env_num, config.pretext.num_steps, human_num)
                data_list.extend(copy.deepcopy(processed_data))
                data.clear()
                for key in ob_key_list:
                    data[key] = []

        print('number of traj in a file:', len(data_list))
        # save observations as pickle files
        # observations is a list of dict [{'x':, 'intent':, 'u':}, ...]
        filePath = os.path.join(config.pretext.data_save_dir, 'train') if train_data \
            else os.path.join(config.pretext.data_save_dir, 'test')
        if not os.path.isdir(filePath):
            os.makedirs(filePath)
        filePath = os.path.join(filePath, str(epoch)+'.pickle')
        with open(filePath, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        data_list.clear()

    envs.close()


'''
process the observation from env, and convert then to the saving format for training

input data: dictionary of nested lists
output return value: list of dictionaries of np array, where each dict = one traj

data: traj_len steps of observation data from env, each key has value with shape [traj_len, nenv, human_num, ?]
save_key_list: list of observation keys to be saved
nenv: number of parallel env
traj_len: max traj length of each traj, slice the input traj data into traj with length = traj_len
human_num: max number of other cars in env
'''
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
                    elif key == 'objective_weight':
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
