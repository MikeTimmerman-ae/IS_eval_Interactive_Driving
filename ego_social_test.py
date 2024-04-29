import os
import sys
import copy
import logging
import argparse
from importlib import import_module
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from driving_sim.envs import *
from driving_sim.utils.info import *
from driving_sim.vec_env.vec_pretext_normalize import VecPretextNormalize
from rl.envs import make_vec_envs
from rl.evaluation import evaluate
from rl.rl_models.policy import Policy
from rl.rl_models_social.policy import SocialPolicy
from pretext.data_loader import makeDataset_objective
from pretext.pretext_models.cvae_model import CVAEIntentPredictor


def main():

    #################################################
    #### 0. Experiment Setup
    #################################################

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--num_eval', type=int, default=int(3))
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--make_video', type=bool, default=False)

    parser.add_argument('--model_dir', type=str, default='data/experiment_1/rl_ego_normal-13')
    parser.add_argument('--test_model_ego', type=str, default='Ego_27776.pt')
    parser.add_argument('--test_model_encoder', type=str, default='Encoder_27776.pt')

    parser.add_argument('--model_dir_social', type=str, default='data/rl_social_gmeta')
    parser.add_argument('--test_model_social', type=str, default='Social_27776.pt')
    parser.add_argument('--social_meta', type=bool, default=True)
    parser.add_argument('--social_idm', type=bool, default=False)                            # for social agent
    ### Batch evaluation under different evaluation distributions using importance sampling
    parser.add_argument('--eval_type', default=None)            # Specify which type of evaluation
    parser.add_argument('--naturalistic_dist', default=None)    # Specify in case of eval under naturalistic dist.
    parser.add_argument('--mean_eval', default=None)            # Specify in case of eval under IS distribution
    parser.add_argument('--std_eval', default=None)
    ################
    test_args = parser.parse_args()

    use_idm = test_args.social_idm
    num_eval = int(test_args.num_eval)
    visualize = test_args.visualize
    make_video = test_args.make_video

    if make_video:
        import pyglet
        import imageio
        from PIL import Image
        filenames = []

    try:
        test_args.model_dir = test_args.model_dir.replace('.', '')
        model_dir_string = test_args.model_dir.replace('/', '.') + '.configs.ego_config.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
    except:
        print('Failed to get Config function from ', test_args.model_dir)
        from configs.ego_config import Config
    config = Config()

    if not use_idm:
        try:
            model_dir_string = test_args.model_dir_social.replace('/', '.') + '.configs.social_config.config'
            model_arguments = import_module(model_dir_string)
            SocialConfig = getattr(model_arguments, 'SocialConfig')
        except:
            print('Failed to get Config function from ', test_args.model_dir)
            from configs.social_config import SocialConfig
        social_config = SocialConfig()

    torch.manual_seed(config.env_config.env.seed)
    torch.cuda.manual_seed_all(config.env_config.env.seed)
    if config.training.cuda:
        if config.training.cuda_deterministic:  # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:  # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    torch.set_num_threads(1)
    device = torch.device("cuda" if config.training.cuda else "cpu")
    print("Device: ", device)
    logging.info('Create other envs with new settings')

    # find the checkpoint to be tested
    print('-----------------------')
    load_path = os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model_ego)
    print(f'Ego Agent        : {load_path}')
    load_path_encoder = os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model_encoder)
    print(f'Ego Agent Encoder: {load_path_encoder}')
    if use_idm:
        print('Social Agent      : IDM')
    else:
        load_path_social = os.path.join(test_args.model_dir_social, 'checkpoints', test_args.test_model_social)
        print(f'Social Agent     : {load_path_social}')
    print('-----------------------')

    if test_args.eval_type == "IS":
        print(f'Evaluation Distribution      : N({test_args.mean_eval}, {test_args.std_eval}) ')
    elif test_args.eval_type == "naturalistic":
        print(f'Evaluation Distribution      : Naturalistic Distribution {test_args.naturalistic_dist}')
    print('-----------------------')

    eval_dir = os.path.join(test_args.model_dir, 'eval')
    print(f'Write evaluation results to: {eval_dir}')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    #################################################
    #### 1. Environment
    #################################################
    num_env = 1
    num_human = config.env_config.car.max_veh_num
    if use_idm:
        if test_args.model_dir.split('_')[-1] == 'morton':
            config.env_config.env.env_name = 'TIntersectionPredictFrontAct-v0'
        else:
            config.env_config.env.env_name = 'TIntersectionPredictFront-v0'
    else:
        config.env_config.env.env_name = 'TIntersectionRobustnessSocial-v0'
    envs = make_vec_envs(config.env_config.env.env_name, config.env_config.env.seed, num_env,
                         config.env_config.reward.gamma, None, device, allow_early_resets=True, config=config,
                         mean=test_args.mean_eval, std=test_args.std_eval, nat=test_args.naturalistic_dist)
    envs.nenv = num_env

    #################################################
    #### 2. Inference network (Ego agent)
    #################################################
    if test_args.model_dir.split('_')[-1] == 'morton':
        inference_model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                                              decoder_base='mlp', config=config)
    elif test_args.model_dir.split('_')[-1] == 'xiaobai':
        from pretext.pretext_models.cvae_model_xma import CVAEIntentPredictor
        inference_model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                                              decoder_base='lstm', config=config)
    else:
        from pretext.pretext_models.cvae_model import CVAEIntentPredictor
        inference_model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                                              decoder_base='lstm', config=config)
    inference_model.load_state_dict(torch.load(load_path_encoder, map_location=device))
    envs.pred_model = inference_model.encoder
    envs.pred_model.eval()

    #################################################
    #### 3. RL network (Ego agent)
    #################################################
    actor_critic = Policy(envs.observation_space.spaces, envs.action_space,
                          base_kwargs=config, base=config.env_config.robot.policy)
    actor_critic.load_state_dict(torch.load(load_path, map_location=device))
    actor_critic.base.nenv = num_env
    recurrent_hidden_states = {'rnn': torch.zeros(num_env, 1, config.network.rnn_hidden_size, device=device)}
    masks = torch.ones(num_env, 1, device=device)

    #################################################
    #### 4. RL network (Social agent)
    #################################################
    if not use_idm:
        actor_critic_social = SocialPolicy(envs.observation_space.spaces, envs.action_space,
                                           base_kwargs=social_config, base=social_config.env_config.robot.policy,
                                           meta=test_args.social_meta)
        actor_critic_social.load_state_dict(torch.load(load_path_social, map_location=device))
        actor_critic_social.base.nenv = num_env
        recurrent_hidden_states_social = {'rnn': torch.zeros(num_env, num_human, social_config.network.rnn_hidden_size, device=device)}
        masks_social = torch.zeros(num_env, num_human, device=device)

    #################################################
    #### 5. Main evaluation loop start
    #################################################
    exp_results = {key_: [] for key_ in ['success', 'collision', 'time_out', 'time_to_success', 'betas', 'num_cars']}
    envs.venv.venv.envs[0].env.always_rl_social = True

    import time
    # scenarios = [0, 12, 18, 30, 32, 33, 37, 41, 43, 51, 52, 54, 59, 60, 61, 64, 69, 71, 72, 73, 74, 81, 83, 87, 96, 98, 104, 107, 123, 127, 129, 130, 139, 141, 149, 155, 159, 160, 162, 167, 168, 182, 187, 192]
    # scenarios = [41]
    # scenarios = [107, 123, 127, 129, 130, 139, 141, 149, 155, 159, 160, 162, 167, 168, 182, 187, 192]

    # selected_scenario_video_save = [43]
    # scenarios = [71, 74, 104] ## 104 good
    for eval_idx in tqdm(range(num_eval)):

        # envs.venv.venv.envs[0].seed(eval_idx+1)
        # print(scenarios[eval_idx])
        envs.venv.venv.envs[0].env.set_seed(eval_idx)
        # envs.venv.venv.envs[0].env._set_seed(scenarios[eval_idx])

        # if scenarios[eval_idx] != 129:
        #     continue

        obs = envs.reset()
        if not use_idm:
            masks_social = copy.deepcopy(obs['pretext_masks'])

        env_t = 0

        if isinstance(envs, VecPretextNormalize):
            envs.step_counter = 0
            envs.clear_pred_inputs_eval()

        while True:
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = \
                    actor_critic.act(obs, recurrent_hidden_states, masks, deterministic=True)

                if not use_idm:
                    _, action_social, _, recurrent_hidden_states_social = \
                        actor_critic_social.act(obs, recurrent_hidden_states_social, masks_social, deterministic=True)

            if visualize:
                envs.render()

            if make_video and scenarios[eval_idx] in selected_scenario_video_save:
                filename = f'img_{env_t}.png'
                filenames.append(filename)
                pyglet.image.get_buffer_manager().get_color_buffer().save(filename)

            if use_idm:
                obs, reward, done, infos = envs.step(action)
            else:
                action_all = torch.cat((action, action_social), dim=-1)
                obs, reward, done, infos = envs.step(action_all)
                masks_social = copy.deepcopy(obs['pretext_masks'])

            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float, device=device)
            env_t += 1

            if done[0]:
                print(infos[0]['info'])

                # Save beta parameters of current episode
                exp_results['betas'].append(infos[0]['betas'])
                exp_results['num_cars'].append(infos[0]['num_cars'])

                # Log success, failure and time-out
                if isinstance(infos[0]['info'], ReachGoal):
                    exp_results['success'].append(eval_idx)
                    exp_results['time_to_success'].append(env_t)
                elif isinstance(infos[0]['info'], Collision):
                    exp_results['collision'].append(eval_idx)
                elif isinstance(infos[0]['info'], Timeout):
                    exp_results['time_out'].append(eval_idx)
                else:
                    raise ValueError("unknown infos")
                break

        if make_video and scenarios[eval_idx] in selected_scenario_video_save:
            paths = [Image.open(filename) for filename in filenames]
            imageio.mimsave(f'test_video_{eval_idx}.gif', paths, format='GIF', fps=10)
            for filename in filenames:
                os.remove(filename)
            filenames = []

    #################################################
    #### 6. Logging
    #################################################
    folder_idx = 0
    while True:
        eval_results_dir = os.path.join(eval_dir, 'test' + str(folder_idx))
        if not os.path.exists(eval_results_dir):
            os.mkdir(eval_results_dir)
            np.savetxt(os.path.join(eval_results_dir, 'betas.csv'), np.array(exp_results['betas']))
            np.savetxt(os.path.join(eval_results_dir, 'successes.csv'), np.array(exp_results['success']))
            np.savetxt(os.path.join(eval_results_dir, 'num_cars.csv'), np.array(exp_results['num_cars']))
            with open(os.path.join(eval_results_dir, 'results.txt'), 'w') as f:

                ###### 6.1. Model directory
                print(f'Ego Agent            : {load_path}', file=f)
                print(f'Ego Agent Encoder: {load_path_encoder}', file=f)
                if use_idm:
                    print('Social Agent          : IDM', file=f)
                else:
                    print(f'Social Agent         : {load_path_social}', file=f)
                print('-----------------------', file=f)

                ###### 6.2. Evaluation distributions
                if test_args.eval_type == "IS":
                    print(f'Evaluation Distribution      : N({test_args.mean_eval}, {test_args.std_eval}) ', file=f)
                elif test_args.eval_type == "naturalistic":
                        print(f'Evaluation Distribution      : Naturalistic Distribution {test_args.naturalistic_dist}', file=f)
                print('-----------------------', file=f)

                ###### 6.3. Experimental results (Rates)
                print(f'Total        : {num_eval}', file=f)
                print(f'Success    : {len(exp_results["success"]) / num_eval:.3f} | '
                      f'({len(exp_results["success"])} / {num_eval})', file=f)
                print(f'Collision   : {len(exp_results["collision"]) / num_eval:.3f} | '
                      f'({len(exp_results["collision"])} / {num_eval})', file=f)
                print(f'Time Out  : {len(exp_results["time_out"]) / num_eval:.3f} | '
                      f'({len(exp_results["time_out"])} / {num_eval})', file=f)
                print('-----------------------', file=f)

                ###### 6.4. Experimental results (Scenario ID)
                print(f'Success Case    : {exp_results["success"]}', file=f)
                print(f'Collision Case   : {exp_results["collision"]}', file=f)
                print(f'Time Out Case  : {exp_results["time_out"]}', file=f)
                print(f'Success - Time  : {exp_results["time_to_success"]}', file=f)

            break
        else:
            folder_idx += 1

    print("---------------------------------------------------")
    print(f'Success   : {len(exp_results["success"]) / num_eval:.3f}')
    print(f'Collision : {len(exp_results["collision"]) / num_eval:.3f}')
    print(f'Time Out  : {len(exp_results["time_out"]) / num_eval:.3f}')
    print("---------------------------------------------------")


if __name__ == '__main__':
    main()