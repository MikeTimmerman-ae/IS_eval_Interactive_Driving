import os
from os.path import join as pjoin
import time
import argparse
from importlib import import_module

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl.envs import make_vec_envs
from pretext.data_loader import loadDataset, loadDataset_objective
from pretext.pretext_models.cvae_model import CVAEIntentPredictor


def main():

    # the following parameters will be determined for each test run
    parser = argparse.ArgumentParser('Parse configuration file')
    # parser.add_argument('--model_dir', type=str, default='data/20230504_SPPO_part12_ego_-11_init_time0')
    parser.add_argument('--model_dir', type=str, default='data/new_pretext')
    parser.add_argument('--test_model_encoder', type=str, default='24.pt')
    parser.add_argument('--is_idm', type=bool, default=False)
    # parser.add_argument('--data_load_dir', type=str, default='data_sl/230421_-2to3_data/new_dataset')
    # parser.add_argument('--data_load_dir', type=str, default='data_sl/idm/new_dataset')
    parser.add_argument('--data_load_dir', type=str, default='data_sl/20230510_SPPO_part1_-13_discrete/new_dataset')
    test_args = parser.parse_args()

    use_lim = False
    x_lim = (-13.709918785095216, 0.8389645695686341)
    y_lim = (-6.739149856567383, 3.486802864074707)
    # import config class from saved directory
    # if not found, import from the default directory
    model_dir_temp = test_args.model_dir
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.ego_config.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
    except:
        print('Failed to get Config function from ', test_args.model_dir)
        from configs.ego_config import Config
    config = Config()
    config.env_config.env.env_name = 'TIntersectionRobustnessSocial-v0'

    torch.set_num_threads(1)
    device = torch.device("cuda" if config.training.cuda else "cpu")

    fig_dir = pjoin(test_args.model_dir, 'fig')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    #################################################
    #### 1. Inference network (Ego agent)
    #################################################
    print('-----------------------')
    load_path_encoder = pjoin(test_args.model_dir, 'checkpoints', test_args.test_model_encoder)
    print(f'Ego agent encoder: {load_path_encoder}')
    print('-----------------------')
    envs = make_vec_envs(config.env_config.env.env_name, config.env_config.env.seed, 1,
                         config.env_config.reward.gamma, None, device, allow_early_resets=True, config=config)
    inference_model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                                          decoder_base='lstm', config=config)
    inference_model.load_state_dict(torch.load(load_path_encoder, map_location=device))
    envs.close()

    #################################################
    #### 2. ..
    #################################################
    x_list, y_list, label_list = [], [], []
    batch_size = 2000
    inference_model.decoder.nenv = batch_size
    if test_args.is_idm:
        data_generator = loadDataset(train_data=False, batch_size=batch_size, num_workers=0,
                                     drop_last=True, load_dir=test_args.data_load_dir)
    else:
        data_generator = loadDataset_objective(train_data=False, batch_size=batch_size, num_workers=0,
                                               drop_last=True, load_dir=test_args.data_load_dir)

    time.sleep(0.1)  # to make progress bar visualization stable
    pbar = tqdm(total=len(data_generator))
    for n_iter, data_minibatch in enumerate(data_generator):
        if test_args.is_idm:
            robot_node, spatial_edges, temporal_edges, labels, seq_len = data_minibatch
            labels = labels.float().to(device)                          # [batch_size]
        else:
            robot_node, spatial_edges, temporal_edges, labels, obj_weight, seq_len = data_minibatch
            obj_weight = obj_weight.to('cpu').numpy()                   # [batch_size * 2]

        # robot_node = robot_node.float().to(device)[:, :, 0, None]     # [batch_size * seq_len * feat_dim]
        # spatial_edges = spatial_edges.float().to(device)              # [batch_size * seq_len * feat_dim]
        # temporal_edges = temporal_edges.float().to(device)            # [batch_size * seq_len * feat_dim]

        # 1. accelerate
        accelerate_edges = torch.zeros((2000, 20, 1))
        accelerate_edges[:, 1:] = temporal_edges[:, 1:] - temporal_edges[:, :-1]
        temporal_edges = torch.cat([temporal_edges, accelerate_edges], -1)

        # 2. jerk
        # jerk_edges = torch.zeros((2000, 20, 1))
        # jerk_edges[:, 2:] = temporal_edges[:, 2:, 1:] - temporal_edges[:, 1:-1, 1:]
        # temporal_edges = torch.cat([temporal_edges, jerk_edges], -1)

        state_dict = {'pretext_nodes': robot_node.float().to(device)[:, :, 0, None],
                      'pretext_spatial_edges': spatial_edges.float().to(device),
                      'pretext_temporal_edges': temporal_edges.float().to(device)}

        # initialize rnn hidden state of encoder
        rnn_hxs_encoder = {'rnn': torch.zeros(batch_size, config.network.rnn_hidden_size, device=device)}
        with torch.no_grad():
            z_mean, z_log_var, _ = inference_model.encoder(state_dict, rnn_hxs_encoder, seq_len)
            z = inference_model.reparameterize(z_mean, z_log_var).to('cpu').numpy()

        x_list.append(z[:, 0])
        y_list.append(z[:, 1])
        if test_args.is_idm:
            label_list.append(labels)
        else:
            label_list.append(obj_weight[:, 1])
        pbar.update(1)

    pbar.close()
    x_list = np.concatenate(x_list)
    y_list = np.concatenate(y_list)
    label_list = np.concatenate(label_list)

    if test_args.is_idm:
        conditions = [
            label_list < 10,
            label_list == 1,
            label_list == 0,
        ]
        conditions_str = ['idm1_all', 'idm2_con', 'idm3_agg']
    else:
        conditions = [
            label_list < 10,
            np.logical_and(-3.0 <= label_list, label_list < -2.0),
            np.logical_and(-2.0 <= label_list, label_list < -1.0),
            np.logical_and(-1.0 <= label_list, label_list < -0.0),
            np.logical_and(0.0 <= label_list, label_list < 1.0),
            np.logical_and(1.0 <= label_list, label_list < 2.0),
            np.logical_and(2.0 <= label_list, label_list < 3.0)
        ]
        conditions_str = ['all', '-3', '-2', '-1', '0', '1', '2']

    for c_iter, cond in enumerate(conditions):
        print("# of data : ", len(x_list[cond]))
        if test_args.is_idm:
            x_, y_, trait_ = x_list[cond], y_list[cond], label_list[cond]
            if (trait_ == 1).sum() > 0:
                plt.scatter(x_[trait_ == 1], y_[trait_ == 1], c=trait_[trait_ == 1],
                            s=0.5, vmin=0., vmax=1., cmap='bwr_r', label='conservative')
            if (trait_ == 0).sum() > 0:
                plt.scatter(x_[trait_ == 0], y_[trait_ == 0], c=trait_[trait_ == 0],
                            s=0.5, vmin=0., vmax=1., cmap='bwr_r', label='aggressive')
            lgnd = plt.legend(loc=2, fontsize=13)
            print(len(lgnd.legendHandles))
            for i in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[i]._sizes = [20]
            cbar = plt.colorbar()
        else:
            plt.scatter(x_list[cond], y_list[cond], c=label_list[cond], s=0.5, vmin=-3., vmax=3., cmap='turbo_r')
            cbar = plt.colorbar()
            cbar.set_label('β (preference)')

        if use_lim:
            plt.xlim(*x_lim)
            plt.ylim(*y_lim)
        else:
            plt.xlim(x_list.min() - 0.3, x_list.max() + 0.3)
            plt.ylim(y_list.min() - 0.3, y_list.max() + 0.3)

        if test_args.is_idm:
            fig_path = pjoin(fig_dir, f'latent_space_{conditions_str[c_iter]}.png')
        else:
            fig_path = pjoin(fig_dir, f'latent_space_{c_iter}_{conditions_str[c_iter]}.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        print(f'Image is saved in: {fig_path}')

        if c_iter == len(conditions) - 1:
            print(f'X lim: {plt.xlim()}')
            print(f'Y lim: {plt.ylim()}')
        plt.close('all')


if __name__ == '__main__':
    main()