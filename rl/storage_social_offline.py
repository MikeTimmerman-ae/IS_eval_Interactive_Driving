import torch
import numpy as np


class RolloutStorageSocialOffline(object):
    def __init__(self, offline_data):
        self.offline_data = offline_data

        self.obs = [off_data_elem[0] for off_data_elem in self.offline_data]
        self.actions = [off_data_elem[1] for off_data_elem in self.offline_data]
        self.masks = [off_data_elem[2] for off_data_elem in self.offline_data]
        self.veh_idx = [off_data_elem[3] for off_data_elem in self.offline_data]

        self.num_data = len(offline_data)
        self.rnn_hidden_dim = 256

    def sample_data(self, num_sample):
        sample_idx = np.random.choice(self.num_data, num_sample, replace=False)

        obs = dict()
        obs_ = [self.obs[idx] for idx in sample_idx]
        for key in obs_[0].keys():
            obs[key] = torch.stack([obs_[i][key] for i in range(num_sample)])
        actions = torch.stack([self.actions[idx] for idx in sample_idx])
        masks = torch.stack([self.masks[idx] for idx in sample_idx])
        veh_idx = torch.tensor([self.veh_idx[idx] for idx in sample_idx])
        rnn_hxs = torch.zeros(num_sample, 1, self.rnn_hidden_dim)

        return obs, rnn_hxs, actions, masks, veh_idx
