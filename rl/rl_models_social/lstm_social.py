import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from rl.utils import init
from rl.rl_models_social.rnn_base import RNNBase, reshapeT

'''
LSTM with attention policy network 
'''
class LSTM_SOCIAL(nn.Module):
    def __init__(self, obs_space_dict, config, task):
        super(LSTM_SOCIAL, self).__init__()
        self.config = config
        self.is_recurrent = True
        self.task = task

        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch

        self.output_size = config.network.rnn_hidden_size

        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # last few layers for policy network and value network
        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

        # robot node size + robot temporal edge feature size + (one) robot spatial edge feature size
        # driving: 2 + 2 = 4
        self.robot_state_size = obs_space_dict['robot_node'].shape[1] + obs_space_dict['temporal_edges'].shape[1]

        # input size = ego car state size + one other car's state size
        input_size = 11

        # network modules
        self.mlp = nn.Sequential(
            init_(nn.Linear(input_size, 150)), nn.ReLU(),
            init_(nn.Linear(150, 200)), nn.ReLU(),
            init_(nn.Linear(200, config.network.embedding_size)), nn.ReLU())

        self.RNN = RNNBase(config)

        self.train()

    '''
    forward part for rl
    returns: output of rnn & new hidden state of rnn
    '''
    def _forward(self, inputs, rnn_hxs, masks, seq_length, nenv):

        # state processing | final shape == seq_length, nenv, human_num, 11
        social_car_information = reshapeT(inputs['social_car_information'], seq_length, nenv)  # seq_length, nenv, human_num, 3
        front_car_information = reshapeT(inputs['front_car_information'], seq_length, nenv)  # seq_length, nenv, human_num, 2
        objective_weight = reshapeT(inputs['objective_weight'], seq_length, nenv)  # seq_length, nenv, human_num, 2
        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)  # seq_length, nenv, 1, 2
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)  # seq_length, nenv, 1, 2
        robot_obs = torch.cat((robot_node, temporal_edges), dim=-1)  # seq_length, nenv, 1, 4
        robot_obs_tile = robot_obs.expand([seq_length, nenv, self.human_num, self.robot_state_size])  # seq_length, nenv, human_num, 4

        # seq_length, nenv, human_num, 11
        input_states = torch.cat((social_car_information, front_car_information, robot_obs_tile, objective_weight), dim=-1)

        hidden_states = reshapeT(rnn_hxs['rnn'], 1, nenv)  # 1, nenv, human_num, 256
        masks = reshapeT(masks, seq_length, nenv)  # seq_length, nenv, human_num

        # mlp1
        # input : [seq_length, nenv, human_num, 11]
        rnn_in = self.mlp(input_states)

        # lstm
        # outputs: x: [seq_length, nenv, 12, 256] h_new: [1, nenv, 12, 256]
        x, h_new = self.RNN._forward_gru(rnn_in, hidden_states, masks)
        # x = x[:, :, 0, :]  # [seq_length, nenv, 256]
        return x, h_new

    '''
    forward function for rl: returns critic output, actor features, and new rnn hidden state
    '''
    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv # 12

        else:
            # Training time
            seq_length = self.seq_length  # 30
            nenv = self.nenv // self.nminibatch  # 12/2 = 6
            # masks : [seq_length * nenv, 12]

        x, h_new = self._forward(inputs, rnn_hxs, masks, seq_length, nenv)
        rnn_hxs['rnn'] = h_new.squeeze(0)  # [nenv, human_num, 256]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs
