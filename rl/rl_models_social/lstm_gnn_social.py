import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from rl.utils import init
from rl.rl_models_social.rnn_base import RNNBase, reshapeT

'''
LSTM with attention policy network 
'''
class LSTM_GNN_SOCIAL(nn.Module):
    def __init__(self, obs_space_dict, config, task):
        super(LSTM_GNN_SOCIAL, self).__init__()
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

        self.actor_upper = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_upper = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear_upper = init_(nn.Linear(self.output_size, 1))

        # robot node size + robot temporal edge feature size + (one) robot spatial edge feature size
        # driving: 2 + 2 = 4
        self.robot_state_size = obs_space_dict['robot_node'].shape[1] + obs_space_dict['temporal_edges'].shape[1]

        # input size = ego car state size + one other car's state size
        self.vehicle_state_size = 6
        input_size = self.vehicle_state_size * 2

        self.edge_model_lower = nn.Sequential(
            init_(nn.Linear(input_size, 150)), nn.ReLU(),
            init_(nn.Linear(150, 50)), nn.ReLU())

        self.edge_model_upper = nn.Sequential(
            init_(nn.Linear(input_size, 150)), nn.ReLU(),
            init_(nn.Linear(150, 50)), nn.ReLU())

        self.aggregator = 'min'

        self.node_model_lower = nn.Sequential(
            init_(nn.Linear(50 + self.vehicle_state_size + 4, 150)), nn.ReLU(),
            init_(nn.Linear(150, 64)), nn.ReLU())

        self.node_model_upper = nn.Sequential(
            init_(nn.Linear(50 + self.vehicle_state_size + 4, 150)), nn.ReLU(),
            init_(nn.Linear(150, 64)), nn.ReLU())

        self.RNN = RNNBase(config)

        if config.training.cuda:
            self.RNN = self.RNN.cuda()
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
            self.critic_linear = self.critic_linear.cuda()
            self.actor_upper = self.critic_upper.cuda()
            self.critic_upper = self.critic_upper.cuda()
            self.critic_linear_upper = self.critic_linear_upper.cuda()
            self.edge_model_lower = self.edge_model_lower.cuda()
            self.edge_model_upper = self.edge_model_upper.cuda()
            self.node_model_lower = self.node_model_lower.cuda()
            self.node_model_upper = self.node_model_upper.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            

    '''
    forward part for rl
    returns: output of rnn & new hidden state of rnn
    '''
    def _forward(self, inputs, rnn_hxs, masks, seq_length, nenv):

        ### important: if you change the _forward, you have to change forward_selective properly !!

        # state processing | final shape == seq_length, nenv, human_num, 11
        social_car_information = reshapeT(inputs['social_car_information'], seq_length, nenv).to(self.device)  # seq_length, nenv, human_num, 3
        objective_weight = reshapeT(inputs['objective_weight'], seq_length, nenv).to(self.device) # seq_length, nenv, human_num, 2
        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv).to(self.device)  # seq_length, nenv, 1, 2
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv).to(self.device)  # seq_length, nenv, 1, 2
        robot_obs = torch.cat((robot_node, temporal_edges), dim=-1).to(self.device)  # seq_length, nenv, 1, 4
        robot_obs_tile = robot_obs.expand([seq_length, nenv, self.human_num, self.robot_state_size])  # seq_length, nenv, human_num, 4

        horizontal_vehicle_information = torch.cat((social_car_information,
                                                    torch.zeros((*social_car_information.size()[:-1], 1), device=self.device),
                                                    objective_weight), dim=-1)
        leftturn_vehicle_information = torch.cat((robot_obs, torch.zeros((*robot_obs.size()[:-1], 2), device=self.device)), dim=-1)
        all_vehicle_information = torch.cat((leftturn_vehicle_information, horizontal_vehicle_information), dim=-2)

        new_shape1 = (*all_vehicle_information.shape[:2], self.human_num, *all_vehicle_information.shape[2:])
        from_vehicle_info = all_vehicle_information.unsqueeze(2).expand(new_shape1)

        new_shape2 = (*horizontal_vehicle_information.shape[:3], self.human_num + 1, *all_vehicle_information.shape[3:])
        to_vehicle_info = horizontal_vehicle_information.unsqueeze(-2).expand(new_shape2)

        valid_training = reshapeT(inputs['valid_training'], seq_length, nenv)
        valid_msg = torch.ones(*valid_training.shape[:2], self.human_num + 1)
        valid_msg[:, :, 1:] = valid_training
        valid_msg = valid_msg.unsqueeze(-2).expand(new_shape2[:4])

        # step 1. make the message.
        edge_input = torch.cat((to_vehicle_info, from_vehicle_info), dim=-1)
        msg_lower = self.edge_model_lower(edge_input[:, :, :6])
        msg_upper = self.edge_model_upper(edge_input[:, :, 6:])
        msg = torch.cat((msg_lower, msg_upper), dim=2)
        msg_clipped = torch.ones_like(msg) * msg.max().item()
        msg_clipped[valid_msg == 1.0] = msg[valid_msg == 1.0]

        # step 2. aggregate the message.
        if self.aggregator == 'min':
            msg_agg = torch.min(msg, dim=-2).values

        # step 3. node update
        node_input = torch.cat([horizontal_vehicle_information, robot_obs_tile, msg_agg], dim=-1)  # [seq_length, nenv, 1, 50 + 4]
        rnn_in_lower = self.node_model_lower(node_input[:, :, :6])
        rnn_in_upper = self.node_model_upper(node_input[:, :, 6:])
        rnn_in = torch.cat((rnn_in_lower, rnn_in_upper), dim=2)

        hidden_states = reshapeT(rnn_hxs['rnn'], 1, nenv)  # 1, nenv, human_num, 256
        masks = reshapeT(masks, seq_length, nenv)  # seq_length, nenv, human_num

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

        hidden_critic_lower = self.critic(x[:, :, :6])
        critic_output_lower = self.critic_linear(hidden_critic_lower)
        actor_output_lower = self.actor(x[:, :, :6])

        hidden_critic_upper = self.critic_upper(x[:, :, 6:])
        critic_output_upper = self.critic_linear_upper(hidden_critic_upper)
        actor_output_upper = self.actor_upper(x[:, :, 6:])

        critic_output = torch.cat((critic_output_lower, critic_output_upper), dim=2)
        actor_output = torch.cat((actor_output_lower, actor_output_upper), dim=2)

        # critic_output = self.critic_linear(self.critic(x))
        # actor_output = self.actor(x)

        if infer:
            return critic_output.squeeze(0), actor_output.squeeze(0), rnn_hxs
        else:
            return critic_output.view(-1, 1), actor_output.view(-1, self.output_size), rnn_hxs, None

        # if infer:
        #     return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        # else:
        #     return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs