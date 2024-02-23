import torch.nn as nn

from rl.distributions import Bernoulli, Categorical, DiagGaussian
from rl.rl_models_social.lstm_social import LSTM_SOCIAL


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


'''
the wrapper class for all policy networks
'''


class SocialPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, meta=False):
        super(SocialPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.meta = meta
        # self.base = LSTM_SOCIAL(obs_shape, base_kwargs, 'rl')
        if meta:
            from rl.rl_models_social.lstm_gnn_social import LSTM_GNN_SOCIAL
            self.base = LSTM_GNN_SOCIAL(obs_shape, base_kwargs, 'rl')
        else:
            from rl.rl_models_social.lstm_gnn_social_multi import LSTM_GNN_SOCIAL
            self.base = LSTM_GNN_SOCIAL(obs_shape, base_kwargs, 'rl')

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        print("-masks: ", masks.is_cuda)
        if self.meta:
            print("-!masks: ", masks.is_cuda)
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
            shape_recon = actor_features.shape[:-1]
            dist = self.dist(actor_features.reshape(-1, self.base.output_size))
        else:
            print("--masks: ", masks.is_cuda)
            value, dist, shape_recon, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        action = action.reshape(shape_recon, self.base.output_size)
        action_log_probs = action_log_probs.reshape(shape_recon, self.base.output_size)
        value = value.squeeze(-1)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        if self.meta:
            value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        else:
            value, _, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        return value.squeeze(-1)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        human_node_output = None
        true_human_node = None

        if self.meta:
            value, actor_features, rnn_hxs, condition = self.base(inputs, rnn_hxs, masks)
            dist = self.dist(actor_features)
        else:
            value, dist, rnn_hxs, condition = self.base(inputs, rnn_hxs, masks)

        num_agent = action.shape[-1]
        action_log_probs = dist.log_probs(action.reshape(-1, 1))
        valid_data = inputs['valid_training'].reshape(-1) == 1.0
        dist_entropy = dist.entropy()[valid_data].mean()

        value = value.reshape(-1, num_agent)
        action_log_probs = action_log_probs.reshape(-1, num_agent)

        return value, action_log_probs, dist_entropy, rnn_hxs, human_node_output, true_human_node, condition, dist.probs
