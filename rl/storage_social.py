import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        for key in _tensor:
            _tensor[key] = _tensor[key].view(T * N, *(_tensor[key].size()[2:]))
        return _tensor
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])

'''
Class for rollout memory that is collected from the environment at each RL epoch
The experience that is stored in the rollout memory is used to compute policy gradient in PPO, 
and is thrown away after each epoch since PPO is on-policy
'''
class RolloutStorageSocial(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):

        if isinstance(obs_shape, dict):
            self.obs = {}
            for key in obs_shape:
                self.obs[key] = torch.zeros(num_steps + 1, num_processes, *(obs_shape[key].shape))
            self.human_num = obs_shape['social_car_information'].shape[0]
        else:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)

        self.recurrent_hidden_states = {}  # a dict of tuple(hidden state, cell state)
        self.recurrent_hidden_states['rnn'] = torch.zeros(num_steps + 1, num_processes,
                                                          self.human_num, recurrent_hidden_state_size)

        self.rewards = torch.zeros(num_steps, num_processes, self.human_num)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, self.human_num)
        self.returns = torch.zeros(num_steps + 1, num_processes, self.human_num)
        self.action_log_probs = torch.zeros(num_steps, num_processes, self.human_num)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = self.human_num
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, self.human_num)
        self.masks_for_reward = torch.ones(num_steps + 1, num_processes, self.human_num)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, self.human_num)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)

        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key] = self.recurrent_hidden_states[key].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.masks_for_reward = self.masks_for_reward.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, masks_for_reward, bad_masks):

        for key in self.obs:
            self.obs[key][self.step + 1].copy_(obs[key])

        for key in recurrent_hidden_states:
            self.recurrent_hidden_states[key][self.step + 1].copy_(recurrent_hidden_states[key])

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.masks_for_reward[self.step + 1].copy_(masks_for_reward)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for key in self.obs:
            self.obs[key][0].copy_(self.obs[key][-1])

        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key][0].copy_(self.recurrent_hidden_states[key][-1])

        self.masks[0].copy_(self.masks[-1])
        self.masks_for_reward[0].copy_(self.masks_for_reward[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks_for_reward[step + 1] \
                            - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks_for_reward[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * gamma * self.masks_for_reward[step + 1]
                                          + self.rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = torch.zeros(*self.value_preds.shape[1:])
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + \
                            gamma * self.value_preds[step + 1] * self.masks_for_reward[step + 1] \
                            - self.value_preds[step]
                    print("delta: ", delta.is_cuda)
                    print("gae: ", gae.is_cuda)
                    print("self.masks_for_reward[step + 1]: ", self.masks_for_reward[step + 1].is_cuda)
                    gae = delta + gamma * gae_lambda * self.masks_for_reward[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * gamma * self.masks_for_reward[step + 1] \
                                         + self.value_preds[step] * (1 - self.bad_masks[step + 1]) \
                                         + self.rewards[step] * self.bad_masks[step + 1]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = {}
            for key in self.obs:
                obs_batch[key] = self.obs[key][:-1].view(-1, *self.obs[key].size()[2:])[indices]

            recurrent_hidden_states_batch = {}
            for key in self.recurrent_hidden_states:
                recurrent_hidden_states_batch[key] = self.recurrent_hidden_states[key][:-1].view(
                -1, self.recurrent_hidden_states[key].size(-1))[indices]

            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = {}
            for key in self.obs:
                obs_batch[key] = []

            recurrent_hidden_states_batch = {}
            for key in self.recurrent_hidden_states:
                recurrent_hidden_states_batch[key] = []

            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for key in self.obs:
                    obs_batch[key].append(self.obs[key][:-1, ind])

                for key in self.recurrent_hidden_states:
                    recurrent_hidden_states_batch[key].append(self.recurrent_hidden_states[key][0:1, ind])

                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)

            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            for key in obs_batch:
                obs_batch[key] = torch.stack(obs_batch[key], 1)

            for key in recurrent_hidden_states_batch:
                temp = torch.stack(recurrent_hidden_states_batch[key], 1)
                recurrent_hidden_states_batch[key] = temp.view(N, *(temp.size()[2:]))


            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
