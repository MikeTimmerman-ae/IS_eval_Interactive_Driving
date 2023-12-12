import torch
import torch.nn as nn
import torch.optim as optim

'''
The PPO RL optimizer
'''


class PPO_Social():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 actor_critic_backup=None):

        self.actor_critic = actor_critic
        self.actor_critic_backup = actor_critic_backup

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.aux_loss_func=nn.MSELoss()
        self.aux_loss_scalar=1.

    def update(self, rollouts):
        valid_data = rollouts.obs['valid_training'][:-1] == 1.0
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages[torch.logical_not(valid_data)] = 0.0
        advantages[valid_data] = (advantages[valid_data] - advantages[valid_data].mean()) / (advantages[valid_data].std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        aux_loss_epoch = 0
        offline_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for online_sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = online_sample
                valid_data = obs_batch['valid_training'] == 1.0
                # Reshape to do in a single forward pass for all steps

                values, action_log_probs, dist_entropy, _, _, _, _, action_dist = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                if self.actor_critic_backup:
                    _, _, _, _, _, _, condition, action_dist_target = self.actor_critic_backup.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                    policy_kl = (action_dist_target * (action_dist_target / action_dist).log()).sum(-1)
                    regularization_loss = policy_kl[torch.logical_and(condition.reshape(-1), masks_batch.reshape(-1))].sum()
                else:
                    regularization_loss = torch.tensor(0.0)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2)[valid_data].mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)[valid_data].mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + 0.01 * regularization_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                offline_loss_epoch += regularization_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        aux_loss_epoch /= num_updates
        offline_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, aux_loss_epoch, offline_loss_epoch
