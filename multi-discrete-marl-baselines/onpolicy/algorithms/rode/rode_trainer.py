# RODE Trainer: PPO-style training with role selector + role-conditioned actors

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.valuenorm import ValueNorm


class RODE_Trainer():
    """PPO-style trainer for RODE policy.
    
    Handles:
    - Value loss (shared critic)
    - Policy loss (per-role actors)
    - Role selector loss (entropy-regularized)
    """
    
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.role_entropy_coef = getattr(args, 'role_entropy_coef', 0.01)

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """PPO update for role-based policy.
        
        Routes each agent's data to its assigned role's actor for policy loss.
        Also trains the role selector with entropy bonus.
        """
        if len(sample) >= 12:
            share_obs_batch = sample[0]
            obs_batch = sample[1]
            rnn_states_batch = sample[2]
            rnn_states_critic_batch = sample[3]
            actions_batch = sample[4]
            value_preds_batch = sample[5]
            return_batch = sample[6]
            masks_batch = sample[7]
            active_masks_batch = sample[8]
            old_action_log_probs_batch = sample[9]
            adv_targ = sample[10]
            available_actions_batch = sample[11]
            roles_batch = sample[12] if len(sample) > 12 else None
        else:
            return {}

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['role_loss'] = 0
        train_info['role_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['role_grad_norm'] = 0
        train_info['ratio'] = 0

        # ---- Role selection with REINFORCE signal ----
        # Use role selector to sample roles based on centralized obs
        share_obs_t = check(share_obs_batch).to(**self.tpdv)
        role_logits = self.policy.role_selector(share_obs_t)  # [T, n_agents, n_roles]
        role_probs = F.softmax(role_logits, dim=-1)
        role_dist = torch.distributions.Categorical(role_probs)
        roles_sampled = role_dist.sample()  # [T, n_agents]
        role_log_probs = role_dist.log_prob(roles_sampled)  # [T, n_agents]
        
        # Role advantage: mean trajectory advantage per agent (detached)
        # adv_targ: [T, n_agents, action_dim] or [T, n_agents, 1]
        adv_for_role = adv_targ.clone()
        if adv_for_role.dim() == 3:
            adv_for_role = adv_for_role.mean(dim=-1)  # [T, n_agents]
        adv_for_role = adv_for_role.squeeze(-1) if adv_for_role.dim() == 3 else adv_for_role
        adv_for_role = adv_for_role.detach()
        
        # Role REINFORCE loss: encourage roles that lead to high advantage
        role_loss = -(role_log_probs * adv_for_role).mean()
        
        # Role entropy bonus for exploration
        role_entropy = role_dist.entropy().mean()
        # -------------------------------

        # 1. Value loss
        values, _, _ = self.policy.evaluate_actions(share_obs_batch, obs_batch,
                                                     rnn_states_batch, rnn_states_critic_batch,
                                                     actions_batch, masks_batch,
                                                     available_actions_batch, active_masks_batch,
                                                     roles=roles_sampled)
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        # 2. Actor loss (per-role PPO)
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch,
            roles=roles_sampled)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        if action_log_probs.shape[-1] > 1:
            ratio = torch.prod(ratio, dim=-1, keepdim=True)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            policy_loss = policy_loss - self.entropy_coef * (dist_entropy * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            policy_loss = policy_loss - self.entropy_coef * dist_entropy

        self.policy.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self._use_max_grad_norm:
            actor_params = []
            for actor in self.policy.role_actors:
                actor_params += list(actor.parameters())
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
        else:
            all_params = []
            for actor in self.policy.role_actors:
                all_params += list(actor.parameters())
            actor_grad_norm = get_gard_norm(all_params)
        self.policy.actor_optimizer.step()

        # 3. Role selector loss (REINFORCE + entropy bonus)
        total_role_loss = role_loss - self.role_entropy_coef * role_entropy

        self.policy.role_optimizer.zero_grad()
        total_role_loss.backward()
        if self._use_max_grad_norm:
            role_grad_norm = nn.utils.clip_grad_norm_(self.policy.role_selector.parameters(), self.max_grad_norm)
        else:
            role_grad_norm = get_gard_norm(self.policy.role_selector.parameters())
        self.policy.role_optimizer.step()

        train_info['value_loss'] = value_loss.item()
        train_info['policy_loss'] = policy_loss.item()
        train_info['dist_entropy'] = dist_entropy.item() if torch.is_tensor(dist_entropy) else dist_entropy
        train_info['role_loss'] = role_loss.item()
        train_info['role_entropy'] = role_entropy.item()
        train_info['actor_grad_norm'] = actor_grad_norm if torch.is_tensor(actor_grad_norm) else actor_grad_norm
        train_info['critic_grad_norm'] = critic_grad_norm if torch.is_tensor(critic_grad_norm) else critic_grad_norm
        train_info['role_grad_norm'] = role_grad_norm if torch.is_tensor(role_grad_norm) else role_grad_norm
        train_info['ratio'] = ratio.mean().item()

        return train_info

    def train(self, buffer, update_actor=True):
        """Train using data from buffer."""
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        for k in ['value_loss', 'policy_loss', 'dist_entropy', 'role_loss', 'role_entropy', 'actor_grad_norm', 'critic_grad_norm', 'role_grad_norm', 'ratio']:
            train_info[k] = 0

        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        else:
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        count = 0
        for sample in data_generator:
            info = self.ppo_update(list(sample), update_actor)
            for k in train_info:
                train_info[k] += info.get(k, 0)
            count += 1

        for k in train_info:
            train_info[k] /= max(count, 1)

        return train_info

    def prep_training(self):
        self.policy.role_selector.train()
        for actor in self.policy.role_actors:
            actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.role_selector.eval()
        for actor in self.policy.role_actors:
            actor.eval()
        self.policy.critic.eval()
