# ROMA: Role-based Multi-Agent learning with continuous role embeddings
# Adapted for on-policy antiAttack environment

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import get_shape_from_obs_space


class RoleEncoder(nn.Module):
    """Encodes agent observation into a continuous role embedding z.
    z ~ N(mu(obs), sigma(obs))
    """
    def __init__(self, args, obs_space, role_dim=8, device=torch.device("cpu")):
        super(RoleEncoder, self).__init__()
        self.role_dim = role_dim
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.mu_head = init_(nn.Linear(self.hidden_size, role_dim))
        self.logstd_head = init_(nn.Linear(self.hidden_size, role_dim))
        self.to(device)

    def forward(self, obs, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        features = self.base(obs)
        mu = self.mu_head(features)
        logstd = self.logstd_head(features)
        logstd = torch.clamp(logstd, -2, 2)
        std = torch.exp(logstd)

        if deterministic:
            z = mu
        else:
            eps = torch.randn_like(std)
            z = mu + eps * std

        return z, mu, std


class RoleConditionedActor(nn.Module):
    """Actor conditioned on role embedding: π(a | o, z)"""
    def __init__(self, args, obs_space, action_space, role_dim=8, device=torch.device("cpu")):
        super(RoleConditionedActor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.role_dim = role_dim
        from onpolicy.algorithms.utils.act import ACTLayer

        obs_shape = get_shape_from_obs_space(obs_space)
        input_dim = obs_shape[0] if isinstance(obs_shape, (list, tuple)) else obs_shape
        # MLP base with role concatenation
        self.base = nn.Sequential(
            nn.Linear(input_dim + role_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, role_z, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        role_z = check(role_z).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        inp = torch.cat([obs, role_z], dim=-1)
        features = self.base(inp)
        actions, action_log_probs = self.act(features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, role_z, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        role_z = check(role_z).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        inp = torch.cat([obs, role_z], dim=-1)
        features = self.base(inp)
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            features, action, available_actions,
            active_masks=active_masks if active_masks is not None else None)
        return action_log_probs, dist_entropy


class ROMA_Policy:
    """ROMA-style policy with per-agent continuous role embeddings.
    
    Architecture:
    - Role Encoder: MLP(obs) → z ~ N(mu, sigma), dim=8
    - Actor: MLP([obs, z]) → action (role-conditioned)
    - Critic: MLP(global_state) → V (shared)
    
    Regularization:
    - Role diversity: maximize variance of z across agents
    - Role consistency: minimize ||z_t - z_{t+1}||
    """
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, 
                 n_roles=3, role_dim=8, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.role_dim = role_dim
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # Role encoder
        self.role_encoder = RoleEncoder(args, obs_space, role_dim, device)
        # Role-conditioned actor (single, shared, receives z per agent)
        self.actor = RoleConditionedActor(args, obs_space, act_space, role_dim, device)
        # Shared critic
        self.critic = R_Critic(args, cent_obs_space, device)

        # Optimizers
        role_params = list(self.role_encoder.parameters())
        self.role_optimizer = torch.optim.Adam(role_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        from onpolicy.utils.util import update_linear_schedule
        update_linear_schedule(self.role_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        cent_obs_check = check(cent_obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 1. Encode roles
        role_z, role_mu, role_std = self.role_encoder(obs, deterministic)

        # 2. Role-conditioned actor
        actions, action_log_probs, _ = self.actor(obs, role_z, rnn_states_actor, masks,
                                                    available_actions, deterministic)

        # 3. Critic
        values, rnn_states_critic_out = self.critic(cent_obs_check, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic_out

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                         actions, masks, available_actions=None, active_masks=None, roles=None):
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # Re-encode roles (or use provided ones)
        if roles is not None:
            role_z = check(roles).to(**self.tpdv)
        else:
            role_z, _, _ = self.role_encoder(obs, deterministic=True)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, role_z, rnn_states_actor, actions, masks,
            available_actions, active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        role_z, _, _ = self.role_encoder(obs, deterministic=True)
        actions, _, rnn_out = self.actor(obs, role_z, rnn_states_actor, masks,
                                           available_actions, deterministic)
        return actions, rnn_out

    def save(self, save_dir, episode):
        torch.save(self.role_encoder.state_dict(), str(save_dir) + f"/role_encoder_ep{episode}.pt")
        torch.save(self.actor.state_dict(), str(save_dir) + f"/actor_ep{episode}.pt")
        torch.save(self.critic.state_dict(), str(save_dir) + f"/critic_ep{episode}.pt")

    def restore(self, model_dir, episode=0):
        self.role_encoder.load_state_dict(torch.load(str(model_dir) + f"/role_encoder_ep{episode}.pt"))
        self.actor.load_state_dict(torch.load(str(model_dir) + f"/actor_ep{episode}.pt"))
        self.critic.load_state_dict(torch.load(str(model_dir) + f"/critic_ep{episode}.pt"))
