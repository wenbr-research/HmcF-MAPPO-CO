# RODE: Role-based Decomposition for Multi-Agent PPO
# Adapted for antiAttack environment with dynamic role learning

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import get_shape_from_obs_space


class RoleSelector(nn.Module):
    """Selects a role for each agent based on global state.
    Runs every role_interval steps to provide temporal abstraction.
    """
    def __init__(self, args, cent_obs_space, n_roles, device=torch.device("cpu")):
        super(RoleSelector, self).__init__()
        self.n_roles = n_roles
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = MLPBase(args, cent_obs_shape)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.role_head = init_(nn.Linear(self.hidden_size, n_roles))
        self.to(device)

    def forward(self, cent_obs):
        """Output role logits for role assignment."""
        cent_obs = check(cent_obs).to(**self.tpdv)
        features = self.base(cent_obs)
        role_logits = self.role_head(features)
        return role_logits


class RODE_Policy:
    """RODE-style policy with role selector + role-conditioned actors.
    
    Architecture:
    - Role Selector: MLP(global_state) → role_logits (n_roles)
    - K Role Actors: independent R_Actor networks
    - 1 Shared Critic: R_Critic (centralized V function)
    
    Role assignment: softmax over role_logits, sample role per agent.
    Role switching: every role_interval steps for temporal abstraction.
    """
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, n_roles=3, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.n_roles = n_roles
        self.role_interval = getattr(args, 'role_interval', 5)
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # Role selector
        self.role_selector = RoleSelector(args, cent_obs_space, n_roles, device)
        
        # K independent role actors
        self.role_actors = nn.ModuleList([
            R_Actor(args, obs_space, act_space, device)
            for _ in range(n_roles)
        ])
        
        # Shared centralized critic
        self.critic = R_Critic(args, cent_obs_space, device)

        # Optimizers
        role_params = list(self.role_selector.parameters())
        actor_params = []
        for actor in self.role_actors:
            actor_params += list(actor.parameters())
        
        self.role_optimizer = torch.optim.Adam(role_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        from onpolicy.utils.util import update_linear_schedule
        update_linear_schedule(self.role_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def select_roles(self, cent_obs, deterministic=False):
        """Select roles for all agents based on centralized observation.
        
        Returns:
            roles: [n_agents] tensor of role indices
            role_log_probs: [n_agents] log probabilities of selected roles
        """
        role_logits = self.role_selector(cent_obs)
        role_probs = F.softmax(role_logits, dim=-1)
        
        if deterministic:
            roles = role_probs.argmax(dim=-1)
        else:
            roles = torch.multinomial(role_probs, 1).squeeze(-1)
        
        role_log_probs = torch.log(role_probs.gather(1, roles.unsqueeze(-1)) + 1e-10).squeeze(-1)
        return roles, role_log_probs

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                    available_actions=None, deterministic=False):
        """Get actions using role-conditioned actors.
        
        First selects roles, then routes each agent to its role's actor.
        """
        obs = check(obs).to(**self.tpdv)
        cent_obs_check = check(cent_obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        n_agents = obs.shape[0]
        action_dim = getattr(self.act_space, "num_discrete_space", None) or 1
        
        # 1. Select roles
        roles, _ = self.select_roles(cent_obs_check, deterministic)
        
        # 2. Route to role actors
        actions = torch.zeros(n_agents, action_dim, device=self.device)
        action_log_probs = torch.zeros(n_agents, action_dim, device=self.device)
        rnn_states_out = rnn_states_actor.clone()
        
        for role_id in range(self.n_roles):
            mask = (roles == role_id)
            if mask.sum() == 0:
                continue
            role_obs = obs[mask]
            role_rnn = rnn_states_actor[mask]
            role_masks_val = masks[mask]
            role_avail = available_actions[mask] if available_actions is not None else None
            
            role_actions, role_log_probs, role_rnn_out = self.role_actors[role_id](
                role_obs, role_rnn, role_masks_val, role_avail, deterministic)
            
            ridx = mask.nonzero(as_tuple=True)[0]
            for j, idx in enumerate(ridx):
                actions[idx] = role_actions[j]
                action_log_probs[idx] = role_log_probs[j]
                rnn_states_out[idx] = role_rnn_out[j]
        
        # 3. Critic
        values, rnn_states_critic_out = self.critic(cent_obs_check, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_out, rnn_states_critic_out

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, 
                         actions, masks, available_actions=None, active_masks=None, roles=None):
        """Evaluate actions for training. Routes each agent to its role's actor."""
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        if roles is None:
            roles, _ = self.select_roles(cent_obs, deterministic=True)
        
        n_agents = obs.shape[0]
        all_action_log_probs = torch.zeros(n_agents, actions.shape[-1], device=self.device)
        all_dist_entropy = torch.zeros(1, device=self.device)
        count = 0
        
        for role_id in range(self.n_roles):
            mask = (roles == role_id)
            if mask.sum() == 0:
                continue
            role_obs = obs[mask]
            role_rnn_actor = rnn_states_actor[mask]
            role_rnn_critic = rnn_states_critic[mask]
            role_actions = actions[mask]
            role_masks_val = masks[mask]
            role_avail = available_actions[mask] if available_actions is not None else None
            role_active = active_masks[mask] if active_masks is not None else None
            
            alp, de = self.role_actors[role_id].evaluate_actions(
                role_obs, role_rnn_actor, role_actions, role_masks_val, role_avail, role_active)
            
            ridx = mask.nonzero(as_tuple=True)[0]
            for j, idx in enumerate(ridx):
                all_action_log_probs[idx] = alp[j]
            all_dist_entropy += de * mask.sum().float()
            count += mask.sum().float()
        
        all_dist_entropy = all_dist_entropy / max(count, 1)
        
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, all_action_log_probs, all_dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """Get actions for evaluation (no role switching, use greedy roles)."""
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        n_agents = obs.shape[0]
        action_dim = getattr(self.act_space, "num_discrete_space", None) or 1
        actions = torch.zeros(n_agents, action_dim, device=self.device)
        rnn_out = rnn_states_actor.clone()
        
        # Use fixed role assignment: split agents evenly across roles
        for role_id in range(self.n_roles):
            start = role_id * n_agents // self.n_roles
            end = (role_id + 1) * n_agents // self.n_roles
            if start >= end:
                continue
            role_obs = obs[start:end]
            role_rnn = rnn_states_actor[start:end]
            role_masks_val = masks[start:end]
            role_avail = available_actions[start:end] if available_actions is not None else None
            
            role_actions, _, role_rnn_out = self.role_actors[role_id](
                role_obs, role_rnn, role_masks_val, role_avail, deterministic)
            
            actions[start:end] = role_actions
            rnn_out[start:end] = role_rnn_out
        
        return actions, rnn_out

    def save(self, save_dir, episode):
        torch.save(self.role_selector.state_dict(), str(save_dir) + f"/role_selector_ep{episode}.pt")
        for i, actor in enumerate(self.role_actors):
            torch.save(actor.state_dict(), str(save_dir) + f"/role_actor_{i}_ep{episode}.pt")
        torch.save(self.critic.state_dict(), str(save_dir) + f"/critic_ep{episode}.pt")

    def restore(self, model_dir, episode=0):
        self.role_selector.load_state_dict(torch.load(str(model_dir) + f"/role_selector_ep{episode}.pt"))
        for i, actor in enumerate(self.role_actors):
            actor.load_state_dict(torch.load(str(model_dir) + f"/role_actor_{i}_ep{episode}.pt"))
        self.critic.load_state_dict(torch.load(str(model_dir) + f"/critic_ep{episode}.pt"))
