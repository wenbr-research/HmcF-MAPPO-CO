import time
import csv

import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class ANTIATTACKRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(ANTIATTACKRunner, self).__init__(config)
        self.assign_model_policy = self.all_args.assign_model_policy

    def run(self):
        start = time.time()
        self.warmup()
        self.best_eval_inter = -1.0
        self.no_improve_count = 0
        self.early_stop_patience = 1000000000  # Effectively disable early stopping for fixed-length experiments.
        # eval log
        self.eval_log_path = str(self.run_dir / 'eval_log.txt') if not self.use_wandb else None
        if self.eval_log_path:
            with open(self.eval_log_path, 'w') as f:
                f.write('')
        self.train_episode_log_path = self.run_dir / 'train_episode_intercepts.csv'
        with open(self.train_episode_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train_update', 'total_env_steps', 'rollout_env_index', 'interception_ratio', 'episode_reward'])
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        for episode in range(episodes):
            print("episode: ", episode)

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # infos_episode = None
            # isover = False # 用于判断游戏是否结束
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # 其中actions是一个list，里面有n_rollout_threads个元素，每个元素是一个list，里面有num_agents个元素，每个元素是一个int
                # Observe reward and next obs

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                done_env = np.all(dones, axis=1)
                if np.any(done_env):
                    current_env_steps = (episode * self.episode_length + step + 1) * self.n_rollout_threads
                    with open(self.train_episode_log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for env_index in np.flatnonzero(done_env):
                            episode_info = infos[env_index][0]
                            writer.writerow([
                                episode,
                                current_env_steps,
                                env_index,
                                episode_info['inter_ratio'],
                                episode_info['reward'],
                            ])
                ## 这里的actions可能需要把mask后的actions传入，todo:需要确认！
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                # if np.all(dones) and not isover:
                #     infos_episode = infos # 在结束时存放清算信息
                #     isover = True

                # if self.envs.show_ui:
                #     self.envs.render()

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                env_infos = {}
                if self.env_name == "antiAttack":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        inter_ratio = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                            if 'inter_ratio' in info[agent_id].keys():
                                inter_ratio.append(info[agent_id]['inter_ratio'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        agent_i = 'agent%i/inter_ratio' % agent_id
                        env_infos[agent_k] = idv_rews
                        env_infos[agent_i] = inter_ratio

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print(f"step: {episode} ==> evaluating...")
                eval_inter = self.eval(total_num_steps)
                # 早停检查：连续 patience 次不提升则停止
                if eval_inter > self.best_eval_inter:
                    self.best_eval_inter = eval_inter
                    self.no_improve_count = 0
                    print(f"new best interception ratio: {self.best_eval_inter:.4f}")
                else:
                    self.no_improve_count += 1
                    print(f"no improvement for {self.no_improve_count}/{self.early_stop_patience} evals (best: {self.best_eval_inter:.4f}, current: {eval_inter:.4f})")
                # 写入 eval_log
                if self.eval_log_path:
                    with open(self.eval_log_path, 'a') as f:
                        f.write(f"episode={episode}  step={total_num_steps}  eval_intercept={eval_inter:.6f}  best_intercept={self.best_eval_inter:.6f}  patience={self.no_improve_count}/{self.early_stop_patience}\n")
                if self.no_improve_count >= self.early_stop_patience:
                    print(f"Early stopping triggered! Best interception ratio: {self.best_eval_inter:.4f}")
                    if self.eval_log_path:
                        with open(self.eval_log_path, 'a') as f:
                            f.write(f"EARLY_STOP episode={episode}  best={self.best_eval_inter:.6f}\n")
                    break

    def warmup(self):
        # reset env
        obs, _, available_actions = self.envs.reset()
        share_obs = obs

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        # rearrange action

        return values, actions, action_log_probs, rnn_states, rnn_states_critic #, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # dones_env = np.all(dones, axis=1)
        # assert self.n_rollout_threads == 1, "only support n_rollout_threads == 1 now!"

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)
        # masks 的作用是将done的agent的mask设置为0，不再计算其loss，也就是agent结束后不再计算其loss, 也就是不再计算其loss, 与active_masks的区别是，masks是不再计算loss，active_masks是不再计算loss和不再计算action
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks,
                           available_actions=available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalar(k, v, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_infos = np.array([[None] * self.num_agents] * self.n_eval_rollout_threads)
        eval_obs, _, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1),
                             dtype=np.float32)  # 作用是将done的agent的mask设置为0，不再计算其loss
        eval_dones_env = np.array([False] * self.n_eval_rollout_threads)
        while True:
        # for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1),
                                                         dtype=np.float32)
            # if self.envs.show_ui:
            #     self.eval_envs.render()

            eval_dones_env[np.array(eval_dones[:, 0]) == True] = True
            # 将eval_dones中第一列为True的位置，eval_infos对应的信息存入eval_episode_infos对应的位置中
            if (eval_dones[:, 0] == True).sum() > 0: # 有存在done的episode
                index = np.where(eval_dones[:, 0])[0]
                eval_episode_infos[index] = np.array(eval_infos)[index]

            if all(eval_dones_env): # 所有episode都结束了
                break

        eval_env_infos = {}
        inter_rewards = np.array([[eval_episode_infos[ei][0]['inter_ratio'], eval_episode_infos[ei][0]['reward']] for ei in range(eval_episode_infos.shape[0])])
        eval_env_infos['eval_episode_intercept'] = inter_rewards[:, 0]
        eval_env_infos['eval_episode_rewards'] = inter_rewards[:, 1]
        # eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        avg_inter = np.mean(inter_rewards[:, 0])
        avg_reward = np.mean(inter_rewards[:, 1])
        self.log_env(eval_env_infos, total_num_steps)
        print(f'eval average episode rewards of agent: {avg_reward}; average episode interception ratio {avg_inter}')
        if self.use_wandb:
            wandb.log({'eval_average_episode_rewards': avg_reward, 'eval_average_episode_intercept': avg_inter}, step=total_num_steps)
        else:
            self.writter.add_scalar('eval_average_episode_intercept', avg_inter, total_num_steps)
            self.writter.add_scalar('eval_average_episode_rewards', avg_reward, total_num_steps)
        return avg_inter
