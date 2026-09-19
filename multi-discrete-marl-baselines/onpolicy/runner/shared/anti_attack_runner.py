import csv
import os
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class TowerDefenseRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(TowerDefenseRunner, self).__init__(config)
        self.eval_index = 0
        self.eval_episode_log_path = os.path.join(
            str(self.run_dir), "eval_episode_intercepts.csv"
        )
        with open(self.eval_episode_log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "eval_index",
                "total_num_steps",
                "eval_episode",
                "thread_id",
                "interception_ratio",
                "reward",
            ])

    def run(self):
        start = time.time()
        self.warmup()
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
                if self.env_name == "towerDefense":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        defense_ratio = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                            if 'defense_ratio' in info[agent_id].keys():
                                defense_ratio.append(info[agent_id]['defense_ratio'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        agent_i = 'agent%i/defense_ratio' % agent_id
                        env_infos[agent_k] = idv_rews
                        env_infos[agent_i] = defense_ratio

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print(f"step: {episode} ==> evaluating...")
                self.eval(total_num_steps)

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
                           actions, action_log_probs, values, rewards[..., np.newaxis], masks,
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
                             dtype=np.float32)
        eval_dones_env = np.array([False] * self.n_eval_rollout_threads)
        # Some scenarios terminate slightly after episode_length. The safety
        # limit lets every evaluation thread finish one episode without
        # risking an unbounded evaluation loop.
        max_eval_steps = max(self.episode_length * 2, self.episode_length + 100)
        for eval_step in range(max_eval_steps):
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

            done_now = np.asarray(eval_dones[:, 0], dtype=bool)
            newly_done = np.where(done_now & ~eval_dones_env)[0]
            if len(newly_done) > 0:
                eval_episode_infos[newly_done] = np.asarray(eval_infos, dtype=object)[newly_done]
                eval_dones_env[newly_done] = True

            if all(eval_dones_env):
                break

        eval_env_infos = {}
        valid_records = [
            (thread_id, info)
            for thread_id, info in enumerate(eval_episode_infos)
            if info[0] is not None
        ]
        if len(valid_records) == 0:
            print('[WARN] eval: no valid episode infos, skipping eval metrics')
            self.eval_index += 1
            return
        if len(valid_records) != self.n_eval_rollout_threads:
            print(
                f'[WARN] eval: collected {len(valid_records)}/'
                f'{self.n_eval_rollout_threads} completed episodes'
            )
        valid_infos = [info for _, info in valid_records]
        inter_rewards = np.array([[info[0]['defense_ratio'], info[0]['reward']] for info in valid_infos])
        with open(self.eval_episode_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for eval_episode, ((thread_id, _), values) in enumerate(
                zip(valid_records, inter_rewards), start=1
            ):
                writer.writerow([
                    self.eval_index,
                    total_num_steps,
                    eval_episode,
                    thread_id,
                    float(values[0]),
                    float(values[1]),
                ])
        eval_env_infos['eval_episode_defense'] = inter_rewards[:, 0]
        eval_env_infos['eval_episode_rewards'] = inter_rewards[:, 1]
        self.log_env(eval_env_infos, total_num_steps)
        print(f'eval average episode rewards of agent: {np.mean(inter_rewards[:, 1])}; average episode defense ratio {np.mean(inter_rewards[:, 0])}')
        if self.use_wandb:
            wandb.log({'eval_average_episode_rewards': np.mean(inter_rewards[:, 1]), 'eval_average_episode_defense': np.mean(inter_rewards[:, 0])}, step=total_num_steps)
        else:
            self.writter.add_scalar('eval_average_episode_defense', np.mean(inter_rewards[:, 0]), total_num_steps)
            self.writter.add_scalar('eval_average_episode_rewards', np.mean(inter_rewards[:, 1]), total_num_steps)
        self.eval_index += 1

