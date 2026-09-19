#!/usr/bin/env python

import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

"""Train script for towerDefense."""


def get_one_hot_tensor(a, space):
    '''
    将a转化为one-hot向量，特别是a的最后一维大小是space
    :param a: [agent_num, fire_power_channels]
    :param space: space of the one-hot vector at the last dimension
    :return: one-hot tensor: [agent_num, fire_power_channels, space]
    '''
    assert np.max(a) < space, f"np.max(a)={np.max(a)}, space={space}"
    # 使用np.eye创建一个单位矩阵，然后对其进行索引以获得one-hot编码
    one_hot_matrix = np.eye(space)[a]

    return one_hot_matrix

def make_train_env(all_args):
    # 注意，在debug期间用这块，如果实际运行，需要将此部分注释掉。 todo：
    # if all_args.env_name == "towerDefense":
    #     from onpolicy.envs.tower_defense_orig.environment import Environment
    #     env = Environment(all_args)
    #     return env

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "towerDefense":
                from onpolicy.envs.anti_Attack.environment import Environment
                env = Environment(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    # if all_args.env_name == "towerDefense":
    #     from onpolicy.envs.tower_defense_orig.environment import Environment
    #     env = Environment(all_args)
    #     return env

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "towerDefense":
                from onpolicy.envs.anti_Attack.environment import Environment
                env = Environment(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(parser):
    ## 是否可视化
    parser.add_argument("--show_ui", type=bool, default=False, help="Whether to show the UI")
    ## 训练参数及测试参数
    parser.add_argument("--num_agents", type=int, help="the number of agents")
    parser.add_argument("--train_episodes", type=int, default=int(200000), help=" Maximum number of training episodes")
    parser.add_argument("--evaluate_freq", type=float, default=1000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=100, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=5000, help="Save the model every 'save_freq' steps")
    ## 网络及模型参数
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=4, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the mlp")

    ## trpo parameters
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False,
                        help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    ## 特征参数
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--add_last_action", action='store_true', default=False)

    all_args = parser.parse_args()

    return all_args


def main():
    parser = get_config()
    all_args = parse_args(parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name in ("mappo", "mat", "mat_dec", "happo", "hatrpo", "rode", "roma"):
        print(f"u are choosing to use {all_args.algorithm_name}")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args, project=all_args.env_name)

        all_args = wandb.config # for wandb sweep
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = len(envs.action_space)

    # Set save dir for per-episode interception ratio logging
    if hasattr(envs, 'envs') and len(envs.envs) > 0:
        envs.envs[0]._ratio_save_dir = str(run_dir)
    if eval_envs is not None and hasattr(eval_envs, 'envs') and len(eval_envs.envs) > 0:
        eval_envs.envs[0]._ratio_save_dir = str(run_dir)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

        # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.anti_attack_runner import TowerDefenseRunner as Runner
    else:
        # print('runner wrong in train_towerdefense_brief.py line 264.')
        from onpolicy.runner.shared.anti_attack_runner import TowerDefenseRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":

    main()
