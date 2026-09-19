# Types and shapes in on-policy project


## For MAPPO in SMAC environment

## Original code in runner.py
```python

for step in range(self.episode_length):
    # Sample actions
    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
        
    # Obser reward and next obs
    obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

    data = obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, \
            rnn_states, rnn_states_critic

    # insert data into buffer
    self.insert(data)

```

## Types and shapes information

- from env step:
- obs type:<class 'numpy.ndarray'>, shape:(32, 3, 64)
- share_obs type:<class 'numpy.ndarray'>, shape:(32, 3, 81)
- rewards type:<class 'numpy.ndarray'>, shape:(32, 3, 1)
- dones type:<class 'numpy.ndarray'>, shape:(32, 3)
- infos type:<class 'tuple'>, len:32
  
infos[0] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[1] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[2] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[3] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[4] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[5] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[6] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[7] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[8] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[9] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[10] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[11] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[12] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[13] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[14] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[15] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[16] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[17] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[18] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[19] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[20] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[21] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[22] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[23] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[24] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[25] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[26] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[27] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[28] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[29] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 3, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[30] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

infos[31] type:<class 'list'>, len:3, content:[{'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}, {'battles_won': 0, 'battles_game': 2, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]

- available_actions type:<class 'numpy.ndarray'>, shape:(32, 3, 9)

- from collect:
- values type:<class 'numpy.ndarray'>, shape:(32, 3, 1)
- actions type:<class 'numpy.ndarray'>, shape:(32, 3, 1)
- action_log_probs type:<class 'numpy.ndarray'>, shape:(32, 3, 1)
- rnn_states type:<class 'numpy.ndarray'>, shape:(32, 3, 1, 64)
- rnn_states_critic type:<class 'numpy.ndarray'>, shape:(32, 3, 1, 64)


## For MAPPO in AntAttack environment
## Original code in runner.py

```python

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
```

## Types and shapes information

- from env step:
- obs type:<class 'list'>
- share_obs type:<class 'list'>
- rewards type:<class 'list'>
- dones type:<class 'numpy.ndarray'>, shape:(6,)
  

infos type:<class 'list'>, len:6

infos[0] type:<class 'dict'>, len:7, content:{'interception_target': 0, 'intercepted_missile': [None, None, None, None, None], 'insight_missile': 26, 'rest_ammo': 0, 'done': True, 'reward': 0, 'individual_reward': 1.751336898395722}

infos[1] type:<class 'dict'>, len:7, content:{'interception_target': 4, 'intercepted_missile': [20, None, 30, 70, 96], 'insight_missile': 13, 'rest_ammo': 0, 'done': True, 'reward': 0, 'individual_reward': 2.0588235294117645}

infos[2] type:<class 'dict'>, len:7, content:{'interception_target': 0, 'intercepted_missile': [None, None, None, None, None], 'insight_missile': 9, 'rest_ammo': 0, 'done': True, 'reward': 0, 'individual_reward': 1.8315508021390374}

infos[3] type:<class 'dict'>, len:7, content:{'interception_target': 1, 'intercepted_missile': [None, None, 83, None, None], 'insight_missile': 12, 'rest_ammo': 0, 'done': True, 'reward': 0, 'individual_reward': 1.8048128342245988}

infos[4] type:<class 'dict'>, len:7, content:{'interception_target': 0, 'intercepted_missile': [None, None, None, None, None], 'insight_missile': 36, 'rest_ammo': 0, 'done': True, 'reward': 0, 'individual_reward': 1.8716577540106951}

infos[5] type:<class 'dict'>, len:7, content:{'interception_target': 0, 'intercepted_missile': [None, None, None, None, None], 'insight_missile': 36, 'rest_ammo': 0, 'done': True, 'reward': 0, 'individual_reward': 1.4705882352941178}

- available_actions type:<class 'numpy.ndarray'>, shape:(6, 5)

- from collect:
- values type:<class 'numpy.ndarray'>, shape:(1, 6, 1)
- actions type:<class 'numpy.ndarray'>, shape:(1, 6, 5)
- action_log_probs type:<class 'numpy.ndarray'>, shape:(1, 6, 5)
- rnn_states type:<class 'numpy.ndarray'>, shape:(1, 6, 1, 64)
- rnn_states_critic type:<class 'numpy.ndarray'>, shape:(1, 6, 1, 64)

