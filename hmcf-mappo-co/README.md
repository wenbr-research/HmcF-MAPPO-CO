# HmcF/MAPPO-CO for Tower Defense

HmcF/MAPPO-CO retains the five-channel tower-defense structure while reducing each channel decision to binary activation. Instead of asking the policy to choose a firing quantity directly, the policy first decides whether each channel should participate. A subsequent coordination stage converts feasible activation decisions into final engagements.

## Problem formulation

Each tower controls five firing channels. The policy action for every channel is:

```text
0: do not activate the channel
1: activate the channel for engagement
```

The action space is therefore `2^5 = 32` combinations per tower, compared with `5^5 = 3125` in the multi-discrete baselines. After action masking, the coordination stage consolidates target assignments across towers and applies feasible engagements while accounting for available channels and remaining ammunition.

This separates two decisions:

1. **MARL policy:** whether a channel should be activated.
2. **Coordination stage:** how feasible active channels are allocated to target engagements.

## Included scenarios

The default `env_config.json5` is the 10-tower scenario. Additional 10- and 15-tower profiles are provided as `env_config_10t.json5` and `env_config_15t.json5`.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Python 3.8 with a PyTorch installation appropriate for the local CPU or GPU is recommended.

## Training

Run training from this directory:

```bash
python onpolicy/scripts/train/train_antiattack.py \
  --env_name antiAttack \
  --algorithm_name mappo \
  --experiment_name hmcf_mappo_co_example \
  --n_rollout_threads 1 \
  --num_env_steps 200000
```

Training outputs are stored below `onpolicy/scripts/results/`, which is ignored by Git.

## Visualization

Export an anonymized 10-tower environment demonstration as a GIF:

```bash
python onpolicy/scripts/train/visualize_interception_gif.py \
  --output visualizations/interception_episode.gif
```

The visualization uses a deterministic all-channel activation pattern to illustrate environment dynamics. It is not a replay of a trained MAPPO policy.

## Key files

- `onpolicy/envs/anti_Attack/environment.py`: binary action processing and coordinated engagement execution.
- `onpolicy/envs/anti_Attack/tower.py`: tower state and firing execution.
- `onpolicy/scripts/train/train_antiattack.py`: supported HmcF/MAPPO-CO training entry point.
- `onpolicy/envs/anti_Attack/env_config*.json5`: scenario configurations.
