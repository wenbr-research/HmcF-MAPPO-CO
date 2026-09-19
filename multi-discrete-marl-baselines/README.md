# Multi-Discrete MARL Baselines for Tower Defense

This directory contains the direct-action multi-discrete MARL baselines for tower defense. Each defense tower controls five firing channels and directly selects the firing quantity of every channel. This formulation preserves fine-grained control, but creates a large per-agent multi-discrete action space.

## Problem formulation

At every decision step, each tower observes target-related features, its channel state, available actions, and resource status. The multi-agent policy selects a five-dimensional action vector. A channel action is one of:

```text
0: do not engage
1-4: select the corresponding firing quantity
```

With five channels and five action values per channel, each tower has `5^5 = 3125` action combinations. Actions are applied directly after feasibility masking, making this formulation the high-dimensional multi-discrete baseline.

## Included scenarios

The `onpolicy/envs/anti_Attack/configs/` directory provides 5-, 10-, and 15-tower configurations. The default `env_config.json5` uses the 10-tower scenario.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Python 3.8 with a PyTorch installation appropriate for the local CPU or GPU is recommended.

## Training

Run the maintained tower-defense entry point from this directory:

```bash
python onpolicy/scripts/train/train_antiattack_brief.py \
  --env_name towerDefense \
  --algorithm_name mappo \
  --experiment_name baseline_example \
  --n_rollout_threads 1 \
  --num_env_steps 200000
```

The same entry point supports the algorithm choices implemented in this release, including MAPPO-family and other MARL baselines. Training outputs are written below `onpolicy/scripts/results/`, which is ignored by Git.

## Visualization

The following command exports one randomly controlled environment episode as a GIF:

```bash
python onpolicy/scripts/render/render_antiattack.py
```

The visualization is an environment demonstration, not replay of a trained policy.

## Key files

- `onpolicy/envs/anti_Attack/environment.py`: environment transition and action masking.
- `onpolicy/envs/anti_Attack/defense_tower.py`: tower state and direct firing execution.
- `onpolicy/scripts/train/train_antiattack_brief.py`: supported baseline training entry point.
- `onpolicy/envs/anti_Attack/configs/`: scenario configurations.
