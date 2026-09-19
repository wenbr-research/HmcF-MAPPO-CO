# HmcF/MAPPO-CO: Multi-Agent Tower Defense

This repository provides a multi-agent reinforcement-learning testbed for coordinated tower defense. In each episode, multiple defense towers observe incoming targets, available firing channels, target states, and remaining ammunition. The agents must decide when and where to engage while avoiding ineffective use of limited resources.

The repository contains two formulations of the same tower-defense problem. They are kept side by side to support controlled comparison between direct high-dimensional multi-discrete MARL baselines and the proposed HmcF/MAPPO-CO formulation.

## What this project studies

The central challenge is joint target engagement under resource and channel constraints. A defense tower may have several firing channels, but independent local decisions can produce redundant assignments or exhaust ammunition too early. The environment therefore exposes both local tower information and global state features for cooperative multi-agent learning.

The project supports MAPPO-style centralized training with decentralized execution, together with the baseline implementations included in the repository. Performance is measured primarily by the episode interception ratio.

## Implementations

| Directory | Decision formulation | Per-tower action space | Main purpose |
| --- | --- | --- | --- |
| [`multi-discrete-marl-baselines`](multi-discrete-marl-baselines/) | Direct firing-quantity selection | Five channels, each taking `0` to `4` | Multi-discrete MARL baselines |
| [`hmcf-mappo-co`](hmcf-mappo-co/) | Binary channel activation followed by coordinated allocation | Five channels, each taking `0` or `1` | Proposed HmcF/MAPPO-CO framework |

In the multi-discrete baseline formulation, a single tower has `5^5 = 3125` possible channel-action combinations. In HmcF/MAPPO-CO, the same five-channel structure is retained, but the binary activation decision gives `2^5 = 32` combinations. The coordination stage then resolves feasible engagement allocation according to target assignments, channel availability, and remaining ammunition.

## Repository layout

```text
Towerdefence/
|- multi-discrete-marl-baselines/  # Direct multi-discrete MARL baselines
`- hmcf-mappo-co/                 # Proposed HmcF/MAPPO-CO framework
```

Each implementation contains its own environment configuration, training entry point, runner, algorithm implementations, and setup instructions.

## Scope of this release

Only the tower-defense environment and code required to train it are included. Historical experiment outputs, logs, trained checkpoints, generated media, caches, and unrelated benchmark environments have been removed.

## Getting started

Choose an implementation directory and follow its README. The two implementations use different environment names and training entry points, so commands should not be mixed.
