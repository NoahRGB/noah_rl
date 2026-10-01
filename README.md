
# RL implementations

many implementations are adapted from my [previous RL repo](https://github.com/NoahRGB/reinforcement-learning)


algorithms are in `noah_rl/algorithms/` 


parameters are in `params/` (using [Hydra](https://hydra.cc/docs/intro/) .yaml files)


minor utils are in `noah_rl/utils` (e.g. logging, basic [PyTorch](https://pytorch.org/) network building, etc.)


any algorithm can be run with `python -m noah_rl.algorithms.NAME` (e.g. `noah_rl.algorithms.ppo`)


algorithms will use their corresponding .yaml file from `params/`


individual params can be overwritten as arguments, e.g. `python -m noah_rl.algorithms.dqn replay_size=1000`
