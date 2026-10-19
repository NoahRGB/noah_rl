
# RL implementations

many implementations are adapted from my [previous RL repo](https://github.com/NoahRGB/reinforcement-learning)


algorithms are in `noah_rl/algorithms/` 


parameters are in `params/` (using [Hydra](https://hydra.cc/docs/intro/) .yaml files)


minor utils are in `noah_rl/utils` (e.g. logging, basic [PyTorch](https://pytorch.org/) network building, etc.)


algorithms will use their corresponding .yaml file from `params/`

any algorithm can be run with `python run.py NAME title="` (e.g. `python run.py ppo title="ppo tests"`)

individual params can be overwritten as arguments, e.g. `python run.py ppo title="ppo tests" lr=0.001 minibatch_size=32`
