#
# 
# 
#

import hydra
from omegaconf import DictConfig
import gymnasium as gym

def train(cfg: DictConfig):

    def make_one_env():
        env = gym.make(cfg.env_name, **cfg.env_kwargs)
        return env

    num_rollouts = cfg.timesteps // cfg.rollout_len
    for rollout in range(num_rollouts):

        for timestep in range(cfg.rollout_len):
            # step env, collect transition
            ...

        for epoch in range(cfg.epochs):
            # optimise in minibatches over the collected rollout
            ...

    







@hydra.main(config_path="../../params", config_name="ppo", version_base=None)
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
