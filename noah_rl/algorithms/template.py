# NoahRGB 09/2026
# template for algorithms 
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/)
#

import hydra
from omegaconf import DictConfig
import gymnasium as gym

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv

def train(cfg: DictConfig):

    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, **cfg.env_kwargs)

    states, _ = env.reset()
    num_rollouts = cfg.timesteps // (cfg.rollout_len * cfg.num_envs)
    for rollout in range(num_rollouts):

        for timestep in range(cfg.rollout_len):
            action = env.action_space.sample()
            sprimes, rewards, is_terms, is_truncs, info = env.step(action)

            # TODO store transition

            states = env.reset_finished_envs(sprimes, is_terms, is_truncs)






@hydra.main(config_path="../../params", config_name="alg", version_base=None)
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
