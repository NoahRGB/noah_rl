# NoahRGB 09/2026
# template for algorithms 
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/)
#

import hydra
from omegaconf import DictConfig

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv
from noah_rl.utils.logging import Logger


def train(cfg: DictConfig, hydra_dir: str):

    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    states, _ = env.reset()
    num_rollouts = cfg.timesteps // (cfg.rollout_len * cfg.num_envs)
    for rollout in range(num_rollouts):
        actions = env.action_space.sample() 
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)


@hydra.main(config_path="../../params", config_name="alg", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train(cfg, run_dir)

if __name__ == "__main__":
    main()
