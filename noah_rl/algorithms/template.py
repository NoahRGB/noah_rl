#
#
# 
#

import hydra
from omegaconf import DictConfig

def train(cfg: DictConfig):
    print(cfg)

@hydra.main(config_path="../../params", config_name="name", version_base=None)
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
