# NoahRGB 09/2026
#
# a file that contains a logger class to simplify the saving/logging
# of various statistics during training
#

import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

class Logger:
    def __init__(self, num_envs: int, title: str, log_dir: str, tensorboard: bool):
        self.num_envs = num_envs
        self.title = title
        self.log_dir = log_dir
        self.tensorboard = tensorboard
        self.episodes_completed = 0
        self.timesteps_completed = 0
        self.episodic_reward = []
        if self.tensorboard:
           from torch.utils.tensorboard import SummaryWriter 
           self.tb_writer = SummaryWriter(self._set_log_path(os.path.join("results/tensorboard", title)), flush_secs=1)

    def _set_log_path(self, log_path: str):
        if os.path.isdir(log_path):
            current_digit = 1
            current_check = f"{log_path}_{current_digit}"
            while os.path.isdir(current_check):
                current_digit += 1
                current_check = f"{log_path}_{current_digit}"
            return current_check
        return log_path 

    def _tensorboard_log(self, name, val, step):
        if self.tensorboard:
            self.tb_writer.add_scalar(name, val, step)

    def log_episodes(self, info):
        self.timesteps_completed += self.num_envs 
        if "episode" in info:
            done_idxs = info["_episode"]
            completed_rewards = info["episode"]["r"][done_idxs]
            for reward in completed_rewards:
                self.episodes_completed += 1
                print(f"episode {self.episodes_completed} timesteps {self.timesteps_completed} reward {reward}")
                self.episodic_reward.append(reward)
                self._tensorboard_log("episodic_reward", reward, self.timesteps_completed)
                self._tensorboard_log("mean_episodic_reward", np.mean(self.episodic_reward[-100:]), self.timesteps_completed)

    def log_stats(self, stats: dict):
        for name, stat in stats.items():
            self._tensorboard_log(name, stat, self.timesteps_completed)

    def log_network(self, network_dict: dict):
        torch.save(network_dict, os.path.join(self.log_dir, "network.pt"))



