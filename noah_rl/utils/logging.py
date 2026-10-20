# NoahRGB 09/2026
#
# a file that contains a logger class to simplify the saving/logging
# of various statistics during training
#

import os, json
from datetime import datetime
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

def update_summary(log_dir: str, **fields):
    path = os.path.join(log_dir, "summary.json")
    summary = {}
    if os.path.isfile(path):
        with open(path) as f:
            summary = json.load(f)
    summary.update(fields)

    # write to temp file and then swap so that summaries
    #are always fully written
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, path)

class Logger:
    def __init__(self, num_envs: int, title: str, log_dir: str, tensorboard: bool):
        self.num_envs = num_envs
        self.title = title
        self.log_dir = log_dir
        self.tensorboard = tensorboard
        self.episodes_completed = 0
        self.timesteps_completed = 0
        self.episodic_reward = []
        self.best_mean_reward = None
        if self.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(os.path.join(log_dir, "tensorboard"), flush_secs=1)
            except ImportError:
                print("tensorboard not installed, logging without it")
                self.tensorboard = False

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

            if len(completed_rewards) > 0:
                self._write_summary()

    def _write_summary(self):
        # write summary file
        mean_reward = float(np.mean(self.episodic_reward[-100:]))
        if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

        update_summary(self.log_dir,
            timesteps=self.timesteps_completed,
            episodes=self.episodes_completed,
            mean_reward_100=mean_reward,
            best_mean_reward_100=self.best_mean_reward,
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )

    def log_stats(self, stats: dict):
        for name, stat in stats.items():
            self._tensorboard_log(name, stat, self.timesteps_completed)

    def log_network(self, network_dict: dict):
        # save it to a temp file first and then swap them
        # because network can sometimes be half written and broken
        path = os.path.join(self.log_dir, "network.pt")
        torch.save(network_dict, path + ".tmp")
        os.replace(path + ".tmp", path)



