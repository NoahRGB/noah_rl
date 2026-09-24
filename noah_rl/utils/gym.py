# NoahRGB 09/2026
# 
# a file to wrap the gymnasium environment process
# into a simple interface that each algorithm can use
#
#

import gymnasium as gym
import numpy as np
import ale_py

# supported wrappers and their classes
WRAPPERS = {
    "normalise_obs": gym.wrappers.NormalizeObservation,
    "normalise_reward": gym.wrappers.NormalizeReward,
    "frame_stack": gym.wrappers.FrameStackObservation,
    "atari": gym.wrappers.AtariPreprocessing,
    "greyscale": gym.wrappers.GrayscaleObservation,
    "resize": gym.wrappers.ResizeObservation,
}

class GymEnv:
    def __init__(self, env_name: str, num_envs: int, seed: int|None = None, wrappers: dict|None = None, **env_kwargs):
        self.env_name = env_name
        self.seed = seed
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs
        self.has_reset = False
        self.wrappers = wrappers or {}

        self.env = self._make_envs()
        self.single_action_space = self.env.single_action_space
        self.single_state_space = self.env.single_observation_space
        self.action_space = self.env.action_space
        self.state_space = self.env.observation_space
        self.state_dim = self._get_space_shape(self.single_state_space) 
        self.action_dim = self._get_space_shape(self.single_action_space)

    def _get_space_shape(self, space):
        # supports gym Discrete or Box
        if type(space) == gym.spaces.Discrete:
            # discrete spaces are 1 number/choice
            return tuple() 
        else:
            # box spaces have a shape 
            return space.shape

    def _make_envs(self):

        def _make_one_env():
            env = gym.make(self.env_name, **self.env_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            for name, wrapper_kwargs in self.wrappers.items():
                env = WRAPPERS[name](env, **(wrapper_kwargs or {}))
            return env
        
        env_list = [_make_one_env for env in range(self.num_envs)]
        env = gym.vector.SyncVectorEnv(env_list)
        return env

    def reset(self, reset_mask=None):
        seed_to_use = self.seed if  not self.has_reset else None
        self.has_reset = True

        options = {"reset_mask": reset_mask} if reset_mask is not None else None
        observation, info = self.env.reset(seed=seed_to_use, options=options)
        return observation, info

    def step(self, actions: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated, info

    def reset_finished_envs(self, sprime, is_terms, is_truncs):
        is_env_done = is_terms|is_truncs
        if not is_env_done.any():
            return sprime

        new_sprime, _ = self.reset(reset_mask=is_env_done)
        return new_sprime


