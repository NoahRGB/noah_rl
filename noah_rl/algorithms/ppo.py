# NoahRGB 09/2026
# PPO implementation (https://arxiv.org/abs/1707.06347) 
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/ppo.yaml)
#

import hydra
from omegaconf import DictConfig
import gymnasium as gym
import torch
import numpy as np

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv
from noah_rl.utils.networks import Encoder, detect_head
from noah_rl.utils.logging import Logger

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple, action_space: gym.Space):
        super(ActorCriticNetwork, self).__init__()

        if "shared_enc" in architecture:
            self.using_shared_enc = True
            self.shared_enc = Encoder(architecture["shared_enc"], input_shape=state_dim)
            self.actor_enc_output = self.value_enc_output = self.shared_enc.encoder_output
        else:
            self.using_shared_enc = False
            self.actor_enc = Encoder(architecture["actor_enc"], input_shape=state_dim)
            self.value_enc = Encoder(architecture["value_enc"], input_shape=state_dim)
            self.actor_enc_output = self.actor_enc.encoder_output
            self.value_enc_output = self.value_enc.encoder_output
            
        self.actor_head = detect_head(action_space=action_space, input_size=self.actor_enc_output)
        self.value_head = torch.nn.Linear(self.value_enc_output, 1)

    def get_actor(self, inp):
        if self.using_shared_enc:
            enc_out = self.shared_enc(inp)
        else:
            enc_out = self.actor_enc(inp)
        return self.actor_head(enc_out)
    
    def get_value(self, inp):
        if self.using_shared_enc:
            enc_out = self.shared_enc(inp)
        else:
            enc_out = self.value_enc(inp)
        return self.value_head(enc_out)    

def train(cfg: DictConfig, hydra_dir: str):
    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    network = ActorCriticNetwork(cfg.network_architecture, env.state_dim, env.single_action_space)
    optim = torch.optim.Adam(network.parameters(), lr=cfg.lr)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    if cfg.load_path is not None:
        network_dict = torch.load(cfg.load_path, weights_only=False)
        network.load_state_dict(network_dict["net"])
        optim.load_state_dict(network_dict["optim"])

    states, _ = env.reset()
    num_rollouts = cfg.timesteps // (cfg.rollout_len * cfg.num_envs)
    for rollout in range(num_rollouts):

        states_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.state_dim), dtype=torch.float32)
        actions_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.action_dim), dtype=torch.float32)
        rewards_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        next_states_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.state_dim), dtype=torch.float32)
        is_terms_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        dones_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        logprobs_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)

        with torch.no_grad():
            # unroll over rollout_len steps and store in buffers
            for timestep in range(cfg.rollout_len):
                action_dist = network.get_actor(torch.from_numpy(states).float())
                actions = action_dist.sample()
                sprimes, rewards, is_terms, is_truncs, info = env.step(actions.cpu().numpy())

                logger.log_episodes(info)

                states_buffer[timestep] = torch.from_numpy(states)
                actions_buffer[timestep] = actions
                rewards_buffer[timestep] = torch.from_numpy(rewards)
                next_states_buffer[timestep] = torch.from_numpy(sprimes)
                is_terms_buffer[timestep] = torch.from_numpy(is_terms)
                dones_buffer[timestep] = torch.from_numpy(is_terms|is_truncs)
                logprobs_buffer[timestep] = action_dist.log_prob(actions)

                states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

            # rollout is over, get state values across the batch
            T, N = cfg.rollout_len, cfg.num_envs
            values_buffer = network.get_value(states_buffer.reshape(T*N, *env.state_dim)).reshape(T, N)
            next_values_buffer = network.get_value(next_states_buffer.reshape(T*N, *env.state_dim)).reshape(T, N)

        # calculate advantages+returns (using GAE)
        gae = 0.0
        advantages = torch.zeros_like(rewards_buffer)
        for t in reversed(range(cfg.rollout_len)):
            delta = rewards_buffer[t] + cfg.gamma * next_values_buffer[t] * (1 - is_terms_buffer[t]) - values_buffer[t] 
            gae = delta + cfg.gamma * cfg.lam * (1 - dones_buffer[t]) * gae
            advantages[t] = gae
        returns = advantages + values_buffer

        # once advantages have been calculated, the time dimension is no longer
        # needed and everything can be flattened
        states_buffer = states_buffer.reshape(T*N, *env.state_dim)
        actions_buffer = actions_buffer.reshape(T*N, *env.action_dim)
        logprobs_buffer = logprobs_buffer.reshape(T*N)
        advantages = advantages.reshape(T*N)
        values_buffer = values_buffer.reshape(T*N)
        returns = returns.reshape(T*N)

        # optimise the network over epochs in minibatches
        # calculate new values/log probs, compute the clipped obj
        # and backpropagate
        stats = {"policy_loss":[], "value_loss":[]}
        for epoch in range(cfg.epochs):
            batch_size = T*N
            all_indices = np.arange(batch_size)
            np.random.shuffle(all_indices)
            for mb_start in range(0, batch_size, cfg.minibatch_size):
                mb_indices = all_indices[mb_start:mb_start+cfg.minibatch_size]
                mb_advantages = advantages[mb_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_values = network.get_value(states_buffer[mb_indices]).squeeze(-1)
                new_dist = network.get_actor(states_buffer[mb_indices])
                new_log_prob = new_dist.log_prob(actions_buffer[mb_indices])

                ratios = torch.exp(new_log_prob - logprobs_buffer[mb_indices])
                clipped_ratios = torch.clamp(ratios, 1 - cfg.epsilon, 1 + cfg.epsilon)
                surrogate_obj = ratios * mb_advantages 
                clipped_surrogate_obj = clipped_ratios * mb_advantages 

                entropy_bonus = new_dist.entropy().mean()
                policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean()
                policy_loss -= (cfg.entropy_weight * entropy_bonus)
                value_loss = torch.nn.functional.mse_loss(new_values, returns[mb_indices])
                combined_loss = policy_loss + cfg.value_weight * value_loss

                optim.zero_grad()
                combined_loss.backward()
                if cfg.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.cgn)
                optim.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())

        logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
        if cfg.save_network: logger.log_network({"net": network.state_dict(), "optim": optim.state_dict()})




@hydra.main(config_path="../../params", config_name="ppo", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train(cfg, run_dir)

if __name__ == "__main__":
    main()
