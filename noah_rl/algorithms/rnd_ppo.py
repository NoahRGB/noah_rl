# NoahRGB 09/2026
# PPO implementation with RND for intrinsic motivation/exploration 
# (https://arxiv.org/abs/1810.12894)
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
from noah_rl.utils.normalisation import Normaliser

class RNDTargetNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple):
        super(RNDTargetNetwork, self).__init__()

        self.target_enc = Encoder(architecture["target_enc"], input_shape=state_dim)
        self.target_enc_output = self.target_enc.encoder_output
        self.target_head = torch.nn.Linear(self.target_enc_output, 64)

    def forward(self, inp):
        target_enc_out = self.target_enc(inp)
        return self.target_head(target_enc_out)

class RNDPredictorNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple):
        super(RNDPredictorNetwork, self).__init__()

        self.predictor_enc = Encoder(architecture["predictor_enc"], input_shape=state_dim)
        self.predictor_enc_output = self.predictor_enc.encoder_output
        self.predictor_head = torch.nn.Linear(self.predictor_enc_output, 64)

    def forward(self, inp):
        predictor_enc_out = self.predictor_enc(inp)
        return self.predictor_head(predictor_enc_out)

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
        self.extrinsic_value_head = torch.nn.Linear(self.value_enc_output, 1)
        self.intrinsic_value_head = torch.nn.Linear(self.value_enc_output, 1)

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
        return self.extrinsic_value_head(enc_out), self.intrinsic_value_head(enc_out)    

def train(cfg: DictConfig, hydra_dir: str):
    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    network = ActorCriticNetwork(cfg.network_architecture, env.state_dim, env.single_action_space)
    target_network = RNDTargetNetwork(cfg.network_architecture, env.state_dim)
    predictor_network = RNDPredictorNetwork(cfg.network_architecture, env.state_dim)
    optim = torch.optim.Adam(network.parameters(), lr=cfg.lr)
    predictor_optim = torch.optim.Adam(predictor_network.parameters(), lr=cfg.lr)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    obs_normaliser = Normaliser(env.state_dim)
    intrinsic_reward_normaliser = Normaliser((1,), use_mean=False)
    running_intrinsic_returns = np.zeros(cfg.num_envs)

    if cfg.load_path is not None:
        network_dict = torch.load(cfg.load_path, weights_only=False)
        network.load_state_dict(network_dict["net"])
        optim.load_state_dict(network_dict["optim"])
        target_network.load_state_dict(network_dict["target"])
        predictor_network.load_state_dict(network_dict["predictor"])
        predictor_optim.load_state_dict(network_dict["predictor_optim"])
        running_intrinsic_returns = network_dict["running_intrinsic_returns"]
        intrinsic_reward_normaliser.load(network_dict["intrinsic_normaliser"])
        obs_normaliser.load(network_dict["obs_normaliser"])

    states, _ = env.reset()

    # need to randomly step through the environment in order to establish
    # a running mu/stdev for observations
    for warmup_step in range(cfg.warmup_steps):
        actions = env.action_space.sample()
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)
        obs_normaliser.add_batch(states)
        states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

    num_rollouts = cfg.timesteps // (cfg.rollout_len * cfg.num_envs)
    for rollout in range(num_rollouts):

        states_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.state_dim), dtype=torch.float32)
        actions_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.action_dim), dtype=torch.float32)
        rewards_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        next_states_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.state_dim), dtype=torch.float32)
        is_terms_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        dones_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        logprobs_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        intrinsic_reward_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)

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

                target_out = target_network(torch.from_numpy(obs_normaliser.normalise(sprimes, clip_range=[-5, 5])).float())
                predictor_out = predictor_network(torch.from_numpy(obs_normaliser.normalise(sprimes, clip_range=[-5, 5])).float())
                # for each sub env in cfg.num_envs, what is the intrinsic reward 
                intrinsic_reward = ((predictor_out - target_out)**2).sum(-1)
                intrinsic_reward_buffer[timestep] = intrinsic_reward

                # update the intrinsic reward normaliser + running return calculation
                running_intrinsic_returns = running_intrinsic_returns * cfg.intrinsic_gamma + intrinsic_reward.cpu().numpy()
                intrinsic_reward_normaliser.add_batch(running_intrinsic_returns)
                logger.log_stats({"raw_intrinsic_reward": intrinsic_reward.mean()})

            # rollout is over, get state values across the batch
            T, N = cfg.rollout_len, cfg.num_envs
            extrinsic_values_buffer, intrinsic_values_buffer = network.get_value(states_buffer.reshape(T*N, *env.state_dim))
            extrinsic_values_buffer = extrinsic_values_buffer.reshape(T, N)
            intrinsic_values_buffer = intrinsic_values_buffer.reshape(T, N)

            extrinsic_next_values_buffer, intrinsic_next_values_buffer = network.get_value(next_states_buffer.reshape(T*N, *env.state_dim))
            extrinsic_next_values_buffer = extrinsic_next_values_buffer.reshape(T, N)
            intrinsic_next_values_buffer = intrinsic_next_values_buffer.reshape(T, N)

            intrinsic_reward_buffer = intrinsic_reward_normaliser.normalise(intrinsic_reward_buffer)

        # calculate EXTRINSIC advantages+returns (using GAE)
        gae = 0.0
        extrinsic_advantages = torch.zeros_like(rewards_buffer)
        for t in reversed(range(cfg.rollout_len)):
            delta = rewards_buffer[t] + cfg.extrinsic_gamma * extrinsic_next_values_buffer[t] * (1 - is_terms_buffer[t]) - extrinsic_values_buffer[t] 
            gae = delta + cfg.extrinsic_gamma * cfg.lam * (1 - dones_buffer[t]) * gae
            extrinsic_advantages[t] = gae
        extrinsic_returns = extrinsic_advantages + extrinsic_values_buffer

        # calculate INTRINSIC advantages+returns (using GAE)
        gae = 0.0
        intrinsic_advantages = torch.zeros_like(rewards_buffer)
        for t in reversed(range(cfg.rollout_len)):
            delta = intrinsic_reward_buffer[t] + cfg.intrinsic_gamma * intrinsic_next_values_buffer[t] - intrinsic_values_buffer[t] 
            gae = delta + cfg.intrinsic_gamma * cfg.lam * gae
            intrinsic_advantages[t] = gae
        intrinsic_returns = intrinsic_advantages + intrinsic_values_buffer

        # once advantages have been calculated, the time dimension is no longer
        # needed and everything can be flattened
        states_buffer = states_buffer.reshape(T*N, *env.state_dim)
        next_states_buffer = next_states_buffer.reshape(T*N, *env.state_dim)
        actions_buffer = actions_buffer.reshape(T*N, *env.action_dim)
        logprobs_buffer = logprobs_buffer.reshape(T*N)
        extrinsic_advantages = extrinsic_advantages.reshape(T*N)
        intrinsic_advantages = intrinsic_advantages.reshape(T*N)
        combined_advantages = cfg.extrinsic_adv_coef * extrinsic_advantages + cfg.intrinsic_adv_coef * intrinsic_advantages 
        extrinsic_values_buffer = extrinsic_values_buffer.reshape(T*N)
        intrinsic_values_buffer = intrinsic_values_buffer.reshape(T*N)
        extrinsic_returns = extrinsic_returns.reshape(T*N)
        intrinsic_returns = intrinsic_returns.reshape(T*N)

        obs_normaliser.add_batch(states_buffer.cpu().numpy())

        # optimise the network over epochs in minibatches
        # calculate new values/log probs, compute the clipped obj
        # and backpropagate
        stats = {"policy_loss":[], "extrinsic_value_loss":[], "intrinsic_value_loss":[], "rnd_loss":[]}
        for epoch in range(cfg.epochs):
            batch_size = T*N
            all_indices = np.arange(batch_size)
            np.random.shuffle(all_indices)
            for mb_start in range(0, batch_size, cfg.minibatch_size):
                mb_indices = all_indices[mb_start:mb_start+cfg.minibatch_size]
                mb_combined_advantages = combined_advantages[mb_indices]
                mb_combined_advantages = (mb_combined_advantages - mb_combined_advantages.mean()) / (mb_combined_advantages.std() + 1e-8)

                new_extrinsic_values, new_intrinsic_values = network.get_value(states_buffer[mb_indices])
                new_extrinsic_values = new_extrinsic_values.squeeze(-1)
                new_intrinsic_values = new_intrinsic_values.squeeze(-1)

                new_dist = network.get_actor(states_buffer[mb_indices])
                new_log_prob = new_dist.log_prob(actions_buffer[mb_indices])

                ratios = torch.exp(new_log_prob - logprobs_buffer[mb_indices])
                clipped_ratios = torch.clamp(ratios, 1 - cfg.epsilon, 1 + cfg.epsilon)
                surrogate_obj = ratios * mb_combined_advantages 
                clipped_surrogate_obj = clipped_ratios * mb_combined_advantages 

                entropy_bonus = new_dist.entropy().mean()
                policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean()
                policy_loss -= (cfg.entropy_weight * entropy_bonus)
                extrinsic_value_loss = torch.nn.functional.mse_loss(new_extrinsic_values, extrinsic_returns[mb_indices])
                intrinsic_value_loss = torch.nn.functional.mse_loss(new_intrinsic_values, intrinsic_returns[mb_indices])
                combined_loss = policy_loss + cfg.value_weight * (extrinsic_value_loss + intrinsic_value_loss) 

                optim.zero_grad()
                combined_loss.backward()
                if cfg.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.cgn)
                optim.step()

                mb_target_out = target_network(obs_normaliser.normalise(next_states_buffer[mb_indices], clip_range=[-5, 5]).float())
                mb_predictor_out = predictor_network(obs_normaliser.normalise(next_states_buffer[mb_indices], clip_range=[-5, 5]).float())
                rnd_loss = ((mb_predictor_out - mb_target_out)**2).sum(-1).mean()

                predictor_optim.zero_grad()
                rnd_loss.backward()
                predictor_optim.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["extrinsic_value_loss"].append(extrinsic_value_loss.item())
                stats["intrinsic_value_loss"].append(intrinsic_value_loss.item())
                stats["rnd_loss"].append(rnd_loss.item())

        logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
        if cfg.save_network: logger.log_network({"net": network.state_dict(), "optim": optim.state_dict(),
                                                 "target": target_network.state_dict(), "predictor": predictor_network.state_dict(),
                                                 "predictor_optim": predictor_optim.state_dict(), "running_intrinsic_returns": running_intrinsic_returns,
                                                 "obs_normaliser": obs_normaliser.save(), "intrinsic_normaliser": intrinsic_reward_normaliser.save()})




@hydra.main(config_path="../../params", config_name="rnd_ppo", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train(cfg, run_dir)

if __name__ == "__main__":
    main()

