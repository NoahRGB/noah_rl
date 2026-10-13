# NoahRGB 09/2026
#
# UNFINISHED
# 
# ICM + PPO implementation
# (https://arxiv.org/abs/1705.05363)
# (http://arxiv.org/abs/1707.06347)
# mind the bugs
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/)
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

class ICM_ActorCriticNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple, action_space: gym.Space):
        super(ICM_ActorCriticNetwork, self).__init__()
        self.action_space = action_space

        self.icm_enc = Encoder(architecture["icm_enc"], input_shape=state_dim)
        self.icm_enc_output = self.icm_enc.encoder_output

        self.inv_enc = Encoder(architecture["icm_inv_enc"], input_shape=(self.icm_enc_output*2,)) 
        self.inv_enc_output = self.inv_enc.encoder_output
        self.inv_head = detect_head(action_space=action_space, input_size=self.inv_enc_output)

        icm_forward_input_size = self.icm_enc_output + (action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0])
        self.forward_enc = Encoder(architecture["icm_forward_enc"], input_shape=(icm_forward_input_size,)) 
        self.forward_enc_output = self.forward_enc.encoder_output
        self.forward_head = torch.nn.Linear(self.forward_enc_output, self.icm_enc_output)

        if "shared_ppo_enc" in architecture:
            self.using_shared_enc = True
            self.shared_enc = Encoder(architecture["shared_ppo_enc"], input_shape=state_dim)
            self.actor_enc_output = self.value_enc_output = self.shared_enc.encoder_output
        else:
            self.using_shared_enc = False
            self.actor_enc = Encoder(architecture["actor_enc"], input_shape=state_dim)
            self.value_enc = Encoder(architecture["value_enc"], input_shape=state_dim)
            self.actor_enc_output = self.actor_enc.encoder_output
            self.value_enc_output = self.value_enc.encoder_output
            
        self.actor_head = detect_head(action_space=action_space, input_size=self.actor_enc_output)
        self.value_head = torch.nn.Linear(self.value_enc_output, 1)

    def get_icm(self, s, sprime, a):
        phi_s = self.icm_enc(s)
        phi_sprime = self.icm_enc(sprime)
    
        # inverse model
        inv_enc_out = self.inv_enc(torch.concat([phi_s, phi_sprime], dim=-1))
        a_pred = self.inv_head(inv_enc_out)

        # forward model
        if isinstance(self.action_space, gym.spaces.Discrete):
            forward_inp = torch.concat([phi_s, torch.nn.functional.one_hot(a.long(), num_classes=self.action_space.n).float()], dim=-1)
        else:
            ...

        forward_enc_out = self.forward_enc(forward_inp)
        phi_sprime_pred = self.forward_head(forward_enc_out)

        return a_pred, phi_sprime, phi_sprime_pred

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
    network = ICM_ActorCriticNetwork(cfg.network_architecture, env.state_dim, env.single_action_space)
    optim = torch.optim.Adam(network.parameters(), lr=cfg.lr)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    obs_normaliser = Normaliser(env.state_dim)
    intrinsic_reward_normaliser = Normaliser((1,), use_mean=False)
    running_intrinsic_returns = np.zeros(cfg.num_envs)

    if cfg.load_path is not None:
        network_dict = torch.load(cfg.load_path, weights_only=False)
        network.load_state_dict(network_dict["net"])
        optim.load_state_dict(network_dict["optim"])
        running_intrinsic_returns = network_dict["running_intrinsic_returns"]
        intrinsic_reward_normaliser.load(network_dict["intrinsic_normaliser"])
        obs_normaliser.load(network_dict["obs_normaliser"])

    states, _ = env.reset()

    # need to randomly step through the environment in order to establish
    # a running mu/stdev for observations
    print("random stepping started")
    for warmup_step in range(cfg.warmup_steps):
        actions = env.action_space.sample()
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)

        obs_normaliser.add_batch(states)
        states = env.reset_finished_envs(sprimes, is_terms, is_truncs)
    print("random stepping done")

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

            stats = {"mean_rollout_intrinsic_rewards": []}
            # unroll over rollout_len steps and store in buffers
            for timestep in range(cfg.rollout_len):
                action_dist = network.get_actor(torch.from_numpy(states).float())
                actions = action_dist.sample()
                sprimes, rewards, is_terms, is_truncs, info = env.step(actions.cpu().numpy())

                logger.log_episodes(info)
                
                # compute intrinsic reward
                a_pred, phi_sprime, phi_sprime_pred = network.get_icm(torch.from_numpy(obs_normaliser.normalise(states, clip_range=[-5, 5])).float(),
                                                                      torch.from_numpy(obs_normaliser.normalise(sprimes, clip_range=[-5, 5])).float(),
                                                                      actions)
                intrinsic_reward = ((phi_sprime_pred - phi_sprime)**2).sum(dim=-1)
                intrinsic_reward_buffer[timestep] = intrinsic_reward
                stats["mean_rollout_intrinsic_rewards"].append(intrinsic_reward.mean().item())

                running_intrinsic_returns = running_intrinsic_returns * cfg.gamma + intrinsic_reward.cpu().numpy()
                intrinsic_reward_normaliser.add_batch(running_intrinsic_returns)

                states_buffer[timestep] = torch.from_numpy(states)
                actions_buffer[timestep] = actions
                rewards_buffer[timestep] = torch.from_numpy(rewards)
                next_states_buffer[timestep] = torch.from_numpy(sprimes)
                is_terms_buffer[timestep] = torch.from_numpy(is_terms)
                dones_buffer[timestep] = torch.from_numpy(is_terms|is_truncs)
                logprobs_buffer[timestep] = action_dist.log_prob(actions)

                states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

            logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
            # rollout is over, get state values across the batch
            T, N = cfg.rollout_len, cfg.num_envs
            values_buffer = network.get_value(states_buffer.reshape(T*N, *env.state_dim)).reshape(T, N)
            next_values_buffer = network.get_value(next_states_buffer.reshape(T*N, *env.state_dim)).reshape(T, N)

            intrinsic_reward_buffer = intrinsic_reward_normaliser.normalise(intrinsic_reward_buffer)

        total_reward = cfg.ext_coef * rewards_buffer + cfg.int_coef * intrinsic_reward_buffer 
        # calculate advantages+returns (using GAE)
        gae = 0.0
        advantages = torch.zeros_like(rewards_buffer)
        for t in reversed(range(cfg.rollout_len)):
            delta = total_reward[t] + cfg.gamma * next_values_buffer[t] * (1 - is_terms_buffer[t]) - values_buffer[t] 
            gae = delta + cfg.gamma * cfg.lam * (1 - dones_buffer[t]) * gae
            advantages[t] = gae
        returns = advantages + values_buffer

        # once advantages have been calculated, the time dimension is no longer
        # needed and everything can be flattened
        states_buffer = states_buffer.reshape(T*N, *env.state_dim)
        next_states_buffer = next_states_buffer.reshape(T*N, *env.state_dim)
        actions_buffer = actions_buffer.reshape(T*N, *env.action_dim)
        logprobs_buffer = logprobs_buffer.reshape(T*N)
        advantages = advantages.reshape(T*N)
        values_buffer = values_buffer.reshape(T*N)
        returns = returns.reshape(T*N)

        obs_normaliser.add_batch(states_buffer.cpu().numpy())

        # optimise the network over epochs in minibatches
        # calculate new values/log probs, compute the clipped obj
        # and backpropagate
        stats = {"policy_loss":[], "value_loss":[], "inverse_loss": [], "forward_loss": []}
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

                a_dist_pred, phi_sprime, phi_sprime_pred = network.get_icm(obs_normaliser.normalise(states_buffer[mb_indices], clip_range=[-5, 5]).float(),
                                                                           obs_normaliser.normalise(next_states_buffer[mb_indices], clip_range=[-5, 5]).float(),
                                                                           actions_buffer[mb_indices])
                inverse_loss = -a_dist_pred.log_prob(actions_buffer[mb_indices].long()).mean()
                forward_loss = 0.5 * ((phi_sprime_pred - phi_sprime.detach())**2).sum(dim=-1).mean() 

                entropy_bonus = new_dist.entropy().mean()
                policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean()
                policy_loss -= (cfg.entropy_weight * entropy_bonus)
                value_loss = torch.nn.functional.mse_loss(new_values, returns[mb_indices])
                ppo_loss = policy_loss + cfg.value_weight * value_loss

                combined_loss = cfg.icm_lam * ppo_loss + (1 - cfg.beta) * inverse_loss + cfg.beta * forward_loss 

                optim.zero_grad()
                combined_loss.backward()
                if cfg.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.cgn)
                optim.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["inverse_loss"].append(inverse_loss.item())
                stats["forward_loss"].append(forward_loss.item())

        logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
        if cfg.save_network: logger.log_network({"net": network.state_dict(), "optim": optim.state_dict(),
                                                 "running_intrinsic_returns": running_intrinsic_returns,
                                                 "obs_normaliser": obs_normaliser.save(),
                                                 "intrinsic_normaliser": intrinsic_reward_normaliser.save()})




@hydra.main(config_path="../../params", config_name="icm_ppo", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train(cfg, run_dir)

if __name__ == "__main__":
    main()

