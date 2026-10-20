# NoahRGB 10/2026
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/ppo.yaml)
#

from omegaconf import DictConfig
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv
from noah_rl.utils.networks import Encoder, detect_head
from noah_rl.utils.logging import Logger

class ReplayMemory:
    # stores s, a, r, s', is_terminated, is_truncated
    # for X transitions
    # 
    # will overwrite old transitions with new ones when max capacity
    # is reached

    def __init__(self, size: int, state_dim: tuple, action_dim: tuple, state_dtype):
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dtype = state_dtype

        self.current_size = 0
        self.pointer = 0
        self.reset()

    def reset(self):
        self.states_buffer = np.empty((self.max_size, *self.state_dim), dtype=self.state_dtype)
        self.actions_buffer = np.empty((self.max_size, *self.action_dim), dtype=np.float32)
        self.rewards_buffer = np.empty((self.max_size), dtype=np.float32)
        self.next_states_buffer = np.empty((self.max_size, *self.state_dim), dtype=self.state_dtype)
        self.is_terms_buffer = np.empty((self.max_size), dtype=np.float32)
        self.is_truncs_buffer = np.empty((self.max_size), dtype=np.float32)

    def add(self, states, actions, rewards, next_states, is_terms, is_truncs):
        batch_dim = states.shape[0]
        for transition in range(batch_dim):
            self.states_buffer[self.pointer] = states[transition]
            self.actions_buffer[self.pointer] = actions[transition]
            self.rewards_buffer[self.pointer] = rewards[transition]
            self.next_states_buffer[self.pointer] = next_states[transition]
            self.is_terms_buffer[self.pointer] = is_terms[transition]
            self.is_truncs_buffer[self.pointer] = is_truncs[transition]
            self.pointer = (self.pointer + 1) % self.max_size
            self.current_size = min(self.current_size+1, self.max_size)

    def sample(self, size, device="cpu"):
        sample_size = min(size, self.current_size)
        indices = np.random.choice(self.current_size, size=sample_size, replace=False)
        return (torch.from_numpy(self.states_buffer[indices]).to(device),
                torch.from_numpy(self.actions_buffer[indices]).to(device),
                torch.from_numpy(self.rewards_buffer[indices]).to(device),
                torch.from_numpy(self.next_states_buffer[indices]).to(device),
                torch.from_numpy(self.is_terms_buffer[indices]).to(device),
                torch.from_numpy(self.is_truncs_buffer[indices]).to(device)), sample_size


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple, action_space: gym.Space, inp_scale: float):
        super(ActorCriticNetwork, self).__init__()
        self.inp_scale = inp_scale

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
        norm_inp = inp.float() / self.inp_scale
        if self.using_shared_enc:
            enc_out = self.shared_enc(norm_inp)
        else:
            enc_out = self.actor_enc(norm_inp)
        return self.actor_head(enc_out)
    
    def get_value(self, inp):
        norm_inp = inp.float() / self.inp_scale
        if self.using_shared_enc:
            enc_out = self.shared_enc(norm_inp)
        else:
            enc_out = self.value_enc(norm_inp)
        return self.value_head(enc_out)

class WorldModel(torch.nn.Module):
    def __init__(self, architecture: dict, action_dim: tuple, state_dim: tuple, action_space: gym.Space):
        super(WorldModel, self).__init__()

        self.latent_dim = 50

        # the ENCODER part of the autoencoder
        # (4, 84, 84) frame stack -> latent z
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, self.latent_dim),
            torch.nn.LayerNorm(self.latent_dim),
            torch.nn.Tanh(),
        )

        # the PREDICTION part of the world model
        # take latent representation z + action and predict
        # the next latent z

        action_input_size = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim+action_input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.latent_dim),
            torch.nn.LayerNorm(self.latent_dim),
            torch.nn.Tanh(),
        )

        # the DECODER part of the autoencoder
        # latent z -> (1, 84, 84) single frame
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 3136),
            torch.nn.ReLU(),
            torch.nn.Unflatten(-1, unflattened_size=(64, 7, 7)),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=8, stride=4),
            torch.nn.Sigmoid(),
        )

def train(cfg: DictConfig, hydra_dir: str):
    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    network = ActorCriticNetwork(cfg.network_architecture, env.state_dim, env.single_action_space, cfg.state_scale)
    optim = torch.optim.Adam(network.parameters(), lr=cfg.lr)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)
    state_dtype = torch.from_numpy(np.empty(0, dtype=env.state_space.dtype)).dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_model = WorldModel(cfg.network_architecture, env.action_dim, env.state_dim, env.single_action_space).to(device)
    world_model_optim = torch.optim.Adam(world_model.parameters(), lr=cfg.lr)
    
    replay = ReplayMemory(size=cfg.replay_size, state_dim=env.state_dim, action_dim=env.action_dim, state_dtype=env.single_state_space.dtype)

    if cfg.load_path is not None:
        network_dict = torch.load(cfg.load_path, weights_only=False)
        network.load_state_dict(network_dict["net"])
        optim.load_state_dict(network_dict["optim"])
        # checkpoints from plain ppo won't have a world model
        if "world_model" in network_dict:
            world_model.load_state_dict(network_dict["world_model"])
            world_model_optim.load_state_dict(network_dict["world_model_optim"])

    states, _ = env.reset()
    num_rollouts = cfg.timesteps // (cfg.rollout_len * cfg.num_envs)
    for rollout in range(num_rollouts):

        states_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.state_dim), dtype=state_dtype)
        actions_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.action_dim), dtype=torch.float32)
        rewards_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        next_states_buffer = torch.empty((cfg.rollout_len, cfg.num_envs, *env.state_dim), dtype=state_dtype)
        is_terms_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        dones_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)
        logprobs_buffer = torch.empty((cfg.rollout_len, cfg.num_envs), dtype=torch.float32)

        with torch.no_grad():
            # unroll over rollout_len steps and store in buffers
            for timestep in range(cfg.rollout_len):
                action_dist = network.get_actor(torch.from_numpy(states))
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

                replay.add(states, actions, rewards, sprimes, is_terms, is_truncs)

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

        # update world model

        stats["world_model_loss"] = []
        for _ in range(cfg.model_updates_per_rollout):
            (batch_states, batch_actions, _, batch_next_states, _, _), sample_size = replay.sample(cfg.minibatch_size, device)
            if sample_size == cfg.minibatch_size:

                batch_states = batch_states.float() / cfg.state_scale
                batch_next_states = batch_next_states.float() / cfg.state_scale
                actions_one_hot = torch.nn.functional.one_hot(batch_actions.long(), num_classes=env.single_action_space.n).float()
                
                z = world_model.encoder(batch_states)
                z_next = world_model.encoder(batch_next_states)
                z_next_pred = world_model.predictor(torch.cat([z, actions_one_hot], -1))

                # how good is the decoder at decoding s_t --> z_t back into z_t --> s_t?
                decoder_loss = ((world_model.decoder(z) - batch_states[:, -1:])**2).mean()

                # how good is the model "end to end"? reconstruct the prediction of s_{t+1} from latent space
                # z using the decoder
                predictor_loss = ((world_model.decoder(z_next_pred) - batch_next_states[:, -1:])**2).mean()

                # how accurate is the "prediction" part of the model? the error in the prediction of latent z
                latent_loss = ((z_next_pred - z_next.detach())**2).mean()

                world_model_loss = decoder_loss + predictor_loss + latent_loss
                world_model_optim.zero_grad()
                world_model_loss.backward()
                world_model_optim.step()

                stats["world_model_loss"].append(world_model_loss.item())

        logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
        if cfg.save_network: logger.log_network({
            "net": network.state_dict(), "optim": optim.state_dict(),
            "world_model": world_model.state_dict(), "world_model_optim": world_model_optim.state_dict(),
        })





    # test world model
    (batch_states, batch_actions, _, batch_next_states, _, _), n = replay.sample(4, device)
    with torch.no_grad():

        batch_states = batch_states.float() / cfg.state_scale
        batch_next_states = batch_next_states.float() / cfg.state_scale
        actions_one_hot = torch.nn.functional.one_hot(batch_actions.long(), num_classes=env.single_action_space.n).float()
            
        z = world_model.encoder(batch_states)
        z_next = world_model.encoder(batch_next_states)

        imagined_next_state = world_model.predictor(torch.cat([z, actions_one_hot], dim=-1))
        decoded_imagined_next_state = world_model.decoder(imagined_next_state)
        decoded_real_next_state = world_model.decoder(z_next)
    
    panels = {"current": batch_states[:, -1], "true next": batch_next_states[:, -1],
            "recon next": decoded_real_next_state[:, 0], "imagined next": decoded_imagined_next_state[:, 0]}
    f, axarr = plt.subplots(n, len(panels), squeeze=False)
    for i in range(n):
        for j, (name, imgs) in enumerate(panels.items()):
            axarr[i, j].imshow(imgs[i].cpu(), cmap="gray", vmin=0, vmax=1)
            axarr[i, j].axis("off")
            if i == 0: axarr[i, j].set_title(name)
    plt.show()