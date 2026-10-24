# NoahRGB 09/2026
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/)
#

from omegaconf import DictConfig
import torch
import gymnasium as gym
import numpy as np
from collections import deque

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv
from noah_rl.utils.logging import Logger


class EpisodicReplayMemory:
    # stores whole episodes and samples windows within the stored
    # episodes
    def __init__(self, size: int, state_dim: tuple, action_dim: tuple, state_dtype):
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dtype = state_dtype

        self.episodes = deque()
        self.total_stored_timesteps = 0

    def add(self, states, actions, rewards, next_states):
        new_episode_length = states.shape[0]
        while self.episodes and self.total_stored_timesteps + new_episode_length > self.max_size:
            removed_ep = self.episodes.popleft()
            self.total_stored_timesteps -= removed_ep[0].shape[0]

        self.total_stored_timesteps += new_episode_length
        self.episodes.append((states, actions, rewards, next_states))

    def sample(self, count, window_size, device):
        lengths = np.array([ep[0].shape[0] - window_size + 1 for ep in self.episodes])
        valid = np.clip(lengths, 0, None)

        if valid.sum() == 0:
            return tuple(), 0

        probs = valid / valid.sum()
        idx = np.random.choice(len(self.episodes), size=count, p=probs)

        windows = []
        for i in idx: 
            episode = self.episodes[i]
            episode_len = episode[0].shape[0]
            start_idx = np.random.randint(0, valid[i])
            windows.append(tuple([arr[start_idx:start_idx+window_size] for arr in episode]))

        return tuple(torch.from_numpy(np.stack(field)).to(device) for field in zip(*windows)), count 




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



class WorldModel(torch.nn.Module):
    def __init__(self, architecture: dict, action_dim: tuple, state_dim: tuple, action_space: gym.Space, latent_dim: int, hidden_dim: int):
        super(WorldModel, self).__init__()

        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim 
        action_input_size = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]

        # the ENCODER part of the autoencoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(state_dim[0], 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(), # 1024
        )

        self.posterior_net = torch.nn.Sequential(
            torch.nn.Linear(1024+self.hidden_dim, 256),
            torch.nn.ReLU(),
        )
        self.posterior_mu_head = torch.nn.Linear(256, self.latent_dim)
        self.posterior_sigma_head = torch.nn.Linear(256, self.latent_dim)

        # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.rnn = torch.nn.GRUCell(self.latent_dim+action_input_size, self.hidden_dim)


        self.prior_net = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.prior_mu_head = torch.nn.Linear(256, self.latent_dim)
        self.prior_sigma_head = torch.nn.Linear(256, self.latent_dim)

        # the DECODER part of the autoencoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim+self.hidden_dim, 1024),
            torch.nn.Unflatten(-1, unflattened_size=(1024, 1, 1)),
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            torch.nn.Sigmoid(),
        )

    def posterior(self, inp):
        net_out = self.posterior_net(inp)
        mu = self.posterior_mu_head(net_out)
        sigma = torch.nn.functional.softplus(self.posterior_sigma_head(net_out)) + 0.1
        dist = torch.distributions.Normal(mu, sigma)
        return dist

    def prior(self, h):
        net_out = self.prior_net(h)
        mu = self.prior_mu_head(net_out)
        sigma = torch.nn.functional.softplus(self.prior_sigma_head(net_out)) + 0.1
        dist = torch.distributions.Normal(mu, sigma)
        return dist


def train(cfg: DictConfig, hydra_dir: str):

    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_model = WorldModel(cfg.network_architecture, env.action_dim, env.state_dim, env.single_action_space, cfg.latent_dim, cfg.hidden_dim).to(device)
    world_model_optim = torch.optim.Adam(world_model.parameters(), lr=cfg.lr)
    
    # replay = ReplayMemory(size=cfg.replay_size, state_dim=env.state_dim, action_dim=env.action_dim, state_dtype=env.single_state_space.dtype)
    replay = EpisodicReplayMemory(size=cfg.replay_size, state_dim=env.state_dim, action_dim=env.action_dim, state_dtype=env.single_state_space.dtype)

    if cfg.load_path is not None:
        network_dict = torch.load(cfg.load_path, map_location=device, weights_only=False)
        world_model.load_state_dict(network_dict["world_model"])
        if "world_model_optim" in network_dict:
            world_model_optim.load_state_dict(network_dict["world_model_optim"])

    episodic_states, episodic_rewards, episodic_actions, episodic_next_states = [], [], [], []

    stat_names = ["world_model_loss", "reconstruction_loss", "reconstruction_mse", "kl", "kl_loss", "prior_std", "posterior_std"]
    stats = {name: [] for name in stat_names}

    states, _ = env.reset()
    for timestep in range(cfg.timesteps):
        actions = env.action_space.sample() 
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)

        logger.log_episodes(info)

        episodic_states.append(states[0])
        episodic_rewards.append(rewards[0])
        episodic_actions.append(actions[0])
        episodic_next_states.append(sprimes[0])

        if "episode" in info:
            # when episode ends, store whole episoide
            replay.add(
                np.array(episodic_states),
                np.array(episodic_actions),
                np.array(episodic_rewards),
                np.array(episodic_next_states),
            )
            episodic_states, episodic_rewards, episodic_actions, episodic_next_states = [], [], [], []

        states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

        # update world model
        if timestep % cfg.world_model_update_interval == 0:

            windows, sample_size = replay.sample(cfg.minibatch_size, cfg.window_size, device)
            if sample_size == cfg.minibatch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states = windows # (minibatch_size, window_size, ..., )

                batch_states = batch_states.float() / cfg.state_scale
                batch_next_states = batch_next_states.float() / cfg.state_scale
                actions_one_hot = torch.nn.functional.one_hot(batch_actions.long(), num_classes=env.single_action_space.n).float()
                actions_shifted = torch.cat([torch.zeros_like(actions_one_hot[:, :1]), actions_one_hot[:, :-1]], dim=1)

                encoded = world_model.encoder(batch_states.flatten(0, 1)).unflatten(0, (cfg.minibatch_size, cfg.window_size))
                hidden_state = torch.zeros(cfg.minibatch_size, cfg.hidden_dim).to(device)
                latent = torch.zeros(cfg.minibatch_size, cfg.latent_dim).to(device)

                latents, hidden_states, prior_mus, prior_stddevs, posterior_mus, posterior_stddevs = [], [], [], [], [], []
                for t in range(cfg.window_size):
                    hidden_state = world_model.rnn(torch.cat([latent, actions_shifted[:, t]], dim=-1), hidden_state)
                    prior = world_model.prior(hidden_state)
                    posterior = world_model.posterior(torch.cat([hidden_state, encoded[:, t]], dim=-1))
                    latent = posterior.rsample()

                    hidden_states.append(hidden_state); latents.append(latent)
                    prior_mus.append(prior.mean); prior_stddevs.append(prior.stddev)
                    posterior_mus.append(posterior.mean); posterior_stddevs.append(posterior.stddev)

                hidden_states = torch.stack(hidden_states, 1)
                latents = torch.stack(latents, 1)
                priors = torch.distributions.Normal(torch.stack(prior_mus, 1), torch.stack(prior_stddevs, 1))
                posteriors = torch.distributions.Normal(torch.stack(posterior_mus, 1), torch.stack(posterior_stddevs, 1))

                reconstructions = torch.distributions.Normal(world_model.decoder(torch.cat([latents, hidden_states], dim=-1).flatten(0, 1)), cfg.recon_std)
                reconstruction_loss = -reconstructions.log_prob(batch_states.flatten(0, 1)).sum((1, 2, 3)).mean()
                kl = torch.distributions.kl_divergence(posteriors, priors).sum(-1)
                kl_loss = torch.clamp(kl, min=3.0).mean()

                world_model_loss = reconstruction_loss + kl_loss
                    
                world_model_optim.zero_grad()
                world_model_loss.backward()
                world_model_optim.step()

                stats["world_model_loss"].append(world_model_loss.item())
                stats["reconstruction_loss"].append(reconstruction_loss.item())
                stats["reconstruction_mse"].append(((reconstructions.mean - batch_states.flatten(0, 1)) ** 2).mean().item())
                stats["kl"].append(kl.mean().item())
                stats["kl_loss"].append(kl_loss.item())
                stats["prior_std"].append(priors.stddev.mean().item())
                stats["posterior_std"].append(posteriors.stddev.mean().item())

        if timestep % cfg.log_interval == 0 and len(stats["world_model_loss"]) > 0:
            logger.log_stats({name: np.mean(stat) for name, stat in stats.items()})
            stats = {name: [] for name in stat_names}

        if cfg.save_network and timestep % cfg.save_interval == 0: logger.log_network({
            "world_model": world_model.state_dict(), "world_model_optim": world_model_optim.state_dict(),
        })

    if cfg.save_network: logger.log_network({
        "world_model": world_model.state_dict(), "world_model_optim": world_model_optim.state_dict(),
    })

