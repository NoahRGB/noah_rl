# NoahRGB 09/2026
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/)
#

from omegaconf import DictConfig
import torch
import gymnasium as gym
import numpy as np

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv
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



class WorldModel(torch.nn.Module):
    def __init__(self, architecture: dict, action_dim: tuple, state_dim: tuple, action_space: gym.Space, latent_dim: int):
        super(WorldModel, self).__init__()

        self.latent_dim = latent_dim 

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
            torch.nn.Flatten(),
        )

        self.vae_mu_head = torch.nn.Linear(1024, self.latent_dim)
        self.vae_sigma_head = torch.nn.Linear(1024, self.latent_dim)

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
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Unflatten(-1, unflattened_size=(256, 2, 2)),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            torch.nn.Sigmoid(),
        )

    def vae(self, inp):
        enc_out = self.encoder(inp)
        mu = self.vae_mu_head(enc_out)
        sigma = self.vae_sigma_head(enc_out)
        dist = torch.distributions.Normal(mu, sigma)
        return dist


def train(cfg: DictConfig, hydra_dir: str):

    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_model = WorldModel(cfg.network_architecture, env.action_dim, env.state_dim, env.single_action_space, cfg.latent_dim).to(device)
    world_model_optim = torch.optim.Adam(world_model.parameters(), lr=cfg.lr)
    
    replay = ReplayMemory(size=cfg.replay_size, state_dim=env.state_dim, action_dim=env.action_dim, state_dtype=env.single_state_space.dtype)

    if cfg.load_path is not None:
        network_dict = torch.load(cfg.load_path, map_location=device, weights_only=False)
        world_model.load_state_dict(network_dict["world_model"])
        # checkpoints from plain ppo won't have a world model optim
        if "world_model_optim" in network_dict:
            world_model_optim.load_state_dict(network_dict["world_model_optim"])

    states, _ = env.reset()
    for timestep in range(cfg.timesteps):
        actions = env.action_space.sample() 
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)

        logger.log_episodes(info)

        replay.add(states, actions, rewards, sprimes, is_terms, is_truncs)

        states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

        # update world model
        if timestep % cfg.world_model_update_interval == 0:

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

                logger.log_stats({"world_model_loss": world_model_loss.item(), "decoder_loss": decoder_loss.item(),
                                  "predictor_loss": predictor_loss.item(), "latent_loss": latent_loss.item()})

        if cfg.save_network and timestep % cfg.save_interval == 0: logger.log_network({
            "world_model": world_model.state_dict(), "world_model_optim": world_model_optim.state_dict(),
        })

    if cfg.save_network: logger.log_network({
        "world_model": world_model.state_dict(), "world_model_optim": world_model_optim.state_dict(),
    })

