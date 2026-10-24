# from noah_rl.algorithms.world_models.ppo_model_tests import ActorCriticNetwork, WorldModel
from noah_rl.algorithms.world_models.random_world_model import WorldModel
import torch
from noah_rl.utils.gym import GymEnv
from omegaconf import OmegaConf
import imageio, numpy as np
import pygame
import sys

def keys_to_action():
    pygame.event.pump()
    k = pygame.key.get_pressed()
    if k[pygame.K_RSHIFT] and k[pygame.K_SPACE]: return 5
    if k[pygame.K_RSHIFT]: return 2
    if k[pygame.K_SPACE]:  return 3
    if k[pygame.K_a]:      return 0
    if k[pygame.K_s]:      return 1
    return 4

run_dir = sys.argv[1]
device = "cpu"
seed = 1

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

cfg.env_kwargs.render_mode = "human"
# cfg.env_kwargs.exploring_starts = False

env = GymEnv(cfg.env_name, 1, seed=seed, wrappers=cfg.wrappers, **cfg.env_kwargs)

world_model = WorldModel({}, env.action_dim, env.state_dim, env.single_action_space, cfg.latent_dim, cfg.hidden_dim).to(device)

checkpoint = torch.load(f"{run_dir}/network.pt", map_location=device, weights_only=False)

world_model.load_state_dict(checkpoint["world_model"])


done = False
states, _ = env.reset()
saved_states = [states[0]]
saved_actions = []
saved_raw = [] # full res frames from before the resize/channel_first wrappers
while not done:
    actions = env.action_space.sample()
    # actions = torch.tensor([keys_to_action()])

    saved_actions.append(actions[0])
    sprimes, rewards, is_terms, is_truncs, info = env.step(actions)

    done = bool((is_terms | is_truncs)[0])
    saved_states.append(sprimes[0])
    saved_raw.append(env.env.envs[0].unwrapped._get_obs().copy())

    states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

with torch.no_grad():
    running_hidden_state = torch.zeros(1, cfg.hidden_dim).to(device)
    latent = torch.zeros(1, cfg.latent_dim).to(device)
    actions_one_hot = torch.nn.functional.one_hot(torch.tensor(saved_actions).long(), num_classes=env.single_action_space.n).float()
    actions_shifted = torch.cat([torch.zeros_like(actions_one_hot[:1]), actions_one_hot], dim=0)
    encoded = world_model.encoder(torch.tensor(np.array(saved_states)).float() / cfg.state_scale)

    # imagination sees the first 'context_window' frames and then samples the
    # rest from the world model  
    context_frames = 5
    imagined_hidden_state = torch.zeros(1, cfg.hidden_dim).to(device)
    imagined_latent = torch.zeros(1, cfg.latent_dim).to(device)

    saved_decoded = []
    saved_imagined = []
    for t in range(len(saved_states)):
        running_hidden_state = world_model.rnn(torch.cat([latent, actions_shifted[t:t+1]], dim=-1), running_hidden_state)
        prior = world_model.prior(running_hidden_state)
        posterior = world_model.posterior(torch.cat([running_hidden_state, encoded[t].unsqueeze(0)], dim=-1))
        latent = posterior.sample()

        saved_decoded.append(world_model.decoder(torch.cat([latent, running_hidden_state], dim=-1)))

        # if context has been used up, start imagining from the prior
        imagined_hidden_state = world_model.rnn(torch.cat([imagined_latent, actions_shifted[t:t+1]], dim=-1), imagined_hidden_state)
        if t < context_frames:
            imagined_latent = world_model.posterior(torch.cat([imagined_hidden_state, encoded[t].unsqueeze(0)], dim=-1)).sample()
        else:
            imagined_latent = world_model.prior(imagined_hidden_state).sample()

        saved_imagined.append(world_model.decoder(torch.cat([imagined_latent, imagined_hidden_state], dim=-1)))


# create a gif and mp4 using the collected frames
frames = []
for raw, real, recon, imagined in zip(saved_raw, saved_states[1:], saved_decoded[1:], saved_imagined[1:]):


    real = real.transpose(1, 2, 0).astype(np.float32) / cfg.state_scale
    recon = recon[0].permute(1, 2, 0).numpy()
    imagined = imagined[0].permute(1, 2, 0).numpy()

    gap = np.full((64, 2, 3), 0.5, dtype=np.float32)
    row = np.concatenate([real, gap, recon, gap, imagined], axis=1)
    row = np.kron(row, np.ones((4, 4, 1)))

    raw = raw.astype(np.float32) / cfg.state_scale
    pad_h = max(row.shape[0] - raw.shape[0], 0)
    raw = np.pad(raw, ((pad_h // 2, pad_h - pad_h // 2), (0, 0), (0, 0)), constant_values=0.5)
    if raw.shape[0] > row.shape[0]:
        row = np.pad(row, ((0, raw.shape[0] - row.shape[0]), (0, 0), (0, 0)), constant_values=0.5)
    gap_w = 16 + (-(raw.shape[1] + 16 + row.shape[1]) % 16)
    row = np.concatenate([raw, np.full((row.shape[0], gap_w, 3), 0.5), row], axis=1)
    frames.append((row * 255).astype(np.uint8))

imageio.mimsave(f"{run_dir}/dream.gif", frames, duration=1/30, loop=0)
imageio.mimsave(f"{run_dir}/dream.mp4", frames, fps=30)
