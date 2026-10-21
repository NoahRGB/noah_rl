import pygame
from noah_rl.algorithms.world_models.ppo_model_tests import ActorCriticNetwork, WorldModel
import torch
from noah_rl.utils.gym import GymEnv
from omegaconf import OmegaConf
import numpy as np
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

cfg.env_kwargs.render_mode = None
cfg.env_kwargs.exploring_starts = False

env = GymEnv(cfg.env_name, 1, seed=seed, wrappers=cfg.wrappers, **cfg.env_kwargs)


world_model = WorldModel({}, env.action_dim, env.state_dim, env.single_action_space).to(device)


checkpoint = torch.load(f"{run_dir}/network.pt", map_location=device, weights_only=False)

world_model.load_state_dict(checkpoint["world_model"])


states, _ = env.reset()
SCALE = 6
screen = pygame.display.set_mode((84 * SCALE, 84 * SCALE))
clock = pygame.time.Clock()

with torch.no_grad():
    z = world_model.encoder(torch.from_numpy(states).float() / cfg.state_scale)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                states, _ = env.reset()
                z = world_model.encoder(torch.from_numpy(states).float() / cfg.state_scale)
        if pygame.key.get_pressed()[pygame.K_ESCAPE]: running = False

        a = torch.tensor([keys_to_action()])
        a_one_hot = torch.nn.functional.one_hot(a, num_classes=env.single_action_space.n).float()
        z = world_model.predictor(torch.cat([z, a_one_hot], dim=-1))

        frame = (world_model.decoder(z)[0, 0].numpy() * 255).astype(np.uint8)
        rgb = np.stack([frame.T] * 3, axis=-1)
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(rgb), screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(5)
