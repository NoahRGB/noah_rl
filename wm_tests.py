from noah_rl.algorithms.ppo_model_tests import ActorCriticNetwork, WorldModel
import torch
from noah_rl.utils.gym import GymEnv
from omegaconf import OmegaConf
import imageio, numpy as np

run_dir = "./results/logs/ppo_wm_tests_20261020_133235/"
device = "cpu"
seed = 1

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

# cfg.env_kwargs.render_mode = "human"
cfg.env_kwargs.exploring_starts = False

env = GymEnv(cfg.env_name, 1, seed=seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
network = ActorCriticNetwork(cfg.network_architecture, env.state_dim, env.single_action_space, cfg.state_scale)


world_model = WorldModel({}, env.action_dim, env.state_dim, env.single_action_space).to(device)


checkpoint = torch.load(f"{run_dir}/network.pt", map_location=device, weights_only=False)


network.load_state_dict(checkpoint["net"])
world_model.load_state_dict(checkpoint["world_model"])


done = False
states, _ = env.reset()
saved_states = [states]
saved_actions = []
while not done:
    action_dist = network.get_actor(torch.from_numpy(states))
    actions = action_dist.sample()
    saved_actions.append(actions)
    sprimes, rewards, is_terms, is_truncs, info = env.step(actions.cpu().numpy())

    done = bool((is_terms | is_truncs)[0])
    saved_states.append(sprimes)

    states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

running_z = world_model.encoder(torch.from_numpy(saved_states[0]).float() / cfg.state_scale)

saved_decoded = []
saved_imagined = []
with torch.no_grad():
    for t in range(len(saved_states)-1):
        actions_one_hot = torch.nn.functional.one_hot(saved_actions[t].long(), num_classes=env.single_action_space.n).float()
        z_next = world_model.encoder(torch.from_numpy(saved_states[t+1]).float() / cfg.state_scale)
        saved_decoded.append(world_model.decoder(z_next))

        running_z = world_model.predictor(torch.cat([running_z, actions_one_hot], dim=-1))
        saved_imagined.append(world_model.decoder(running_z))

frames = []
for real, recon, imagined in zip(saved_states[1:], saved_decoded, saved_imagined):


    real = real[0, -1].astype(np.float32) / cfg.state_scale
    recon = recon[0, -1]
    imagined = imagined[0, -1]

    gap = np.full((84, 2), 0.5, dtype=np.float32)   # 2px grey strip, becomes 8px after the 4x upscale
    row = np.concatenate([real, gap, recon, gap, imagined], axis=1)
    row = np.kron(row, np.ones((4, 4)))                            # upscale 4x so it's not tiny
    frames.append((row * 255).astype(np.uint8))

imageio.mimsave(f"{run_dir}/dream.gif", frames, duration=1/30, loop=0)   # gif
imageio.mimsave(f"{run_dir}/dream.mp4", frames, fps=30)                  # mp4
