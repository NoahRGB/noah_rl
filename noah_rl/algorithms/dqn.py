# NoahRGB 09/2026
# DQN implementaion (https://www.nature.com/articles/nature14236) 
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/dqn.yaml)
#

from collections import deque
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import gymnasium as gym

from noah_rl.utils.utils import seed
from noah_rl.utils.gym import GymEnv
from noah_rl.utils.networks import Encoder, detect_head
from noah_rl.utils.logging import Logger
from noah_rl.utils.schedulers import detect_scheduler

class ReplayMemory:
    # stores s, a, r, s', is_terminated, is_truncated
    # for X transitions
    # 
    # will overwrite old transitions with new ones when max capacity
    # is reached

    def __init__(self, size: int, state_dim: tuple, action_dim: tuple):
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.current_size = 0
        self.pointer = 0
        self.reset()

    def reset(self):
        self.states_buffer = np.empty((self.max_size, *self.state_dim), dtype=np.float32)
        self.actions_buffer = np.empty((self.max_size, *self.action_dim), dtype=np.float32)
        self.rewards_buffer = np.empty((self.max_size), dtype=np.float32)
        self.next_states_buffer = np.empty((self.max_size, *self.state_dim), dtype=np.float32)
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

    def sample(self, size):
        sample_size = min(size, self.current_size)
        indices = np.random.choice(self.current_size, size=sample_size, replace=False)
        return (torch.from_numpy(self.states_buffer[indices]),
                torch.from_numpy(self.actions_buffer[indices]),
                torch.from_numpy(self.rewards_buffer[indices]),
                torch.from_numpy(self.next_states_buffer[indices]),
                torch.from_numpy(self.is_terms_buffer[indices]),
                torch.from_numpy(self.is_truncs_buffer[indices])), sample_size


class QNet(torch.nn.Module):
    # simple DQN network with a Q(s,a) output for each discret action 

    def __init__(self, architecture: dict, state_dim: tuple, action_count: int):
        super(QNet, self).__init__()

        self.encoder = Encoder(architecture["qnet_enc"], input_shape=state_dim)
        self.enc_output = self.encoder.encoder_output

        self.qvals_out = torch.nn.Linear(self.enc_output, action_count)

    def forward(self, inp):
        enc_out = self.encoder(inp)
        return self.qvals_out(enc_out)

def train(cfg: DictConfig, hydra_dir: str):

    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    assert type(env.single_action_space) == gym.spaces.Discrete, "DQN only works with discrete aciton spaces"

    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)
    network = QNet(cfg.network_architecture, env.state_dim, env.single_action_space.n)
    target_network = QNet(cfg.network_architecture, env.state_dim, env.single_action_space.n)
    optim = torch.optim.Adam(network.parameters(), lr=cfg.lr)
    replay = ReplayMemory(size=cfg.replay_size, state_dim=env.state_dim, action_dim=env.action_dim) 

    epsilon_scheduler = detect_scheduler(cfg.epsilon_scheduler)
    epsilon = epsilon_scheduler.get_value()

    states, _ = env.reset()

    for warmup_step in range(cfg.warmup_steps):
        actions = env.action_space.sample()
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)
        replay.add(states, actions, rewards, sprimes, is_terms, is_truncs)
        states = env.reset_finished_envs(sprimes, is_terms, is_truncs)

    num_rollouts = cfg.timesteps // cfg.update_freq
    for rollout in range(num_rollouts):
        
        for timestep in range(cfg.update_freq):
            
            # epsilon-greedy action selection
            if np.random.random() >= epsilon:
                qvals = network(torch.from_numpy(states))
                actions = qvals.argmax(dim=-1).cpu().numpy()
            else:
                actions = env.action_space.sample()

            sprimes, rewards, is_terms, is_truncs, info = env.step(actions)

            logger.log_episodes(info)

            replay.add(states, actions, rewards, sprimes, is_terms, is_truncs)

            states = env.reset_finished_envs(sprimes, is_terms, is_truncs)
            epsilon = epsilon_scheduler.step() 

            # every C timesteps, clone network Q to get ^Q (target)
            if logger.timesteps_completed % cfg.target_update_interval == 0:
                target_network.load_state_dict(network.state_dict())
        
        # rollout finished, update time
        stats =  {"qnet_loss": []}
        for grad_update in range(cfg.gradient_steps):

            (batch_states, batch_actions, batch_rewards, batch_next_states, batch_is_terms, batch_is_truncs), sample_size = replay.sample(cfg.minibatch_size)
            masks = 1 - (batch_is_terms)
            if sample_size == cfg.minibatch_size:

                # the Q(s,a) values for each action 
                qvals = network(batch_states) # (batch_size, num_actions)

                # the Q(s,a) values that were actually chosen
                chosen_qvals = qvals.gather(-1, batch_actions.unsqueeze(-1).long()).squeeze(-1)

                with torch.no_grad():
                    next_qvals = target_network(batch_next_states)
                    # y = r + γ * max ^Q(s_t+1)
                    targets = batch_rewards + cfg.gamma * next_qvals.max(-1)[0] * masks

                # L = (y - Q(s,a))^2
                loss = torch.nn.functional.mse_loss(chosen_qvals, targets)
                optim.zero_grad()
                loss.backward()
                if cfg.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.cgn)
                optim.step()

                stats["qnet_loss"].append(loss.item())

        logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
        if cfg.save_network: logger.log_network({"net": network.state_dict(), "target_net": target_network.state_dict(), "optim": optim.state_dict()})

        

@hydra.main(config_path="../../params", config_name="dqn", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train(cfg, run_dir)

if __name__ == "__main__":
    main()

