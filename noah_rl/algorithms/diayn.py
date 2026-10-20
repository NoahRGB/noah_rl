# NoahRGB 10/2026
# DIAYN implementaion with SAC (https://arxiv.org/abs/1802.06070) 
#
# UNFINISHED
#
# Hydra is used for specifying hyperparams
# (see noah_rl/params/dqn.yaml)
#

from collections import deque
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
        self.skills_buffer = np.empty((self.max_size), dtype=np.float32)
        self.next_states_buffer = np.empty((self.max_size, *self.state_dim), dtype=self.state_dtype)
        self.is_terms_buffer = np.empty((self.max_size), dtype=np.float32)
        self.is_truncs_buffer = np.empty((self.max_size), dtype=np.float32)

    def add(self, states, actions, rewards, skill, next_states, is_terms, is_truncs):
        batch_dim = states.shape[0]
        for transition in range(batch_dim):
            self.states_buffer[self.pointer] = states[transition]
            self.actions_buffer[self.pointer] = actions[transition]
            self.rewards_buffer[self.pointer] = rewards[transition]
            self.skills_buffer[self.pointer] = skill
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
                torch.from_numpy(self.skills_buffer[indices]),
                torch.from_numpy(self.next_states_buffer[indices]),
                torch.from_numpy(self.is_terms_buffer[indices]),
                torch.from_numpy(self.is_truncs_buffer[indices])), sample_size


class Actor(torch.nn.Module):

    def __init__(self, architecture: dict, state_dim: tuple, action_space: gym.Space, num_skills: int):
        super(Actor, self).__init__()

        self.encoder = Encoder(architecture["actor_enc"], input_shape=(state_dim[0]+num_skills,))
        self.enc_output = self.encoder.encoder_output

        self.actor_head = detect_head(action_space=action_space, input_size=self.enc_output)
    
    def forward(self, inp):
        enc_out = self.encoder(inp)
        return self.actor_head(enc_out)


class QNet(torch.nn.Module):

    def __init__(self, architecture: dict, state_dim: tuple, action_dim: tuple, num_skills: int):
        super(QNet, self).__init__()

        self.encoder = Encoder(architecture["qnet_enc"], input_shape=(state_dim[0]+num_skills+action_dim[0],))
        self.enc_output = self.encoder.encoder_output

        self.qvals_out = torch.nn.Linear(self.enc_output, 1)

    def forward(self, states, actions):
        inp = torch.concat([states, actions], dim=-1)
        enc_out = self.encoder(inp)
        return self.qvals_out(enc_out)

class Discriminator(torch.nn.Module):

    def __init__(self, architecture: dict, state_dim: tuple, num_skills: int):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(architecture["discriminator_enc"], input_shape=state_dim)
        self.enc_output = self.encoder.encoder_output

        self.head = torch.nn.Linear(self.enc_output, num_skills)

    def forward(self, inp):
        enc_out = self.encoder(inp)
        return self.head(enc_out)

def prep(tens: torch.Tensor):
    if tens.dtype == torch.uint8:
        return tens.float() / 255.0
    return tens.float()

def train(cfg: DictConfig, hydra_dir: str):

    seed(cfg.seed)
    env = GymEnv(cfg.env_name, cfg.num_envs, seed=cfg.seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
    assert type(env.single_action_space) == gym.spaces.Box, "SAC only works with continuous aciton spaces"
    assert env.num_envs == 1, "This DIAYN only works with 1 env running"

    logger = Logger(cfg.num_envs, cfg.title, hydra_dir, tensorboard=True)

    if "shared_enc" in cfg.network_architecture:
        shared_enc = Encoder(cfg.network_architecture["shared_enc"], input_shape=env.state_dim)
        target_shared_enc = Encoder(cfg.network_architecture["shared_enc"], input_shape=env.state_dim)
        networks_input = (shared_enc.encoder_output,)
    else:
        shared_enc = torch.nn.Identity()
        target_shared_enc = torch.nn.Identity()
        networks_input = env.state_dim

    target_shared_enc.load_state_dict(shared_enc.state_dict())


    # 1 actor, 2 Q networks (with targets)
    actor = Actor(cfg.network_architecture, networks_input, env.single_action_space, cfg.num_skills)
    qfunc1 = QNet(cfg.network_architecture, networks_input, env.action_dim, cfg.num_skills)
    qfunc2 = QNet(cfg.network_architecture, networks_input, env.action_dim, cfg.num_skills)
    target_qfunc1 = QNet(cfg.network_architecture, networks_input, env.action_dim, cfg.num_skills)
    target_qfunc2 = QNet(cfg.network_architecture, networks_input, env.action_dim, cfg.num_skills)

    target_qfunc1.load_state_dict(qfunc1.state_dict())
    target_qfunc2.load_state_dict(qfunc2.state_dict())

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optim = torch.optim.Adam([*qfunc1.parameters(), *qfunc2.parameters(), *shared_enc.parameters()], lr=cfg.lr)

    # a categorical distribution over X skills
    skills_distribution = torch.distributions.Categorical(probs=torch.ones(cfg.num_skills) / cfg.num_skills)
    discriminator = Discriminator(cfg.network_architecture, env.state_dim, cfg.num_skills)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr)

    if cfg.load_path is not None:
        d = torch.load(cfg.load_path, weights_only=False)
        shared_enc.load_state_dict(d["shared_enc"])
        actor.load_state_dict(d["actor"]); actor_optim.load_state_dict(d["actor_optim"])
        qfunc1.load_state_dict(d["qfunc1"]); qfunc2.load_state_dict(d["qfunc2"]);
        critic_optim.load_state_dict(d["critic_optim"])
        target_qfunc1.load_state_dict(qfunc1.state_dict()); target_qfunc2.load_state_dict(qfunc2.state_dict())
        target_shared_enc.load_state_dict(shared_enc.state_dict())
        discriminator.load_state_dict(d["discriminator"]); discriminator_optim.load_state_dict(d["discriminator_optim"])

    replay = ReplayMemory(size=cfg.replay_size, state_dim=env.state_dim, action_dim=env.action_dim, state_dtype=env.single_state_space.dtype)

    if not cfg.do_eval:
        states, _ = env.reset()
        skill = skills_distribution.sample()

        for warmup_step in range(cfg.warmup_steps):

            with torch.no_grad():
                skill_one_hot = torch.nn.functional.one_hot(skill, cfg.num_skills).unsqueeze(0).float()
                action_dist = actor(torch.concat([shared_enc(prep(torch.from_numpy(states))), skill_one_hot], dim=-1))
                u = action_dist.rsample()
                actions = torch.tanh(u)

            sprimes, rewards, is_terms, is_truncs, info = env.step(actions.cpu().numpy())

            with torch.no_grad():
                discriminator_logits = discriminator(prep(torch.from_numpy(sprimes)))
                discriminator_logq = torch.log_softmax(discriminator_logits, dim=-1).gather(-1, skill.view(1, 1)).squeeze(-1)
                diayn_rewards = discriminator_logq - skills_distribution.log_prob(skill)

            replay.add(states, actions, diayn_rewards.cpu().numpy(), skill, sprimes, is_terms, is_truncs)

            # if the episode ended, a new skill needs to be chosen
            if "episode" in info:
                skill = skills_distribution.sample()
            states = env.reset_finished_envs(sprimes, is_terms, is_truncs)


        num_rollouts = cfg.timesteps // cfg.update_freq
        for rollout in range(num_rollouts):
            
            for timestep in range(cfg.update_freq):

                with torch.no_grad():
                    skill_one_hot = torch.nn.functional.one_hot(skill, cfg.num_skills).unsqueeze(0).float()
                    action_dist = actor(torch.concat([shared_enc(prep(torch.from_numpy(states))), skill_one_hot], dim=-1))
                    u = action_dist.rsample()
                    actions = torch.tanh(u)

                sprimes, rewards, is_terms, is_truncs, info = env.step(actions.cpu().numpy())

                with torch.no_grad():
                    discriminator_logits = discriminator(prep(torch.from_numpy(sprimes)))
                    discriminator_logq = torch.log_softmax(discriminator_logits, dim=-1).gather(-1, skill.view(1, 1)).squeeze(-1)
                    diayn_rewards = discriminator_logq - skills_distribution.log_prob(skill)

                logger.log_episodes(info)

                replay.add(states, actions, diayn_rewards.cpu().numpy(), skill, sprimes, is_terms, is_truncs)

                # if the episode ended, a new skill needs to be chosen
                if "episode" in info:
                    skill = skills_distribution.sample()

                states = env.reset_finished_envs(sprimes, is_terms, is_truncs)


            # rollout finished, update time
            stats =  {"policy_loss": [], "critic_loss": [], "discriminator_loss": [], "mean_reward": []}
            for grad_update in range(cfg.gradient_steps):

                (batch_states, batch_actions, batch_rewards, batch_skills, batch_next_states, batch_is_terms, batch_is_truncs), sample_size = replay.sample(cfg.minibatch_size)
                masks = 1 - (batch_is_terms)
                skills_one_hot = torch.nn.functional.one_hot(batch_skills.long(), cfg.num_skills).float()

                if sample_size == cfg.minibatch_size:
                    batch_states = prep(batch_states); batch_next_states = prep(batch_next_states)

                    with torch.no_grad():
                        discriminator_logits = discriminator(batch_next_states)
                        discriminator_logq = torch.log_softmax(discriminator_logits, dim=-1).gather(-1, batch_skills.long().unsqueeze(-1)).squeeze(-1)
                        batch_rewards = discriminator_logq - skills_distribution.log_prob(batch_skills.long())

                    # UPDATE Q FUNCTIONS 

                    # input to qfuncs is state+action
                    shared_features = shared_enc(batch_states)
                    qfunc1_vals = qfunc1(torch.concat([shared_features, skills_one_hot], dim=-1), batch_actions).view(sample_size)
                    qfunc2_vals = qfunc2(torch.concat([shared_features, skills_one_hot], dim=-1), batch_actions).view(sample_size)

                    with torch.no_grad():

                        # comptute fresh/on-policy actions + logprobs using s_{t+1}
                        next_dist = actor(torch.concat([shared_enc(batch_next_states), skills_one_hot], dim=-1))
                        next_u = next_dist.rsample()
                        next_actions = torch.tanh(next_u) # this is a_{t+1}
                        # log probs from Appendix C of the SAC paper
                        next_log_probs = next_dist.log_prob(next_u) - torch.log(1 - next_actions.pow(2) + 1e-6).sum(-1)

                        # SAC paper Section 6 "we use the minimum of the the soft Q-functions"
                        next_shared_features = target_shared_enc(batch_next_states)
                        qfunc1_next_vals = target_qfunc1(torch.concat([next_shared_features, skills_one_hot], dim=-1), next_actions).view(sample_size)
                        qfunc2_next_vals = target_qfunc2(torch.concat([next_shared_features, skills_one_hot], dim=-1), next_actions).view(sample_size)
                        min_next_qvals = torch.min(qfunc1_next_vals, qfunc2_next_vals)

                        # y = r + γ * ( Q(s_{t+1}, a_{t+1}) - alpha * log( π(a_{t+1}|s_{t+1}) ) )
                        qfunc_targets = batch_rewards + cfg.gamma * (min_next_qvals - cfg.alpha * next_log_probs) * masks

                    # backprop + SGD on qfuncs
                    critic_loss = torch.nn.functional.mse_loss(qfunc1_vals, qfunc_targets) + torch.nn.functional.mse_loss(qfunc2_vals, qfunc_targets)
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_optim.step()

                    # UPDATE ACTOR

                    # compute fresh/on-policy actions + logprobs using s_t
                    shared_features = shared_features.detach()
                    states_dist = actor(torch.concat([shared_features, skills_one_hot], dim=-1))
                    states_u = states_dist.rsample()
                    states_actions = torch.tanh(states_u) # this is a_t
                    # log probs from Appendix C of the SAC paper
                    states_log_probs = states_dist.log_prob(states_u) - torch.log(1 - states_actions.pow(2) + 1e-6).sum(-1)

                    # SAC paper section 6 "we use the minimum of the the soft Q-functions"
                    qfunc1_actor = qfunc1(torch.concat([shared_features, skills_one_hot], dim=-1), states_actions).view(sample_size)
                    qfunc2_actor = qfunc2(torch.concat([shared_features, skills_one_hot], dim=-1), states_actions).view(sample_size)
                    min_actor_qvals = torch.min(qfunc1_actor, qfunc2_actor)

                    # alpha * log( π(a_t|s_t) ) - Q(s_t a_t)
                    policy_loss = -(min_actor_qvals - cfg.alpha * states_log_probs).mean()

                    actor_optim.zero_grad()
                    policy_loss.backward()
                    actor_optim.step()

                    # UPDATE TARGET NETWORKS 
                    # uses polyak / target smoothing coefficient (SAC paper section 4.2)
                    with torch.no_grad():
                        for param, target_param in zip(qfunc1.parameters(), target_qfunc1.parameters()):
                            target_param.mul_(1 - cfg.tau).add_(cfg.tau * param)

                        for param, target_param in zip(qfunc2.parameters(), target_qfunc2.parameters()):
                            target_param.mul_(1 - cfg.tau).add_(cfg.tau * param)

                        for param, target_param in zip(shared_enc.parameters(), target_shared_enc.parameters()):
                            target_param.mul_(1 - cfg.tau).add_(cfg.tau * param)


                    discriminator_logits = discriminator(batch_next_states)
                    discriminator_loss = torch.nn.functional.cross_entropy(discriminator_logits, batch_skills.long())
                    discriminator_optim.zero_grad()
                    discriminator_loss.backward()
                    discriminator_optim.step()

                    stats["policy_loss"].append(policy_loss.item())
                    stats["critic_loss"].append(critic_loss.item()); stats["discriminator_loss"].append(discriminator_loss.item())
                    stats["mean_reward"].append(batch_rewards.mean())

            logger.log_stats({name: np.mean(loss) for name, loss in stats.items()})
            if cfg.save_network: logger.log_network({
                "shared_enc": shared_enc.state_dict(),
                "actor": actor.state_dict(), "actor_optim": actor_optim.state_dict(),
                "qfunc1": qfunc1.state_dict(), "qfunc2": qfunc2.state_dict(),
                "critic_optim": critic_optim.state_dict(),
                "discriminator": discriminator.state_dict(), "discriminator_optim": discriminator_optim.state_dict(),
            })
    
    else:

        # in eval mode, iterate through skills and 
        # act using them for 1 episode each

        # skills_to_watch = [i for i in range(cfg.num_skills)]
        skills_to_watch = [0, 10, 13, 17]

        for skill in skills_to_watch:
            print(f"acting skill {skill}")
            states, _ = env.reset()
            done = False
            
            while not done:

                with torch.no_grad(): 
                    skill_one_hot = torch.nn.functional.one_hot(torch.tensor(skill), cfg.num_skills).unsqueeze(0).float()
                    action_dist = actor(torch.concat([shared_enc(prep(torch.from_numpy(states))), skill_one_hot], dim=-1))
                    u = action_dist.rsample()
                    actions = torch.tanh(u)

                sprimes, rewards, is_terms, is_truncs, info = env.step(actions.cpu().numpy())

                # if the episode ended, a new skill needs to be chosen
                if "episode" in info:
                    done = True

                states = env.reset_finished_envs(sprimes, is_terms, is_truncs)
