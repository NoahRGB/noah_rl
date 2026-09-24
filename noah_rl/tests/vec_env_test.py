#
# contains a simple test for verifying the way that
# Gymnasium's SyncVectorEnv resets each sub env
#
# when a sub env terminates at time t, the next timestep
# is a useless transition, where s is the final termination
# state, the reward is 0.0 and s' is the fresh reset() state
#
# issues can arise if the resets are not handled
# properly, since bootstrapping can silently fail
#

import gymnasium as gym

def vec_env_test():

    # basic params for the test
    ENV_NAME = "CartPole-v1"
    NUM_ENVS = 5
    TIMESTEPS = 100
    SEED = 1

    # make the SyncVectorEnv with NUM_ENVS sub envs
    def make_one_env():
        env = gym.make(ENV_NAME)
        return env
    envs = [make_one_env for i in range(NUM_ENVS)]
    env = gym.vector.SyncVectorEnv(envs)
    env = gym.wrappers.vector.RecordEpisodeStatistics(env)

    # state_tracker will record s, a, s', term, trunc per timestep per sub env
    state_tracker = {env: {t: {} for t in range(TIMESTEPS)} for env in range(NUM_ENVS)}

    # run TIMESTEPS and store transitions in state_tracker
    states, info = env.reset(seed=SEED)
    for timestep in range(TIMESTEPS):
        actions = env.action_space.sample()
        next_states, rewards, is_terms, is_truncs, info = env.step(actions)

        for sub_env in range(NUM_ENVS):
            state_tracker[sub_env][timestep] = {
                "s": states[sub_env],
                "a": actions[sub_env],
                "sprime": next_states[sub_env],
                "reward": rewards[sub_env],
                "terminated": is_terms[sub_env],
                "truncated": is_truncs[sub_env]
            }

        states = next_states


    for env_idx, timesteps in state_tracker.items():
        for timestep, transition in timesteps.items():
            if transition["terminated"]:
                if timestep != TIMESTEPS-1 and timestep != 0:

                    print(f"\nenv {env_idx} terminated at timestep {timestep}")

                    print(f"\nso s at timestep {timestep+1} is a duplicate of s' at timestep {timestep}:")
                    print(f"{list(timesteps[timestep+1]['s'])} == {list(transition['sprime'])} ")
                    assert (timesteps[timestep+1]['s'] == transition['sprime']).all()

                    print(f"\nand the reward at timestep {timestep+1} is 0 (meaningless)")
                    print(f"{timestep+1} reward: {timesteps[timestep+1]['reward']}")
                    assert timesteps[timestep+1]['reward'] == 0

                    print(f"\ntherefore an extra env.reset() is executed so that s at timestep {timestep+1} is the true fresh reset state")

                    # full debug print

                    # print(f"env {env_idx} timestep {timestep-1}:")
                    # print(f"\ts: {timesteps[timestep-1]['s']}")
                    # print(f"\ta: {timesteps[timestep-1]['a']}")
                    # print(f"\ts': {timesteps[timestep-1]['sprime']}")
                    # print(f"\treward: {timesteps[timestep-1]['reward']}")
                    # print(f"\tterminated: {timesteps[timestep-1]['terminated']}")
                    # print(f"\ttruncated: {timesteps[timestep-1]['truncated']}")
                    #
                    # print(f"\nenv {env_idx} timestep {timestep}:")
                    # print(f"\ts: {timesteps[timestep]['s']}")
                    # print(f"\ta: {timesteps[timestep]['a']}")
                    # print(f"\ts': {timesteps[timestep]['sprime']}")
                    # print(f"\treward: {timesteps[timestep]['reward']}")
                    # print(f"\tterminated: {timesteps[timestep]['terminated']}")
                    # print(f"\ttruncated: {timesteps[timestep]['truncated']}")
                    #
                    # print(f"\nenv {env_idx} timestep {timestep+1}:")
                    # print(f"\ts: {timesteps[timestep+1]['s']}")
                    # print(f"\ta: {timesteps[timestep+1]['a']}")
                    # print(f"\ts': {timesteps[timestep+1]['sprime']}")
                    # print(f"\treward: {timesteps[timestep+1]['reward']}")
                    # print(f"\tterminated: {timesteps[timestep+1]['terminated']}")
                    # print(f"\ttruncated: {timesteps[timestep+1]['truncated']}")
                    
                    return

if __name__ == "__main__":
    vec_env_test()
