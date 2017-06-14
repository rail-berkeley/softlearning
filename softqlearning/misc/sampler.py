import numpy as np
import time


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length,))
    rewards = np.zeros((path_length,))
    all_infos = list()
    t = 0
    for t in range(t, path_length):

        action, _ = policy.get_action(observation)
        next_obs, reward, terminal, info = env.step(action)

        all_infos.append(info)

        actions[t, :] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t, :] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    last_obs = observation

    concat_infos = dict()
    for key in all_infos[0].keys():
        all_vals = [np.array(info[key])[None] for info in all_infos]
        concat_infos[key] = np.concatenate(all_vals)

    path = dict(
        last_obs=last_obs,
        dones=terminals[:t+1],
        actions=actions[:t+1],
        observations=observations[:t+1],
        rewards=rewards[:t+1],
        env_infos=concat_infos
    )

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length))

    return paths
