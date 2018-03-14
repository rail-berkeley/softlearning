import numpy as np
import time


def rollout(env, policy, path_length, render=False, speedup=10, callback=None,
            render_mode='human'):

    ims = list()

    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length,))
    rewards = np.zeros((path_length,))
    all_infos = list()
    t = 0  # To make edge case path_length=0 work.
    for t in range(t, path_length):

        action, _ = policy.get_action(observation)

        if callback is not None:
            callback(observation, action)

        next_obs, reward, terminal, info = env.step(action)

        all_infos.append(info)

        actions[t, :] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t, :] = observation

        observation = next_obs

        if render:
            if render_mode == 'rgb_array':
                ims.append(env.render(
                    mode=render_mode,
                ))
            else:
                env.render(render_mode)
                time_step = 0.05
                time.sleep(time_step / speedup)

        if terminal:
            break

    last_obs = observation

    concat_infos = {}
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

    if render_mode == 'rgb_array':
        path['ims'] = np.stack(ims, axis=0)

    return path


def rollouts(env, policy, path_length, n_paths, render=False, render_mode='human'):
    paths = [
        rollout(env, policy, path_length, render, render_mode=render_mode)
        for i in range(n_paths)
    ]

    return paths
