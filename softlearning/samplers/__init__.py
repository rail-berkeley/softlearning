import time

import numpy as np

from .sampler import Sampler
from .dummy_sampler import DummySampler
from .simple_sampler import SimpleSampler
from .extra_policy_info_sampler import ExtraPolicyInfoSampler


def rollout(env, policy, path_length, render=False, speedup=10, callback=None,
            render_mode='human'):

    ims = []

    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    log_pis = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0  # To make edge case path_length=0 work.
    for t in range(path_length):
        action, log_pis, agent_info = policy.get_action(observation, with_log_pis=True)

        if callback is not None:
            callback(observation, action)

        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        log_pis[t] = reward
        rewards[t] = reward
        observations[t] = observation

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

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'log_pis': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }


    if render_mode == 'rgb_array':
        path['ims'] = np.stack(ims, axis=0)

    return path


def rollouts(env, policy, path_length, n_paths, render=False,
             render_mode='human'):
    paths = [
        rollout(env, policy, path_length, render, render_mode=render_mode)
        for i in range(n_paths)
    ]

    return paths
