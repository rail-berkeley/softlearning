import sys

import numpy as np

from softlearning.replay_pools import SimpleReplayPool
from .simple_sampler import SimpleSampler


def get_sampler_from_variant(variant):
    sampler_params = variant['sampler_params']
    sampler_type = sampler_params['type']
    # Use getattr instead of import here since otherwise we would
    # have a circular import between remote_sampler and utils.
    SamplerClass = getattr(sys.modules[__package__], sampler_type)
    sampler_args = sampler_params.get('args', ())
    sampler_kwargs = sampler_params.get('kwargs', {})
    sampler = SamplerClass(*sampler_args, **sampler_kwargs)

    return sampler


def rollout(env,
            policy,
            path_length,
            render=False,
            callback=None,
            render_mode='human',
            break_on_terminal=True):
    observation_space = env.observation_space
    action_space = env.action_space

    pool = SimpleReplayPool(
        observation_space, action_space, max_size=path_length)
    sampler = SimpleSampler(
        max_path_length=path_length,
        min_pool_size=None,
        batch_size=None)

    sampler.initialize(env, policy, pool)

    images = []
    env_infos = []

    t = 0
    for t in range(path_length):
        observation, reward, terminal, info = sampler.sample()
        env_infos.append(info)

        if callback is not None:
            callback(observation)

        if render:
            if render_mode == 'rgb_array':
                image = env.render(mode=render_mode)
                images.append(image)
            else:
                env.render()

        if terminal:
            policy.reset()
            if break_on_terminal: break

    assert pool._size == t + 1

    path = pool.batch_by_indices(np.arange(pool._size))
    path['env_infos'] = env_infos

    if render_mode == 'rgb_array':
        path['images'] = np.stack(images, axis=0)

    return path


def rollouts(env, policy, path_length, n_paths, render=False,
             render_mode='human'):
    paths = [
        rollout(env, policy, path_length, render, render_mode=render_mode)
        for i in range(n_paths)
    ]

    return paths
