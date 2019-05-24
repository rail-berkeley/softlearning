from copy import deepcopy
from collections import defaultdict

import numpy as np

from softlearning import replay_pools
from . import (
    dummy_sampler,
    remote_sampler,
    base_sampler,
    simple_sampler)


def get_sampler_from_variant(variant, *args, **kwargs):
    SAMPLERS = {
        'DummySampler': dummy_sampler.DummySampler,
        'RemoteSampler': remote_sampler.RemoteSampler,
        'Sampler': base_sampler.BaseSampler,
        'SimpleSampler': simple_sampler.SimpleSampler,
    }

    sampler_params = variant['sampler_params']
    sampler_type = sampler_params['type']

    sampler_args = deepcopy(sampler_params.get('args', ()))
    sampler_kwargs = deepcopy(sampler_params.get('kwargs', {}))

    sampler = SAMPLERS[sampler_type](
        *sampler_args, *args, **sampler_kwargs, **kwargs)

    return sampler


DEFAULT_PIXEL_RENDER_KWARGS = {
    'mode': 'rgb_array',
    'width': 100,
    'height': 100,
}

DEFAULT_HUMAN_RENDER_KWARGS = {
    'mode': 'human',
    'width': 500,
    'height': 500,
}


def rollout(env,
            policy,
            path_length,
            callback=None,
            render_kwargs=None,
            break_on_terminal=True):
    observation_space = env.observation_space
    action_space = env.action_space

    pool = replay_pools.SimpleReplayPool(
        observation_space, action_space, max_size=path_length)
    sampler = simple_sampler.SimpleSampler(
        max_path_length=path_length,
        min_pool_size=None,
        batch_size=None)

    sampler.initialize(env, policy, pool)

    render_mode = (render_kwargs or {}).get('mode', None)
    if render_mode == 'rgb_array':
        render_kwargs = {
            **DEFAULT_PIXEL_RENDER_KWARGS,
            **render_kwargs
        }
    elif render_mode == 'human':
        render_kwargs = {
            **DEFAULT_HUMAN_RENDER_KWARGS,
            **render_kwargs
        }
    else:
        render_kwargs = None

    images = []
    infos = defaultdict(list)

    t = 0
    for t in range(path_length):
        observation, reward, terminal, info = sampler.sample()
        for key, value in info.items():
            infos[key].append(value)

        if callback is not None:
            callback(observation)

        if render_kwargs:
            image = env.render(**render_kwargs)
            images.append(image)

        if terminal:
            policy.reset()
            if break_on_terminal: break

    assert pool._size == t + 1

    path = pool.batch_by_indices(np.arange(pool._size))
    path['infos'] = infos

    if render_mode == 'rgb_array':
        path['images'] = np.stack(images, axis=0)

    return path


def rollouts(n_paths, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_paths)]
    return paths
