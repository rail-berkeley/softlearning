from collections import defaultdict

import numpy as np

from softlearning import replay_pools
from . import simple_sampler


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


def rollout(environment,
            policy,
            path_length,
            replay_pool_class=replay_pools.SimpleReplayPool,
            sampler_class=simple_sampler.SimpleSampler,
            render_kwargs=None,
            break_on_terminal=True):
    pool = replay_pool_class(environment, max_size=path_length)
    sampler = sampler_class(
        environment=environment,
        policy=policy,
        pool=pool,
        max_path_length=path_length)

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

        if render_kwargs:
            image = environment.render(**render_kwargs)
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
