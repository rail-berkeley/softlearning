from .distance_pool import DistanceReplayPool


POOL_CLASSES = {
    'DistanceReplayPool': DistanceReplayPool
}


def get_replay_pool_from_variant(variant, env):
    replay_pool_params = variant['replay_pool_params']
    sampler_params = variant['sampler_params']

    pool_type = replay_pool_params.get('type', 'DistanceReplayPool')

    pool = POOL_CLASSES[pool_type](
        observation_space=env.observation_space,
        action_space=env.action_space,
        path_length=sampler_params['kwargs']['max_path_length'],
        **replay_pool_params)

    return pool
