from . import (
    simple_replay_pool,
    union_pool,
    goal_replay_pool,
    hindsight_experience_replay_pool)


POOL_CLASSES = {
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    'GoalReplayPool': goal_replay_pool.GoalReplayPool,
    'UnionPool': union_pool.UnionPool,
    'HindsightExperienceReplayPool': (
        hindsight_experience_replay_pool.HindsightExperienceReplayPool),
}

DEFAULT_REPLAY_POOL = 'SimpleReplayPool'


def get_replay_pool_from_params(replay_pool_params, env, *args, **kwargs):
    replay_pool_type = replay_pool_params['type']
    replay_pool_kwargs = replay_pool_params['kwargs'].copy()

    replay_pool = POOL_CLASSES[replay_pool_type](
        *args,
        environment=env,
        **replay_pool_kwargs,
        **kwargs)

    return replay_pool


def get_replay_pool_from_variant(variant, *args, **kwargs):
    replay_pool_params = variant['replay_pool_params']
    replay_pool = get_replay_pool_from_params(
        replay_pool_params, *args, **kwargs)

    return replay_pool
