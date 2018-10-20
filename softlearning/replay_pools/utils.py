from .simple_replay_pool import SimpleReplayPool
from .image_replay_pool import ImageReplayPool
from .extra_policy_info_replay_pool import ExtraPolicyInfoReplayPool
from .union_pool import UnionPool
# from .distance_pool import DistanceReplayPool


POOL_CLASSES = {
    # 'DistanceReplayPool': DistanceReplayPool,
    'ExtraPolicyInfoReplayPool': ExtraPolicyInfoReplayPool,
    'SimpleReplayPool': SimpleReplayPool,
    'UnionPool': UnionPool,
    'ImageReplayPool': ImageReplayPool,
}

DEFAULT_REPLAY_POOL = 'SimpleReplayPool'


def get_replay_pool_from_variant(variant, env):
    replay_pool_params = variant['replay_pool_params'].copy()

    pool_type = replay_pool_params.pop('type', DEFAULT_REPLAY_POOL)

    pool = POOL_CLASSES[pool_type](
        observation_space=env.observation_space,
        action_space=env.action_space,
        **replay_pool_params)

    return pool
