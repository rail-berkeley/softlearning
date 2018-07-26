from rllab.envs.mujoco.swimmer_env import SwimmerEnv as RllabSwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv as RllabAntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv as RllabHumanoidEnv
from rllab.envs.normalized_env import normalize

from .gym_env import GymEnv
from . import rllab as custom_rllab_envs


GYM_ENVIRONMENTS = {
    'swimmer': {
        'default': lambda: GymEnv('Swimmer-v1')
    },
    'ant': {
        'default': lambda: GymEnv('Ant-v1')
    },
    'humanoid': {
        'default': lambda: GymEnv('Humanoid-v1'),
        'standup': lambda: GymEnv('HumanoidStandup-v1')
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v1')
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1')
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v1')
    },
}


RLLAB_ENVIRONMENTS = {
    'swimmer': {
        'default': RllabSwimmerEnv,
        'multi-direction': custom_rllab_envs.MultiDirectionSwimmerEnv,
    },
    'ant': {
        'default': RllabAntEnv,
        'multi-direction': custom_rllab_envs.MultiDirectionAntEnv,
        'cross-maze': custom_rllab_envs.CrossMazeAntEnv
    },
    'humanoid': {
        'default': RllabHumanoidEnv,
        'multi-direction': custom_rllab_envs.MultiDirectionHumanoidEnv,
    },
}


ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
    'rllab': RLLAB_ENVIRONMENTS
}


def get_environment(universe, domain, task, env_params):
    env = ENVIRONMENTS[universe][domain][task](**env_params)
    return normalize(env)
