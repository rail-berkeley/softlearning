"""Custom Gym environments.

Every class inside this module should extend a gym.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under gym.mujoco submodule.
"""

import gym


CUSTOM_GYM_ENVIRONMENTS_PATH = __package__
MUJOCO_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.mujoco'

MUJOCO_ENVIRONMENT_SPECS = (
    {
        'id': 'Swimmer-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.swimmer_v3:SwimmerEnv'),
    },
    {
        'id': 'Hopper-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.hopper_v3:HopperEnv'),
    },
    {
        'id': 'Walker2d-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.walker2d_v3:Walker2dEnv'),
    },
    {
        'id': 'HalfCheetah-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv'),
    },
    {
        'id': 'Ant-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.ant_v3:AntEnv'),
    },
    {
        'id': 'Humanoid-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.humanoid_v3:HumanoidEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImagePusher2dEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImageForkReacher2dEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:BlindForkReacher2dEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
)


def register_mujoco_environments():
    """Register softlearning mujoco environments."""
    for mujoco_environment in MUJOCO_ENVIRONMENT_SPECS:
        gym.register(**mujoco_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MUJOCO_ENVIRONMENT_SPECS)

    return gym_ids


def register_general_environments():
    """Register gym environments that don't fall under a specific category."""
    for general_environment in GENERAL_ENVIRONMENT_SPECS:
        gym.register(**general_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  GENERAL_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
    )
