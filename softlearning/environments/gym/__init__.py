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
        'id': 'Swimmer-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.swimmer_env:SwimmerEnv'),
    },
    {
        'id': 'Hopper-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper_env:HopperEnv'),
    },
    {
        'id': 'Walker2d-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d_env:Walker2dEnv'),
    },
    {
        'id': 'HalfCheetah-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah_env:HalfCheetahEnv'),
    },
    {
        'id': 'Ant-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant_env:AntEnv'),
    },
    {
        'id': 'Humanoid-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_env:HumanoidEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d_env:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d_env:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher:ImagePusherEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher:ImageForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher:BlindForkReacherEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
)

MULTIWORLD_ENVIRONMENT_SPECS = (
    {
        'id': 'Point2DEnv-Default-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
    {
        'id': 'Point2DEnv-Wall-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


MULTIWORLD_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MULTIWORLD_ENVIRONMENT_SPECS)

GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
    *MULTIWORLD_ENVIRONMENTS,
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


def register_multiworld_environments():
    """Register custom environments from multiworld package."""
    for multiworld_environment in MULTIWORLD_ENVIRONMENT_SPECS:
        gym.register(**multiworld_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MULTIWORLD_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()
    registered_multiworld_environments = register_multiworld_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
        *registered_multiworld_environments,
    )
