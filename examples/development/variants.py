
from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.utils.git import get_git_rev
from softlearning.utils.misc import get_host_name
from softlearning.utils.dict import deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

M = 256
NUM_COUPLING_LAYERS = 2


ALGORITHM_PARAMS_BASE = {
    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),

            'discount': 0.99,
            'tau': 5e-3,
            'reward_scale': 1.0,
        },
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'discount': 0.99,
            'tau': 5e-3,
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        },
    },
}


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
        'observation_keys': None,
        'observation_preprocessors_params': {}
    }
}

TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: int(1e4),
    'gym': {
        DEFAULT_KEY: int(1e4),
        'Swimmer': {
            DEFAULT_KEY: int(1e5),
            'v3': int(5e5),
        },
        'Hopper': {
            DEFAULT_KEY: int(5e6),
            'v3': int(5e6),
        },
        'HalfCheetah': {
            DEFAULT_KEY: int(3e6),
            'v3': int(3e6),
        },
        'Walker2d': {
            DEFAULT_KEY: int(5e6),
            'v3': int(5e6),
        },
        'Ant': {
            DEFAULT_KEY: int(3e6),
            'v3': int(3e6),
        },
        'Humanoid': {
            DEFAULT_KEY: int(3e6),
            'Stand-v3': int(1e8),
            'SimpleStand-v3': int(1e8),
            'v3': int(1e8),
        },
        'Pendulum': {
            DEFAULT_KEY: int(1e4),
            'v3': int(1e4),
        },
        'Point2DEnv': {
            DEFAULT_KEY: int(5e4),
        }
    },
    'dm_control': {
        # BENCHMARKING
        DEFAULT_KEY: int(3e6),
        'acrobot': {
            DEFAULT_KEY: int(3e6),
            # 'swingup': int(None),
            # 'swingup_sparse': int(None),
        },
        'ball_in_cup': {
            DEFAULT_KEY: int(3e6),
            # 'catch': int(None),
        },
        'cartpole': {
            DEFAULT_KEY: int(3e6),
            # 'balance': int(None),
            # 'balance_sparse': int(None),
            # 'swingup': int(None),
            # 'swingup_sparse': int(None),
            # 'three_poles': int(None),
            # 'two_poles': int(None),
        },
        'cheetah': {
            DEFAULT_KEY: int(3e6),
            'run': int(1e7),
        },
        'finger': {
            DEFAULT_KEY: int(3e6),
            # 'spin': int(None),
            # 'turn_easy': int(None),
            # 'turn_hard': int(None),
        },
        'fish': {
            DEFAULT_KEY: int(3e6),
            # 'upright': int(None),
            # 'swim': int(None),
        },
        'hopper': {
            DEFAULT_KEY: int(3e6),
            # 'stand': int(None),
            'hop': int(1e7),
        },
        'humanoid': {
            DEFAULT_KEY: int(1e7),
            'stand': int(1e7),
            'walk': int(1e7),
            'run': int(1e7),
            # 'run_pure_state': int(1e7),
        },
        'manipulator': {
            DEFAULT_KEY: int(3e6),
            'bring_ball': int(1e7),
            # 'bring_peg': int(None),
            # 'insert_ball': int(None),
            # 'insert_peg': int(None),
        },
        'pendulum': {
            DEFAULT_KEY: int(3e6),
            # 'swingup': int(None),
        },
        'point_mass': {
            DEFAULT_KEY: int(3e6),
            # 'easy': int(None),
            # 'hard': int(None),
        },
        'reacher': {
            DEFAULT_KEY: int(3e6),
            # 'easy': int(None),
            # 'hard': int(None),
        },
        'swimmer': {
            DEFAULT_KEY: int(3e6),
            # 'swimmer6': int(None),
            # 'swimmer15': int(None),
        },
        'walker': {
            DEFAULT_KEY: int(3e6),
            # 'stand': int(None),
            'walk': int(1e7),
            'run': int(1e7),
        },
        # EXTRA
        'humanoid_CMU': {
            DEFAULT_KEY: int(3e6),
            'run': int(1e7),
            # 'stand': int(None),
        },
        'quadruped': {
            DEFAULT_KEY: int(3e6),
            'run': int(1e7),
            'walk': int(1e7),
        },
    },
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Point2DEnv': {
            DEFAULT_KEY: 50,
        },
        'Pendulum': {
            DEFAULT_KEY: 200,
        },
    },
}

EPOCH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Swimmer': {  # 2 DoF
        },
        'Hopper': {  # 3 DoF
        },
        'HalfCheetah': {  # 6 DoF
        },
        'Walker2d': {  # 6 DoF
        },
        'Ant': {  # 8 DoF
            'Parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'Humanoid': {  # 17 DoF
            'Parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'Pusher2d': {  # 3 DoF
            'Default-v3': {
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 1.0,
                'goal': (0, -1),
            },
            'DefaultReach-v0': {
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'ImageDefault-v0': {
                'image_shape': (32, 32, 3),
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 3.0,
            },
            'ImageReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'BlindReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            }
        },
        'Point2DEnv': {
            'Default-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
            'Wall-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
        },
        'Sawyer': {
            task_name: {
                'has_renderer': False,
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
                'reward_shaping': tune.grid_search([True, False]),
            }
            for task_name in (
                    'Lift',
                    'NutAssembly',
                    'NutAssemblyRound',
                    'NutAssemblySingle',
                    'NutAssemblySquare',
                    'PickPlace',
                    'PickPlaceBread',
                    'PickPlaceCan',
                    'PickPlaceCereal',
                    'PickPlaceMilk',
                    'PickPlaceSingle',
                    'Stack',
            )
        },
    },
    'dm_control': {
        'ball_in_cup': {
            'catch': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': True,
                    'render_kwargs': {
                        'pixels': {
                            'width': 84,
                            'height': 84,
                            'camera_id': 0,
                        },
                    },
                },
            },
        },
        'cheetah': {
            'run': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': True,
                    'render_kwargs': {
                        'pixels': {
                            'width': 84,
                            'height': 84,
                            'camera_id': 0,
                        },
                    },
                },
            },
        },
        'finger': {
            'spin': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': True,
                    'render_kwargs': {
                        'pixels': {
                            'width': 84,
                            'height': 84,
                            'camera_id': 0,
                        },
                    },
                },
            },
        },
    },
}


def get_epoch_length(universe, domain, task):
    level_result = EPOCH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    num_exploration_episodes = 10
    initial_exploration_steps = num_exploration_episodes * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    num_checkpoints = 10
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // num_checkpoints

    return checkpoint_frequency


def get_policy_params(spec):
    # config = spec.get('config', spec)
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_total_timesteps(universe, domain, task):
    level_result = TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, (int, float)):
            return level_result

        level_result = (
            level_result.get(level_key)
            or level_result[DEFAULT_KEY])

    return level_result


def get_algorithm_params(universe, domain, task):
    total_timesteps = get_total_timesteps(universe, domain, task)
    epoch_length = get_epoch_length(universe, domain, task)
    n_epochs = total_timesteps / epoch_length
    assert n_epochs == int(n_epochs)
    algorithm_params = {
        'kwargs': {
            'n_epochs': int(n_epochs),
            'n_initial_exploration_steps': tune.sample_from(
                get_initial_exploration_steps),
            'epoch_length': epoch_length,
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task):
    environment_params = (
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
        get_algorithm_params(universe, domain, task),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': tune.sample_from(get_policy_params),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'observation_preprocessors_params': {}
            },
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6),
            },
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': get_max_path_length(universe, domain, task),
                'batch_size': 256,
            }
        },
        'run_params': {
            'host_name': get_host_name(),
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def is_image_env(universe, domain, task, variant_spec):
    return 'pixel_wrapper_kwargs' in (
        variant_spec['environment_params']['training']['kwargs'])


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if is_image_env(universe, domain, task, variant_spec):
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': 'layer',
                'downsampling_type': 'conv',
            },
        }

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
