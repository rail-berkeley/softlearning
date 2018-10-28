from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

LSP_POLICY_PARAMS_BASE = {
    'type': 'LatentSpacePolicy',
    'kwargs': {
        'squash': True,
    }
}

NUM_COUPLING_LAYERS = 2

LSP_POLICY_PARAMS_FOR_DOMAIN = {
    'swimmer': {  # 2 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (2, 2),
            },
        }
    },
    'hopper': {  # 3 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (3, 3),
            },
        }
    },
    'half-cheetah': {  # 6 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (6, 6),
            },
        }
    },
    'walker': {  # 6 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (6, 6),
            },
        }
    },
    'ant': {  # 8 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (8, 8),
            },
        }
    },
    'humanoid': {
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (21, 21),
            },
        }
    },
    'pusher-2d': {  # 3 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (3, 3),
            },
        }
    },
    'HandManipulatePen': {  # 20 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (128, 128),
            },
        }
    },
    'HandManipulateEgg': {  # 20 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (128, 128),
            },
        }
    },
    'HandManipulateBlock': {  # 20 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (128, 128),
            },
        }
    },
    'HandReach': {  # 20 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (128, 128),
            },
        }
    },
    'DClaw3': {  # 9 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (128, 128),
            },
        }
    },
    'ImageDClaw3': {  # 9 DoF
        'kwargs': {
            'bijector_config': {
                'num_coupling_layers': NUM_COUPLING_LAYERS,
                'hidden_layer_sizes': (128, 128),
            },
        }
    },
}

GMM_POLICY_PARAMS_BASE = {
    'type': 'GMMPolicy',
    'kwargs': {
        'K': 1,
        'reg': 1e-3,
    }
}

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicyV2',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
        'regularization_coeff': 1e-3,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'LatentSpacePolicy': LSP_POLICY_PARAMS_BASE,
    'GMMPolicy': GMM_POLICY_PARAMS_BASE,
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'lsp': POLICY_PARAMS_BASE['LatentSpacePolicy'],
    'gmm': POLICY_PARAMS_BASE['GMMPolicy'],
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})


POLICY_PARAMS_FOR_DOMAIN = {
    'LatentSpacePolicy': LSP_POLICY_PARAMS_FOR_DOMAIN,
    'GMMPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'lsp': POLICY_PARAMS_FOR_DOMAIN['LatentSpacePolicy'],
    'gmm': POLICY_PARAMS_FOR_DOMAIN['GMMPolicy'],
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

PREPROCESSOR_PARAMS_BASE = {
    'LatentSpacePolicy': {
        'type': 'feedforward_preprocessor',
    },
    'GMMPolicy': {
        'type': None
    },
    'GaussianPolicy': {
        'type': None
    },
}

PREPROCESSOR_PARAMS_BASE.update({
    'lsp': PREPROCESSOR_PARAMS_BASE['LatentSpacePolicy'],
    'gmm': PREPROCESSOR_PARAMS_BASE['GMMPolicy'],
    'gaussian': PREPROCESSOR_PARAMS_BASE['GaussianPolicy'],
})


LSP_PREPROCESSOR_PARAMS = {
    'swimmer': {  # 2 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 4,  # 2*DoF
        }
    },
    'hopper': {  # 3 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 6,  # 2*DoF
        }
    },
    'half-cheetah': {  # 6 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 12,  # 2*DoF
        }
    },
    'walker': {  # 6 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 12,  # 2*DoF
        }
    },
    'ant-gym': {  # 8 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 16,  # 2*DoF
        }
    },
    'ant-rllab': {  # 8 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 16,  # 2*DoF
        }
    },
    'humanoid-gym': {  # 17 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 34,  # 2*DoF
        }
    },
    'humanoid-rllab': {  # 21 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 42,  # 2*DoF
        }
    },
    'pusher-2d': {
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 6,
        }
    },
}

PREPROCESSOR_PARAMS = {
    'LatentSpacePolicy': LSP_PREPROCESSOR_PARAMS,
    'GMMPolicy': {},
    'GaussianPolicy': {},
}

PREPROCESSOR_PARAMS.update({
    'lsp': PREPROCESSOR_PARAMS['LatentSpacePolicy'],
    'gmm': PREPROCESSOR_PARAMS['GMMPolicy'],
    'gaussian': PREPROCESSOR_PARAMS['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': int(1e3),
        'reparameterize': REPARAMETERIZE,
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'lr': 3e-4,
        'discount': 0.99,
        'target_update_interval': 1,
        'tau': 0.005,
        'target_entropy': 'auto',
        'reward_scale': 1.0,
        'store_extra_policy_info': False,
        'action_prior': 'uniform',
        'save_full_state': False,
    }
}

NUM_EPOCHS_PER_DOMAIN = {
    'swimmer': int(5e2 + 1),
    'hopper': int(3e3 + 1),
    'half-cheetah': int(3e3 + 1),
    'walker': int(3e3 + 1),
    'ant': int(3e3 + 1),
    'humanoid': int(1e4 + 1),
    'pusher-2d': int(2e3 + 1),
    'sawyer-torque': int(1e3 + 1),
    'HandManipulatePen': int(1e4 + 1),
    'HandManipulateEgg': int(1e4 + 1),
    'HandManipulateBlock': int(1e4 + 1),
    'HandReach': int(1e4 + 1),
    'DClaw3': int(5e2 + 1),
    'ImageDClaw3': int(5e3 + 1),
    'Point2DEnv': int(200 + 1)
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN[domain],
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENV_PARAMS = {
    'swimmer': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant': {  # 8 DoF
        'custom-default': {
            'survive_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'humanoid': {  # 17/21 DoF (gym/rllab)
        'custom-default': {
            'survive_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'pusher-2d': {  # 3 DoF
        'default': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': tune.grid_search([(0, -1)]),
        },
        'default-reach': {
            'arm_goal_distance_cost_coeff': tune.grid_search([1.0]),
            'arm_object_distance_cost_coeff': 0.0,
        },
    },
    'sawyer-torque': {

    },
    'DClaw3': {
        'ScrewV2': {
            'object_target_distance_cost_coeff': 2.0,
            'pose_difference_cost_coeff': 0.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        }
    },
    'ImageDClaw3': {
        'Screw': {
            'image_size': (32, 32, 3),
            'object_target_distance_cost_coeff': 2.0,
            'pose_difference_cost_coeff': 0.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        }
    },
    'Point2DEnv': {
        'default': {
            'observation_keys': ('observation', ),
        },
        'wall': {
            'observation_keys': ('observation', ),
        },
    }
}

NUM_SNAPSHOTS = 5


def get_variant_spec(universe, domain, task, policy):
    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'V_params': {
            'type': 'feedforward_V_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'preprocessor_params': deep_update(
            PREPROCESSOR_PARAMS_BASE[policy],
            PREPROCESSOR_PARAMS[policy].get(domain, {}),
        ),
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_PER_DOMAIN[domain]
        ),
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': 1e6,
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.grid_search([1, 2]),
            'snapshot_mode': 'gap',
            'snapshot_gap': NUM_EPOCHS_PER_DOMAIN[domain] // NUM_SNAPSHOTS,
            'sync_pkl': True,
        },
    }

    return variant_spec


def get_variant_spec_image(universe, domain, task, policy, *args, **kwargs):
    variant_spec = get_variant_spec(
        universe, domain, task, policy, *args, **kwargs)

    if 'image' in task or 'image' in domain.lower():
        variant_spec['preprocessor_params'].update({
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_size': variant_spec['env_params']['image_size'],
                'output_size': 18,
                'num_conv_layers': tune.grid_search([2, 3, 4]), # 4 later
                'filters_per_layer': tune.grid_search([16, 32]),
                'kernel_size_per_layer': (5, 5),
            }
        })

    if task == 'image-default':
        variant_spec['env_params'].update({
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search([(32, 32, 3)]),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_distance_cost_coeff': 3.0,
        })
    elif task == 'image-reach':
        variant_spec['env_params'].update({
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search([(32, 32, 3)]),
            'arm_goal_distance_cost_coeff': tune.grid_search([1.0]),
            'arm_object_distance_cost_coeff': 0.0,
        })
    elif task == 'blind-reach':
        variant_spec['env_params'].update({
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search([(32, 32, 3)]),
            'arm_goal_distance_cost_coeff': tune.grid_search([1.0]),
            'arm_object_distance_cost_coeff': 0.0,
        })

    return variant_spec
