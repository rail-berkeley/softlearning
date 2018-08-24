from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

LSP_POLICY_PARAMS_BASE = {
    'type': 'lsp',
    'coupling_layers': 2,
    's_t_layers': 1,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE,
    'squash': True
}

LSP_POLICY_PARAMS_FOR_DOMAIN = {
    'swimmer': {  # 2 DoF
        'preprocessing_layer_sizes': (M, M, 4),
        's_t_units': 2,
    },
    'hopper': {  # 3 DoF
        'preprocessing_layer_sizes': (M, M, 6),
        's_t_units': 3,
    },
    'half-cheetah': {  # 6 DoF
        'preprocessing_layer_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'walker': {  # 6 DoF
        'preprocessing_layer_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'ant': {  # 8 DoF
        'preprocessing_layer_sizes': (M, M, 16),
        's_t_units': 8,
    },
    'humanoid': {
        'preprocessing_layer_sizes': (M, M, 42),
        's_t_units': 21,
    },
    'pusher-2d': {  # 3 DoF
        's_t_units': 3,
    },
    'HandManipulatePen': {  # 20 DoF
        'preprocessing_layer_sizes': (M, M, 40),
        's_t_units': 128,
    },
    'HandManipulateEgg': {  # 20 DoF
        'preprocessing_layer_sizes': (M, M, 40),
        's_t_units': 128,
    },
    'HandManipulateBlock': {  # 20 DoF
        'preprocessing_layer_sizes': (M, M, 40),
        's_t_units': 128,
    },
    'HandReach': {  # 20 DoF
        'preprocessing_layer_sizes': (M, M, 40),
        's_t_units': 128,
    },
    'DClaw3': {  # 9 DoF
        'preprocessing_layer_sizes': (M, M, 18),
        's_t_units': 128,
    },
    'ImageDClaw3': {  # 9 DoF
        'preprocessing_layer_sizes': (M, M, 18),
        's_t_units': 128,
    },
}

GMM_POLICY_PARAMS_BASE = {
    'type': 'gmm',
    'K': 1,
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': False  # GMM can't be parameterized
}

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'gaussian',
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE,
    'hidden_layer_width': tune.grid_search([M//4])
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {
    'swimmer': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant': {  # 8 DoF
    },
    'humanoid': {  # 17/21 DoF (gym/rllab)
    },
    'pusher-2d': {  # 3 DoF
    },
    'sawyer-torque': {
    },
    'HandManipulatePen': {  # 20 DoF
    },
    'HandManipulateEgg': {  # 20 DoF
    },
    'HandManipulateBlock': {  # 20 DoF
    },
    'HandReach': {  # 20 DoF
    },
}

POLICY_PARAMS_BASE = {
    'lsp': LSP_POLICY_PARAMS_BASE,
    'gmm': GMM_POLICY_PARAMS_BASE,
    'gaussian': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_FOR_DOMAIN = {
    'lsp': LSP_POLICY_PARAMS_FOR_DOMAIN,
    'gmm': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
    'gaussian': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

PREPROCESSOR_PARAMS_BASE = {
    'function_name': None
}

LSP_PREPROCESSOR_PARAMS = {
    'swimmer-gym': {  # 2 DoF
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 4,  # 2*DoF
        }
    },
    'swimmer-rllab': {  # 2 DoF
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
        'function_name': 'feedforward',
        'kwargs': {
            'hidden_layer_sizes': (M, M),
            'output_size': 6,
        }
    },
}

PREPROCESSOR_PARAMS = {
    'lsp': LSP_PREPROCESSOR_PARAMS,
    'gmm': {},
    'gaussian': {},
}

VALUE_FUNCTION_PARAMS = {
    'layer_size': M,

}

ALGORITHM_PARAMS_BASE = {
    'lr': 3e-4,
    'discount': tune.grid_search([0.99]),
    'target_update_interval': 1,
    'tau': 0.005,
    'target_entropy': 'auto',
    'reward_scale': 1.0,
    'store_extra_policy_info': False,

    'base_kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': int(1e3),
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'swimmer': {  # 2 DoF
        'base_kwargs': {
            'n_epochs': int(5e2 + 1),
        }
    },
    'hopper': {  # 3 DoF
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
        }
    },
    'half-cheetah': {  # 6 DoF
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
            'n_initial_exploration_steps': int(1e4),
        }
    },
    'walker': {  # 6 DoF
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
        }
    },
    'ant': {  # 8 DoF
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
            'n_initial_exploration_steps': int(1e4),
        }
    },
    'humanoid': {  # 17/21 DoF (gym/rllab)
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
    'pusher-2d': {  # 3 DoF
        'base_kwargs': {
            'n_epochs': int(2e3 + 1),
            'n_initial_exploration_steps': int(1e4),
        }
    },
    'sawyer-torque': {
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
        }
    },
    'HandManipulatePen': {
        'base_kwargs': {
            'n_epochs': int(1e4 + 1)
        }
    },
    'HandManipulateEgg': {
        'base_kwargs': {
            'n_epochs': int(1e4 + 1)
        }
    },
    'HandManipulateBlock': {
        'base_kwargs': {
            'n_epochs': int(1e4 + 1)
        }
    },
    'HandReach': {
        'base_kwargs': {
            'n_epochs': int(1e4 + 1)
        }
    },
    'DClaw3': {
        'base_kwargs': {
            'n_epochs': int(5e2 + 1)
        }
    },
    'ImageDClaw3': {
        'base_kwargs': {
            'n_epochs': int(5e3 + 1)
        }
    }
}

REPLAY_POOL_PARAMS = {
    'max_size': 1e6,
}

SAMPLER_PARAMS = {
    'type': 'SimpleSampler',
    'kwargs': {
        'max_path_length': 100,
        'min_pool_size': 1000,
        'batch_size': 256,
    }
}

RUN_PARAMS_BASE = {
    'seed': tune.grid_search([1, 2, 3, 4]),
    'snapshot_mode': 'gap',
    'snapshot_gap': 1000,
    'sync_pkl': True,
}

RUN_PARAMS = {
    'swimmer': {  # 2 DoF
        'snapshot_gap': 200
    },
    'hopper': {  # 3 DoF
        'snapshot_gap': 600
    },
    'half-cheetah': {  # 6 DoF
        'snapshot_gap': 2000
    },
    'walker': {  # 6 DoF
        'snapshot_gap': 1000
    },
    'ant': {  # 8 DoF
        'snapshot_gap': 2000
    },
    'humanoid': {  # 17/21 DoF (gym/rllab)
        'snapshot_gap': 2000
    },
    'pusher-2d': {  # 21 DoF
        'snapshot_gap': 500
    },
    'sawyer-torque': {
        'snapshot_gap': 1000
    },
    'DClaw3': {
        'snapshot_gap': 100
    },
    'ImageDClaw3': {
        'snapshot_gap': 500
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
    },
    'humanoid': {  # 17/21 DoF (gym/rllab)
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
            'pose_difference_cost_coeff': 1.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
        }
    },
    'ImageDClaw3': {
        'Screw': {
            'image_size': '64x64x3',
            'object_target_distance_cost_coeff': 2.0,
            'pose_difference_cost_coeff': 1.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
        }
    }
}


def get_variant_spec(universe, domain, task, policy):
    variant_spec = {
        'prefix': '{}/{}/{}'.format(universe, domain, task),
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'preprocessor_params': deep_update(
            PREPROCESSOR_PARAMS_BASE,
            PREPROCESSOR_PARAMS[policy].get(domain, {}),
        ),
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS.get(domain, {})
        ),
        'replay_pool_params': REPLAY_POOL_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS.get(domain, {})),
    }

    return variant_spec


def get_variant_spec_image(universe, domain, task, policy, *args, **kwargs):
    variant_spec = get_variant_spec(
        universe, domain, task, policy, *args, **kwargs)

    if 'image' in task or 'image' in domain.lower():
        variant_spec['preprocessor_params'].update({
            'function_name': 'simple_convnet',
            'kwargs': {
                'image_size': variant_spec['env_params']['image_size'],
                'output_size': 18,
            }
        })

    if task == 'image-default':
        variant_spec['env_params'].update({
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search(['32x32x3']),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_distance_cost_coeff': 3.0,
        })
    elif task == 'image-reach':
        variant_spec['env_params'].update({
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search(['32x32x3']),
            'arm_goal_distance_cost_coeff': tune.grid_search([1.0]),
            'arm_object_distance_cost_coeff': 0.0,
        })
    elif task == 'blind-reach':
        variant_spec['env_params'].update({
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search(['32x32x3']),
            'arm_goal_distance_cost_coeff': tune.grid_search([1.0]),
            'arm_object_distance_cost_coeff': 0.0,
        })

    return variant_spec
