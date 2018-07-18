import numpy as np
from ray import tune

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

LSP_POLICY_PARAMS = {
    'swimmer-gym': {  # 2 DoF
        'preprocessing_layer_sizes': (M, M, 4),
        's_t_units': 2,
    },
    'swimmer-rllab': {  # 2 DoF
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
    'ant-gym': {  # 8 DoF
        'preprocessing_layer_sizes': (M, M, 16),
        's_t_units': 8,
    },
    'ant-rllab': {  # 8 DoF
        'preprocessing_layer_sizes': (M, M, 16),
        's_t_units': 8,
    },
    'humanoid-gym': {  # 17 DoF
        'preprocessing_layer_sizes': (M, M, 34),
        's_t_units': 17,
    },
    'humanoid-rllab': {  # 21 DoF
        'preprocessing_layer_sizes': (M, M, 42),
        's_t_units': 21,
    },
    'pusher': { # 12 DoF
        'coupling_layers': 4,
        'preprocessing_layer_sizes': (M*2, M, M//2, 24),
        's_t_layers': 1,
        's_t_units': 12,
    },
}

GMM_POLICY_PARAMS_BASE = {
    'type': 'gmm',
    'K': 1,
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': False  # GMM can't be parameterized
}

GMM_POLICY_PARAMS = {
    'swimmer-gym': {  # 2 DoF
    },
    'swimmer-rllab': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant-gym': {  # 8 DoF
    },
    'ant-rllab': {  # 8 DoF
    },
    'humanoid-gym': {  # 17 DoF
    },
    'humanoid-rllab': {  # 21 DoF
    },
    'pusher': { # 12 DoF
    },
}

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'gaussian',
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE
}

GAUSSIAN_POLICY_PARAMS = {
    'swimmer-gym': {  # 2 DoF
    },
    'swimmer-rllab': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant-gym': {  # 8 DoF
    },
    'ant-rllab': {  # 8 DoF
    },
    'humanoid-gym': {  # 17 DoF
    },
    'humanoid-rllab': {  # 21 DoF
    },
    'pusher': { # 12 DoF
    },
}

POLICY_PARAMS = {
    'lsp': {
        k: dict(LSP_POLICY_PARAMS_BASE, **v)
        for k, v in LSP_POLICY_PARAMS.items()
    },
    'gmm': {
        k: dict(GMM_POLICY_PARAMS_BASE, **v)
        for k, v in GMM_POLICY_PARAMS.items()
    },
    'gaussian': {
        k: dict(GAUSSIAN_POLICY_PARAMS_BASE, **v)
        for k, v in GAUSSIAN_POLICY_PARAMS.items()
    },
}

PREPROCESSOR_PARAMS_BASE = {
    # 'function_name': 'feedforward',
    # 'kwargs': {
    #     'hidden_layer_sizes': (M, M),
    #     'output_size': 42,
    # },
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
    'pusher': {
        'function_name': 'simple_convnet',
        'kwargs': {
            'image_size': lambda spec: (
                spec['env_params']['image_size']),
            'num_outputs': 6  # 2*DoF
        }
    }
}

PREPROCESSOR_PARAMS = {
    'lsp': LSP_PREPROCESSOR_PARAMS,
    'gmm': {},
    'gaussian': {},
}

VALUE_FUNCTION_PARAMS = {
    'layer_size': M,

}

ENV_DOMAIN_PARAMS = {
    'swimmer-gym': {  # 2 DoF
    },
    'swimmer-rllab': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant-gym': {  # 8 DoF
    },
    'ant-rllab': {  # 8 DoF
    },
    'humanoid-gym': {  # 17 DoF
    },
    'humanoid-rllab': {  # 21 DoF
    },
    'pusher': { # 12 DoF
    },
}

ENV_PARAMS = {
    'swimmer-gym': {  # 2 DoF
    },
    'swimmer-rllab': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant-gym': {  # 8 DoF
    },
    'ant-rllab': {  # 8 DoF
        'resume-training': {
            'low_level_policy_path': [
                # 'ant-low-level-policy-00-00/itr_4000.pkl',
            ]
        },
        'cross-maze': {
            'terminate_at_goal': True,
            'goal_reward_weight': 1000,
            'goal_radius': 2,
            'velocity_reward_weight': 0,
            'ctrl_cost_coeff': 0,  # 1e-2,
            'contact_cost_coeff': 0,  # 1e-3,
            'survive_reward': 0,  # 5e-2,
            'goal_distance': 12,
            'goal_angle_range': (0, 2 * np.pi),

            'env_fixed_goal_position': tune.grid_search([[6, -6], [6, 6], [12, 0]]),

            'pre_trained_policy_path': []
        },
    },
    'humanoid-gym': {  # 17 DoF
        'resume-training': {
            'low_level_policy_path': [
                # 'humanoid-low-level-policy-00-00/itr_4000.pkl',
            ]
        }
    },
    'humanoid-rllab': {  # 21 DOF
    },
    'pusher': { # 12 DoF
        'image': {
            # Can't use tuples because they break ray.tune log_syncer
            'image_size': tune.grid_search(['16x16x3', '32x32x3'])
        }
    },
}

ALGORITHM_PARAMS_BASE = {
    'lr': 3e-4,
    'discount': 0.99,
    'target_update_interval': 1,
    'tau': 0.005,
    'target_entropy': 'auto',
    'reward_scale': 1.0,
    'store_extra_policy_info': True,

    'base_kwargs': {
        'epoch_length': 1000,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': int(1e3),
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'swimmer-gym': {  # 2 DoF
        'base_kwargs': {
            'n_epochs': int(5e2 + 1),
        }
    },
    'swimmer-rllab': {  # 2 DoF
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
    'ant-gym': {  # 8 DoF
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
            'n_initial_exploration_steps': int(1e4),
        }
    },
    'ant-rllab': {  # 8 DoF
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
            'n_initial_exploration_steps': int(1e4),
        }
    },
    'humanoid-gym': {  # 17 DoF
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
    'pusher': { # 12 DoF
        'base_kwargs': {
            'n_epochs': int(4e3 + 1),
            'n_initial_exploration_steps': int(1e4),
        }
    },
    'humanoid-rllab': {  # 21 DoF
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
    'humanoid-standup-gym': {  # 17 DoF
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
}

REPLAY_POOL_PARAMS = {
    'max_size': 1e6,
}

SAMPLER_PARAMS = {
    'max_path_length': 1000,
    'min_pool_size': 1000,
    'batch_size': 256,
}

RUN_PARAMS_BASE = {
    'seed': tune.grid_search([1,2,3,4,5]),
    'snapshot_mode': 'gap',
    'snapshot_gap': 1000,
    'sync_pkl': True,
}

RUN_PARAMS = {
    'swimmer-gym': {  # 2 DoF
        'snapshot_gap': 200
    },
    'swimmer-rllab': {  # 2 DoF
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
    'ant-gym': {  # 8 DoF
        'snapshot_gap': 2000
    },
    'ant-rllab': {  # 8 DoF
        'snapshot_gap': 2000
    },
    'humanoid-gym': {  # 21 DoF
        'snapshot_gap': 2000
    },
    'humanoid-rllab': {  # 21 DoF
        'snapshot_gap': 2000
    },
    'pusher': { # 17 DoF
        'snapshot_gap': 500
    },
}

DOMAINS = [
    'swimmer-gym',  # 2 DoF
    'swimmer-rllab',  # 2 DoF
    'hopper',  # 3 DoF
    'half-cheetah',  # 6 DoF
    'walker',  # 6 DoF
    'ant-gym',  # 8 DoF
    'ant-rllab',  # 8 DoF
    'humanoid-gym',  # 17 DoF
    'humanoid-rllab',  # 21 DoF
    'pusher',  # 12 DoF
]

TASKS = {
    'swimmer-gym': [
        'default',
    ],
    'swimmer-rllab': [
        'default',
        'multi-direction',
    ],
    'hopper': [
        'default',
    ],
    'half-cheetah': [
        'default',
    ],
    'walker': [
        'default',
    ],
    'ant-gym': [
        'default',
    ],
    'ant-rllab': [
        'default',
        'multi-direction',
        'cross-maze'
    ],
    'humanoid-gym': [
        'default',
        'standup'
    ],
    'humanoid-rllab': [
        'default',
        'multi-direction'
    ],
    'pusher': [
        'default',
        'image'
    ],
}


def parse_domain_and_task(env_name):
    domain = next(domain for domain in DOMAINS if domain in env_name)
    domain_tasks = TASKS[domain]
    task = next((task for task in domain_tasks if task in env_name), 'default')
    return domain, task

def get_variant_spec(domain, task, policy):
    variant_spec = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'preprocessor_params': deep_update(
            PREPROCESSOR_PARAMS_BASE,
            PREPROCESSOR_PARAMS[policy].get(domain, {}),
        ),
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS[domain]
        ),
        'replay_pool_params': REPLAY_POOL_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    return variant_spec
