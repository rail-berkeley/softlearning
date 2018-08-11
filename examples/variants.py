from ray import tune

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

LSP_POLICY_PARAMS_BASE = {
    'type': 'lsp',
    'coupling_layers': 2,
    's_t_layers': 1,
    'action_prior': 'uniform',
    # 'preprocessing_layer_sizes': None,
    'preprocessing_output_nonlinearity': 'relu',
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
    'reparameterize': REPARAMETERIZE
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
    'pusher': {  # 3 DoF
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

VALUE_FUNCTION_PARAMS = {
    'layer_size': M,

}

ALGORITHM_PARAMS_BASE = {
    'lr': 3e-4,
    'discount': 0.998,
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
    'humanoid-standup': {  # 17 DoF
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
    'pusher': {  # 3 DoF
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
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
    'seed': tune.grid_search([1, 2, 3]),
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
    'pusher': {  # 21 DoF
        'snapshot_gap': 500
    },
    'sawyer-torque': {
        'snapshot_gap': 1000
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
    'pusher': {  # 3 DoF
        'default': {
            'arm_distance_coeff': 0.0,
            'goal': tune.grid_search([(0, -1)]),
        }
    },
    'sawyer-torque': {

    }
}


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
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS.get(domain, {})
        ),
        'replay_pool_params': REPLAY_POOL_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS.get(domain, {})),
    }

    return variant_spec
