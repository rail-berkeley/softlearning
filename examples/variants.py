import numpy as np

from rllab.misc.instrument import VariantGenerator
from sac.misc.utils import flatten, get_git_rev, deep_update

LSP_POLICY_PARAMS_BASE = {
    'type': 'lsp',
    'coupling_layers': 2,
    's_t_layers': 1,
    'scale_regularization': 0,
    # 'preprocessing_hidden_sizes': None,
    'preprocessing_output_nonlinearity': 'relu',
    'squash': True
}

LSP_POLICY_PARAMS = {
    'swimmer': { # 2 DoF
        'preprocessing_hidden_sizes': (128, 128, 4),
        's_t_units': 2,
    },
    'hopper': { # 3 DoF
        'preprocessing_hidden_sizes': (128, 128, 6),
        's_t_units': 3,
    },
    'half-cheetah': { # 6 DoF
        'preprocessing_hidden_sizes': (128, 128, 12),
        's_t_units': 6,
    },
    'walker': { # 6 DoF
        'preprocessing_hidden_sizes': (128, 128, 12),
        's_t_units': 6,
    },
    'ant': { # 8 DoF
        'preprocessing_hidden_sizes': (128, 128, 16),
        's_t_units': 8,
    },
    'humanoid': { # 21 DoF
        'preprocessing_hidden_sizes': (128, 128, 42),
        's_t_units': 21,
    }
}

GMM_POLICY_PARAMS_BASE = {
    'type': 'gmm',
    'K': 4,
    'reg': 1e-3,
}

GMM_POLICY_PARAMS = {
    'swimmer': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid': { # 21 DoF
    }
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
}

VALUE_FUNCTION_PARAMS = {
    'layer_size': 128,

}

ENV_DOMAIN_PARAMS = {
    'swimmer': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid': { # 21 DoF
    }
}

ENV_PARAMS = {
    'swimmer': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
        'cross-maze': {
            'terminate_at_goal': True,
            'goal_reward_weight': [1000],
            'goal_radius': 2,
            'velocity_reward_weight': 0,
            'ctrl_cost_coeff': 0, # 1e-2,
            'contact_cost_coeff': 0, # 1e-3,
            'survive_reward': 0, # 5e-2,
            'goal_distance': np.linalg.norm([6, -6]),
            'goal_angle_range': (0, 2*np.pi),

            'pre_trained_policy_path': []
        }
    },
    'humanoid': { # 21 DoF
    }
}

ALGORITHM_PARAMS_BASE = {
    'lr': 3e-4,
    'discount': 0.99,
    'target_update_interval': 1,
    'tau': 1e-2,

    'base_kwargs': {
        'min_pool_size': 1000,
        'epoch_length': 1000,
        'max_path_length': 1000,
        'batch_size': 128,
        'n_train_repeat': 1,
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'swimmer': { # 2 DoF
        'scale_reward': 100,
        'base_kwargs': {
            'n_epochs': int(5e2 + 1),
        }
    },
    'hopper': { # 3 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': int(3e3 + 1),
        }
    },
    'half-cheetah': { # 6 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
    'walker': { # 6 DoF
        'scale_reward': 3,
        'base_kwargs': {
            'n_epochs': int(5e3 + 1),
        }
    },
    'ant': { # 8 DoF
        'scale_reward': 10,
        'base_kwargs': {
            'n_epochs': int(1e4 + 1),
        }
    },
    'humanoid': { # 21 DoF
        'scale_reward': 3,
        'base_kwargs': {
            'n_epochs': int(2e4 + 1),
        }
    }
}

REPLAY_BUFFER_PARAMS = {
    'max_replay_buffer_size': 1e6,
}

RUN_PARAMS = {
    'seed': [1,2,3,4,5],
    'snapshot_mode': 'gap',
    'snapshot_gap': 1000,
    'sync_pkl': True,
}


DOMAINS = [
    'swimmer', # 2 DoF
    'hopper', # 3 DoF
    'half-cheetah', # 6 DoF
    'walker', # 6 DoF
    'ant', # 8 DoF
    'humanoid', # 21 DoF
]

TASKS = {
    'swimmer': [
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
    'ant': [
        'default',
        'multi-direction',
        'cross-maze'

    ],
    'humanoid': [
        'default',
        'multi-direction'
    ],
}

def parse_domain_and_task(env_name):
    domain = next(domain for domain in DOMAINS if domain in env_name)
    domain_tasks = TASKS[domain]
    task = next((task for task in domain_tasks if task in env_name), 'default')
    return domain, task

def get_variants(domain, task, policy):
    params = dict(
        prefix='{}/{}'.format(domain, task),
        domain=domain,
        task=task,
        git_sha=get_git_rev(),

        env_params = ENV_PARAMS[domain].get(task, {}),
        policy_params = POLICY_PARAMS[policy][domain],
        value_fn_params = VALUE_FUNCTION_PARAMS,
        algorithm_params = deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS[domain]
        ),
        replay_buffer_params = REPLAY_BUFFER_PARAMS,
        run_params=RUN_PARAMS,
    )

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg
