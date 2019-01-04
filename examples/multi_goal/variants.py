from softlearning.misc.utils import deep_update

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 100,
        'n_epochs': 1000,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': False,

        'discount': 0.99,
        'reward_scale': 1.0,
        'save_full_state': True,
        'tau': 1e-4,
    }
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': True,
            'lr': 3e-4,
            'target_update_interval': 1,
            'target_entropy': -2.0,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'initial_exploration_policy': None
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'td_target_update_interval': 1,
        }
    }
}


def get_variant_spec(universe, domain, task, policy, local_dir, algorithm):
    layer_size = 64
    variant_spec = {
        'seed': 1,

        'universe': universe,
        'domain': domain,
        'task': task,

        'policy': policy,
        'local_dir': local_dir,
        'layer_size': layer_size,
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
            },
        },
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
            }
        },
        'run_params': {}
    }

    return variant_spec
