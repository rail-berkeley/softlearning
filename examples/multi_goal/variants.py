from softlearning.misc.utils import deep_update

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 100,
        'n_epochs': 1000,
        'n_train_repeat': 1,
        'eval_render_kwargs': {
            'mode': 'human',
        },
        'eval_n_episodes': 10,
        'eval_deterministic': True,

        'discount': 0.99,
        'reward_scale': 1.0,
        'save_full_state': True,
        'target_update_interval': 1000,
        'tau': 1.0,
    }
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': True,
            'lr': 3e-4,
            'reward_scale': 0.1,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'initial_exploration_policy': None
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'reward_scale': 0.1,
            'value_n_particles': 16,
            'kernel_n_particles': 32,
            'kernel_update_ratio': 0.5,
        }
    }
}


def get_variant_spec(args):
    algorithm = args.algorithm

    layer_size = 128
    variant_spec = {
        'layer_size': layer_size,
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
                'squash': True,
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
            },
        },
        'run_params': {
            'seed': 1,
        },
    }

    return variant_spec
