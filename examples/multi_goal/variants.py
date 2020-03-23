from softlearning.utils.dict import deep_update

ALGORITHM_PARAMS_BASE = {
    'class_name': 'SAC',

    'config': {
        'epoch_length': 100,
        'n_epochs': 1000,
        'n_train_repeat': 1,
        'eval_render_kwargs': {
            'mode': 'human',
        },
        'eval_n_episodes': 10,

        'discount': 0.99,
        'reward_scale': 1.0,
        'save_full_state': True,
        'target_update_interval': 1000,
        'tau': 1.0,
    }
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'class_name': 'SAC',
        'config': {
            'lr': 3e-4,
            'reward_scale': 0.1,
            'target_entropy': 'auto',
            'initial_exploration_policy': None
        }
    },
    'SQL': {
        'class_name': 'SQL',
        'config': {
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
            'class_name': 'FeedforwardGaussianPolicy',
            'config': {
                'hidden_layer_sizes': (layer_size, layer_size),
                'squash': True,
            },
        },
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        ),
        'Q_params': {
            'class_name': 'double_feedforward_Q_function',
            'config': {
                'hidden_layer_sizes': (layer_size, layer_size),
            },
        },
        'run_params': {
            'seed': 1,
        },
    }

    return variant_spec
