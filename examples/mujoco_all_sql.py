from ray import tune

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms import SQL
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.policies import StochasticNNPolicy
from softlearning.samplers import SimpleSampler
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)

LAYER_SIZE = 128
COMMON_PARAMS = {
    'seed': tune.grid_search([1, 2, 3]),
    'policy_lr': 3e-4,
    'Q_lr': 3e-4,
    'discount': 0.99,
    'layer_size': LAYER_SIZE,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
    'Q_params': {
        'type': 'double_feedforward_Q_function',
        'kwargs': {
            'hidden_layer_sizes': (LAYER_SIZE, LAYER_SIZE),
        }
    },
}


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 500,
        'reward_scale': 30,
    },
    'hopper': {  # 3 DoF
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'reward_scale': 30,
    },
    'half-cheetah': {  # 6 DoF
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 30,
        'max_pool_size': 1E7,
    },
    'walker': {  # 6 DoF
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'reward_scale': 10,
    },
    'ant': {  # 8 DoF
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 300,
    },
    'ant-rllab': {  # 8 DoF
        'env_name': 'ant-rllab',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 300
    },
    'humanoid': {  # 21 DoF
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    },
}


def run_experiment(variant):
    env = get_environment_from_variant(variant)

    pool = SimpleReplayPool(
        observation_space=env.observation_space,
        action_space=env.action_space,
        max_size=variant['max_pool_size'])

    sampler = SimpleSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size'])

    layer_size = variant['layer_size']
    Q = get_Q_function_from_variant(variant, env)

    policy = StochasticNNPolicy(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=(layer_size, layer_size))

    algorithm = SQL(
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler,

        env=env,
        pool=pool,
        Q=Q,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=variant['kernel_particles'],
        kernel_update_ratio=variant['kernel_update_ratio'],
        value_n_particles=variant['value_n_particles'],
        td_target_update_interval=variant['td_target_update_interval'],
        Q_lr=variant['Q_lr'],
        policy_lr=variant['policy_lr'],
        discount=variant['discount'],
        reward_scale=variant['reward_scale'],
        save_full_state=False)

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = {
        **COMMON_PARAMS,
        **ENV_PARAMS[domain],
        **{
            'prefix': '{}/{}/{}'.format(universe, domain, task),
            'universe': universe,
            'task': task,
            'domain': domain,
            'env_params': {},
        }
    }

    launch_experiments_rllab(variant_spec, args, run_experiment)


if __name__ == '__main__':
    main()
