"""Run SAC with asynchronous sampling.

This script demonstrates how we can train a policy and collect new experience
asynchronously using two processes. Asynchronous sampling is particularly
important when working with physical hardware and data collection becomes a
bottleneck. In that case, it is desirable to allocate all available compute to
optimizers rather then waiting for new sample to arrive.
"""
from ray import tune

from softlearning.environments.utils import get_environment
from softlearning.algorithms import SQL
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions import NNQFunction
from softlearning.policies import StochasticNNPolicy
from softlearning.environments.rllab import DelayedEnv
from softlearning.samplers import RemoteSampler
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)


COMMON_PARAMS = {
    'seed': tune.grid_search([1]),
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'reward_scale': 100,
    },
    'hopper': {  # 3 DoF
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'reward_scale': 1,
    },
    'half-cheetah': {  # 6 DoF
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 30,
        'max_size': 1E7,
    },
    'walker': {  # 6 DoF
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'reward_scale': 3,
    },
    'ant': {  # 8 DoF
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 3,
    },
    'humanoid': {  # 21 DoF
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 3,
    }
}


def run_experiment(variant):
    universe = variant['universe']

    assert universe in ['rllab', 'gym'], universe

    task = variant['task']
    domain = variant['domain']

    env = get_environment(universe, domain, task, env_params={})
    env = DelayedEnv(env, delay=0.01)

    pool = SimpleReplayPool(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        max_size=variant['max_size'])

    sampler = RemoteSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size'])

    base_kwargs = dict(
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler)

    M = variant['layer_size']
    qf = NNQFunction(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=(M, M))

    policy = StochasticNNPolicy(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=(M, M))

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        value_n_particles=16,
        td_target_update_interval=1000,
        qf_lr=variant['qf_lr'],
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

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[args.env],
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    launch_experiments_rllab(variant_spec, args, run_experiment)


if __name__ == '__main__':
    main()
