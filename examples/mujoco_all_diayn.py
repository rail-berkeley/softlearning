"""Script for launching DIAYN experiments.

Usage:
    python mujoco_all_diayn.py --env=point --log_dir=/dev/null
"""
import os

import numpy as np
from ray import tune

from gym import spaces

from softlearning.algorithms import DIAYN
from softlearning.environments.utils import get_environment_from_variant
from softlearning.policies.gmm import GMMPolicy
from softlearning.replay_pools import SimpleReplayPool
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.misc.nn import feedforward_model
from softlearning.value_functions.utils import (
    create_Q_function_from_variant,
    create_V_function_from_variant)

from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)

COMMON_PARAMS = {
    'algorithm_params': {
        'type': 'DIAYN'
    },
    'seed': tune.grid_search([1]),
    'lr': 3E-4,
    'discount': 0.99,
    'tau': 0.01,
    'K': 4,
    'layer_size': 256,
    'batch_size': 128,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 10,
    'sync_pkl': True,
    'num_skills': 50,
    'scale_entropy': 0.1,
    'include_actions': False,
    'learn_p_z': False,
    'add_p_z': True,
    'Q_params': {
        'hidden_layer_sizes': lambda spec: ((
            spec['layer_size'], spec['layer_size']))
    },
    'V_params': {
        'hidden_layer_sizes': lambda spec: ((
            spec['layer_size'], spec['layer_size']))
    },
    'discriminator_params': {'hidden_layer_sizes': (256, 256)},
    'sampler_params': {
        'type': 'SimpleSampler',
        'kwargs': {
            'max_path_length': 1000,
            'min_pool_size': 1000,
            'batch_size': 256,
        }
    },
    'replay_pool_params': {
        'max_size': 1e6
    },
}

TAG_KEYS = ['seed']

ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'env_name': 'Swimmer-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'hopper': {  # 3 DoF
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'half-cheetah': {  # 6 DoF
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'walker': {  # 6 DoF
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'ant': {  # 8 DoF
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'humanoid': {  # 21 DoF
        'env_name': 'Humanoid-v1',
        'max_path_length': 1000,
        'n_epochs': 20000,
    },
    'point': {
        'env_name': 'point-rllab',
        'layer_size': 32,
        'max_path_length': 100,
        'n_epochs': 50,
        'target_entropy': -1,
    },
    'inverted-pendulum': {
        'env_name': 'InvertedPendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'inverted-double-pendulum': {
        'env_name': 'InvertedDoublePendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'pendulum': {
        'env_name': 'Pendulum-v0',
        'max_path_length': 200,
        'layer_size': 32,
        'n_epochs': 1000,
        'num_skills': 5,
    },
    'mountain-car': {
        'env_name': 'MountainCarContinuous-v0',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'add_p_z': False,
    },
    'lunar-lander': {
        'env_name': 'LunarLanderContinuous-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'bipedal-walker': {
        'env_name': 'BipedalWalker-v2',
        'max_path_length': 1600,
        'n_epochs': 1000,
        'scale_entropy': 0.1,
    },
}


def run_experiment(variant):
    env = get_environment_from_variant(variant)

    obs_space = env.observation_space
    low = np.hstack([obs_space.low, np.full(variant['num_skills'], 0)])
    high = np.hstack([obs_space.high, np.full(variant['num_skills'], 1)])
    aug_obs_space = spaces.Box(low=low, high=high)

    replay_pool = SimpleReplayPool(
        observation_space=aug_obs_space,
        action_space=env.observation_space,
        **variant['replay_pool_params'])
    sampler = get_sampler_from_variant(variant)

    M = variant['layer_size']
    Q = create_Q_function_from_variant(
        variant['Q_params'],
        input_shapes=(aug_obs_space.shape, env.action_space.shape))
    V = create_V_function_from_variant(
        variant['V_params'],
        input_shapes=(aug_obs_space.shape, ))

    policy = GMMPolicy(
        observation_shape=aug_obs_space.shape,
        action_shape=env.action_space.shape,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        Q=Q,
        reg=0.001,
    )

    discriminator = feedforward_model(
        input_shapes=(env.active_observation_shape, ),
        output_size=variant['num_skills'],
        **variant['discriminator_params'])

    algorithm = DIAYN(
        sampler=sampler,
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,

        env=env,
        policy=policy,
        discriminator=discriminator,
        pool=replay_pool,
        Q=Q,
        V=V,

        lr=variant['lr'],
        scale_entropy=variant['scale_entropy'],
        discount=variant['discount'],
        tau=variant['tau'],
        num_skills=variant['num_skills'],
        save_full_state=False,
        include_actions=variant['include_actions'],
        learn_p_z=variant['learn_p_z'],
        add_p_z=variant['add_p_z'],
    )

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def build_tagged_log_dir(spec):
    tag = '__'.join(['%s_%s' % (key, spec[key]) for key in TAG_KEYS])
    log_dir = os.path.join(spec['log_dir_base'], tag)
    return log_dir


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[args.env],
        **{
            'log_dir_base': args.log_dir or '',
            'log_dir': build_tagged_log_dir
        },
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
            'env_params': {}
        })

    launch_experiments_rllab(variant_spec, args, run_experiment)


if __name__ == '__main__':
    main()
