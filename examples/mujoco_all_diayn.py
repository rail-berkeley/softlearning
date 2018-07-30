"""Script for launching DIAYN experiments.

Usage:
    python mujoco_all_diayn.py --env=point --log_dir=/dev/null
"""
import os

import numpy as np

from ray import tune
from ray.tune.variant_generator import generate_variants

from rllab.envs.env_spec import EnvSpec
from rllab import spaces

from softlearning.algorithms import DIAYN
from softlearning.environments.utils import get_environment
from softlearning.policies.gmm import GMMPolicy
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions import (
    NNQFunction,
    NNVFunction,
    NNDiscriminatorFunction)

from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)


COMMON_PARAMS = {
    'seed': tune.grid_search([1]),
    'lr': 3E-4,
    'discount': 0.99,
    'tau': 0.01,
    'K': 4,
    'layer_size': 300,
    'batch_size': 128,
    'max_size': 1E6,
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
}

TAG_KEYS = ['seed']

ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'Swimmer-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'hopper': {  # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'half-cheetah': {  # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'max_size': 1E7,
    },
    'walker': {  # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'ant': {  # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'humanoid': {  # 21 DoF
        'prefix': 'humanoid',
        'env_name': 'Humanoid-v1',
        'max_path_length': 1000,
        'n_epochs': 20000,
    },
    'point': {
        'prefix': 'point',
        'env_name': 'point-rllab',
        'layer_size': 32,
        'max_path_length': 100,
        'n_epochs': 50,
        'target_entropy': -1,
    },
    'inverted-pendulum': {
        'prefix': 'inverted-pendulum',
        'env_name': 'InvertedPendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'inverted-double-pendulum': {
        'prefix': 'inverted-double-pendulum',
        'env_name': 'InvertedDoublePendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'pendulum': {
        'prefix': 'pendulum',
        'env_name': 'Pendulum-v0',
        'max_path_length': 200,
        'layer_size': 32,
        'n_epochs': 1000,
        'num_skills': 5,
    },
    'mountain-car': {
        'prefix': 'mountain-car',
        'env_name': 'MountainCarContinuous-v0',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'add_p_z': False,
    },
    'lunar-lander': {
        'prefix': 'lunar-lander',
        'env_name': 'LunarLanderContinuous-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'bipedal-walker': {
        'prefix': 'bipedal-walker',
        'env_name': 'BipedalWalker-v2',
        'max_path_length': 1600,
        'n_epochs': 1000,
        'scale_entropy': 0.1,
    },
}


def run_experiment(variant):
    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']

    env = get_environment(universe, domain, task, env_params={})

    obs_space = env.spec.observation_space
    assert isinstance(obs_space, spaces.Box)
    low = np.hstack([obs_space.low, np.full(variant['num_skills'], 0)])
    high = np.hstack([obs_space.high, np.full(variant['num_skills'], 1)])
    aug_obs_space = spaces.Box(low=low, high=high)
    aug_env_spec = EnvSpec(aug_obs_space, env.spec.action_space)
    pool = SimpleReplayPool(
        observation_shape=aug_env_spec.observation_space.shape,
        action_shape=aug_env_spec.action_space.shape,
        max_size=variant['max_size'],
    )

    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = variant['layer_size']
    qf = NNQFunction(
        observation_shape=aug_env_spec.observation_space.shape,
        action_shape=aug_env_spec.action_space.shape,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        observation_shape=aug_env_spec.observation_space.shape,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        observation_shape=aug_env_spec.observation_space.shape,
        action_shape=aug_env_spec.action_space.shape,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )

    discriminator = NNDiscriminatorFunction(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=[M, M],
        num_skills=variant['num_skills'],
    )

    algorithm = DIAYN(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        discriminator=discriminator,
        pool=pool,
        qf=qf,
        vf=vf,

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
            'log_dir_base': args.log_dir,
            'log_dir': build_tagged_log_dir
        },
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments_rllab(variants, args, run_experiment)


if __name__ == '__main__':
    main()
