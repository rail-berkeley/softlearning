"""Script for launching DIAYN experiments.

Usage:
    python mujoco_all_diayn.py \
        --env=point \
        --snapshot_dir=<snapshot_dir> \
        --log_dir=/dev/null
"""
import os
import joblib

import numpy as np
import tensorflow as tf
from ray import tune

from softlearning.algorithms import SAC
from softlearning.environments.rllab import FixedOptionEnv
from softlearning.samplers import rollouts
from softlearning.value_functions.utils import (
    create_Q_function_from_variant,
    create_V_function_from_variant)
from softlearning.policies.hierarchical_policy import FixedOptionPolicy
from softlearning.replay_pools import SimpleReplayPool
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)


COMMON_PARAMS = {
    'seed': tune.grid_search([1]),
    'lr': 3E-4,
    'discount': 0.99,
    'tau': 0.01,
    'layer_size': 300,
    'batch_size': 128,
    'max_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 10,
    'sync_pkl': True,
    'use_pretrained_values': False, # Whether to use Q and V from pretraining
}

TAG_KEYS = ['lr', 'use_pretrained_values']

ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'env_name': 'Swimmer-v1',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'target_entropy': -2,
    },
    'hopper': {  # 3 DoF
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'target_entropy': -3,
    },
    'half-cheetah': {  # 6 DoF
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'target_entropy': -6,
        'max_size': 1E7,
    },
    'walker': {  # 6 DoF
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'target_entropy': -6,
    },
    'ant': {  # 8 DoF
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'target_entropy': -8,
    },
    'humanoid': {  # 21 DoF
        'env_name': 'Humanoid-v1',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'target_entropy': -21,
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
        'target_entropy': -1
    },
    'inverted-double-pendulum': {
        'env_name': 'InvertedDoublePendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'target_entropy': -1,
    },
    'pendulum': {
        'env_name': 'Pendulum-v0',
        'layer_size': 32,
        'max_path_length': 200,
        'n_epochs': 50,
        'target_entropy': -1,
    },
    'mountain-car': {
        'env_name': 'MountainCarContinuous-v0',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'target_entropy': -1,
    },
    'lunar-lander': {
        'env_name': 'LunarLanderContinuous-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'target_entropy': -4,
    },
    'bipedal-walker': {
        'env_name': 'BipedalWalker-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'target_entropy': -4,
    },
}


def get_best_skill(policy, env, num_skills, max_path_length):
    tf.logging.info('Finding best skill to finetune...')
    reward_list = []
    with policy.deterministic(True):
        for z in range(num_skills):
            fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
            new_paths = rollouts(env, fixed_z_policy,
                                 max_path_length, n_paths=2)
            total_returns = np.mean([
                path['rewards'].sum() for path in new_paths])
            tf.logging.info('Reward for skill %d = %.3f', z, total_returns)
            reward_list.append(total_returns)

    best_z = np.argmax(reward_list)
    tf.logging.info('Best skill found: z = %d, reward = %d', best_z,
                    reward_list[best_z])
    return best_z


def run_experiment(variant):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session():
        data = joblib.load(variant['snapshot_filename'])
        policy = data['policy']
        env = data['env']

        num_skills = (
            np.prod(data['policy']._observation_shape)
            - np.prod(data['env'].observation_space.shape))
        best_z = get_best_skill(policy, env, num_skills, variant['max_path_length'])
        fixed_z_env = FixedOptionEnv(env, num_skills, best_z)

        tf.logging.info('Finetuning best skill...')

        pool = SimpleReplayPool(
            observation_shape=fixed_z_env.spec.observation_space.shape,
            action_shape=fixed_z_env.spec.action_space.shape,
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

        if variant['use_pretrained_values']:
            Q = data['Q']
            V = data['V']
        else:
            del data['Q']
            del data['V']

            raise NotImplementedError(
                "TODO(hartikainen): This was broken on 2018/10/20 when I"
                " refactored our value functions and algorithms to use"
                " keras layers.")

            Q = create_Q_function_from_variant(variant, env)
            V = create_V_function_from_variant(variant, env)

        algorithm = SAC(
            base_kwargs=base_kwargs,
            env=fixed_z_env,
            policy=policy,
            pool=pool,
            Q=Q,
            V=V,
            lr=variant['lr'],
            target_entropy=variant['target_entropy'],
            discount=variant['discount'],
            tau=variant['tau'],
            save_full_state=False,
        )

        # Do the training
        for epoch, mean_return in algorithm.train():
            pass


def build_tagged_log_dir(spec):
    tag = 'finetune__{}____{}'.format(
        spec['snapshot_filename'].split('/')[-2],
        '__'.join(['%s_%s' % (key, spec[key]) for key in TAG_KEYS]))
    log_dir = os.path.join(spec['log_dir_base'], tag)
    return log_dir


def build_video_dir(spec):
    log_dir = os.path.join(spec['log_dir'])
    video_dir = os.path.join(log_dir, 'videos')
    return video_dir


def main():
    parser = get_parser()
    parser.add_argument('--snapshot', type=str, default=None)
    args = parser.parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[domain],
        **{
            'log_dir_base': args.log_dir,
            'snapshot_filename': args.snapshot,
            'log_dir': build_tagged_log_dir,
            'video_dir': build_video_dir
        },
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    launch_experiments_rllab(variant_spec, args)


if __name__ == '__main__':
    main()
