"""Script for launching DIAYN experiments.

Usage:
    python mujoco_all_diayn.py --env=point --snapshot_dir=<snapshot_dir> --log_dir=/dev/null
"""

from rllab.misc.instrument import VariantGenerator

from sac.algos import SAC
from sac.envs.meta_env import FixedOptionEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.sampler import rollouts
from sac.misc.utils import timestamp
from sac.policies.hierarchical_policy import FixedOptionPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction

import argparse
import joblib
import numpy as np
import os
import tensorflow as tf


SHARED_PARAMS = {
    'seed': 1,
    'lr': 3E-4,
    'discount': 0.99,
    'tau': 0.01,
    'layer_size': 300,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 10,
    'sync_pkl': True,
    'use_pretrained_values': False, # Whether to use qf and vf from pretraining
}

TAG_KEYS = ['lr', 'use_pretrained_values']

ENV_PARAMS = {
    'swimmer': { # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'Swimmer-v1',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'scale_reward': 100,
    },
    'hopper': { # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'scale_reward': 1,
    },
    'half-cheetah': { # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
        'max_pool_size': 1E7,
    },
    'walker': { # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'scale_reward': 3,
    },
    'ant': { # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'scale_reward': 3,
    },
    'humanoid': { # 21 DoF
        'prefix': 'humanoid',
        'env_name': 'Humanoid-v1',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'scale_reward': 3,
    },
    'point': {
        'prefix': 'point',
        'env_name': 'point-rllab',
        'layer_size': 32,
        'max_path_length': 100,
        'n_epochs': 50,
        'scale_reward': 1,
    },
    'inverted-pendulum': {
        'prefix': 'inverted-pendulum',
        'env_name': 'InvertedPendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
    },
    'inverted-double-pendulum': {
        'prefix': 'inverted-double-pendulum',
        'env_name': 'InvertedDoublePendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
    },
    'pendulum': {
        'prefix': 'pendulum',
        'env_name': 'Pendulum-v0',
        'layer_size': 32,
        'max_path_length': 200,
        'n_epochs': 50,
        'scale_reward': 1,
    },
    'mountain-car': {
        'prefix': 'mountain-car',
        'env_name': 'MountainCarContinuous-v0',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
    },
    'lunar-lander': {
        'prefix': 'lunar-lander',
        'env_name': 'LunarLanderContinuous-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
    },
    'bipedal-walker': {
        'prefix': 'bipedal-walker',
        'env_name': 'BipedalWalker-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
    },
}
DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default='swimmer')
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default=None)
    args = parser.parse_args()

    return args

def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params.update(env_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])
    vg.add('snapshot_filename', [args.snapshot])
    return vg



def get_best_skill(policy, env, num_skills, max_path_length):
    tf.logging.info('Finding best skill to finetune...')
    reward_list = []
    with policy.deterministic(True):
        for z in range(num_skills):
            fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
            new_paths = rollouts(env, fixed_z_policy,
                                 max_path_length, n_paths=2)
            total_returns = np.mean([path['rewards'].sum() for path in new_paths])
            tf.logging.info('Reward for skill %d = %.3f', z, total_returns)
            reward_list.append(total_returns)

    best_z = np.argmax(reward_list)
    tf.logging.info('Best skill found: z = %d, reward = %d', best_z,
                    reward_list[best_z])
    return best_z



def run_experiment(variant):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session() as sess:
        data = joblib.load(variant['snapshot_filename'])
        policy = data['policy']
        env = data['env']

        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim
        best_z = get_best_skill(policy, env, num_skills, variant['max_path_length'])
        fixed_z_env = FixedOptionEnv(env, num_skills, best_z)

        tf.logging.info('Finetuning best skill...')

        pool = SimpleReplayBuffer(
            env_spec=fixed_z_env.spec,
            max_replay_buffer_size=variant['max_pool_size'],
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
            qf = data['qf']
            vf = data['vf']
        else:
            del data['qf']
            del data['vf']

            qf = NNQFunction(
                env_spec=fixed_z_env.spec,
                hidden_layer_sizes=[M, M],
                var_scope='qf-finetune',
            )

            vf = NNVFunction(
                env_spec=fixed_z_env.spec,
                hidden_layer_sizes=[M, M],
                var_scope='vf-finetune',
            )

        algorithm = SAC(
            base_kwargs=base_kwargs,
            env=fixed_z_env,
            policy=policy,
            pool=pool,
            qf=qf,
            vf=vf,
            lr=variant['lr'],
            scale_reward=variant['scale_reward'],
            discount=variant['discount'],
            tau=variant['tau'],
            save_full_state=False,
        )

        algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        tag = 'finetune__'
        print(variant['snapshot_filename'])
        tag += variant['snapshot_filename'].split('/')[-2]
        tag += '____'
        tag += '__'.join(['%s_%s' % (key, variant[key]) for key in TAG_KEYS])
        log_dir = os.path.join(args.log_dir, tag)
        variant['video_dir'] = os.path.join(log_dir, 'videos')
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,  # Increasing this barely effects performance,
                           # but breaks learning of hierarchical policy.
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
