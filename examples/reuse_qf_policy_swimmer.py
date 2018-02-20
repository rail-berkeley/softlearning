"""Example script for training from an existing Q-function and Policy"""
import argparse

import joblib
import tensorflow as tf
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize

from softqlearning.algorithms import SQL
from softqlearning.misc.instrument import run_sql_experiment
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc.sampler import SimpleSampler
from softqlearning.misc.utils import timestamp
from softqlearning.replay_buffers import SimpleReplayBuffer


def run_experiment(variant):
    env = normalize(SwimmerEnv())

    pool = SimpleReplayBuffer(
        env_spec=env.spec, max_replay_buffer_size=1e6)

    sampler = SimpleSampler(
        max_path_length=1000,
        min_pool_size=1000,
        batch_size=128)

    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=500,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler)

    with tf.Session().as_default():
        data = joblib.load(variant['file'])
        if 'algo' in data.keys():
            saved_qf = data['algo'].qf
            saved_policy = data['algo'].policy
        else:
            saved_qf = data['qf']
            saved_policy = data['policy']

        algorithm = SQL(
            base_kwargs=base_kwargs,
            env=env,
            pool=pool,
            qf=saved_qf,
            policy=saved_policy,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            value_n_particles=16,
            td_target_update_interval=1000,
            qf_lr=3E-4,
            policy_lr=3E-4,
            discount=0.99,
            reward_scale=30,
            use_saved_qf=True,
            use_saved_policy=True,
            save_full_state=False)

        algorithm.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    args = parser.parse_args()
    return args


def main():
    full_experiment_name = 'swimmer'
    full_experiment_name += '-' + timestamp()
    args = parse_args()
    saved_file = args.file
    run_sql_experiment(
        run_experiment,
        mode='local',
        variant=dict(file=saved_file),
        exp_prefix='swimmer' + '/' + 'reuse' + '/' + timestamp(),
        exp_name=full_experiment_name,
        n_parallel=1,
        seed=1,
        terminate_machine=True,
        log_dir=None,
        snapshot_mode='gap',
        snapshot_gap=100,
        sync_s3_pkl=True)
    return


if __name__ == "__main__":
    main()
