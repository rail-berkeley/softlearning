"""Run SAC with asynchronous sampling.

This script demonstrates how we can train a policy and collect new experience
asynchronously using two processes. Asynchronous sampling is particularly
important when working with physical hardware and data collection becomes a
bottleneck. In that case, it is desirable to allocate all available compute to
optimizers rather then waiting for new sample to arrive.
"""
import argparse

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.misc.instrument import VariantGenerator

from softqlearning.misc.instrument import run_sql_experiment
from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc.utils import timestamp
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.value_functions import NNQFunction
from softqlearning.policies import StochasticNNPolicy
from softqlearning.environments import GymEnv, DelayedEnv
from softqlearning.misc.remote_sampler import RemoteSampler

SHARED_PARAMS = {
    'seed': 1,
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'reward_scale': 100,
    },
    'hopper': {  # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'reward_scale': 1,
    },
    'half-cheetah': {  # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 30,
        'max_pool_size': 1E7,
    },
    'walker': {  # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'reward_scale': 3,
    },
    'ant': {  # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 3,
    },
    'humanoid': {  # 21 DoF
        'prefix': 'humanoid',
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 3,
    }
}
DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, choices=AVAILABLE_ENVS, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
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

    return vg


def run_experiment(variant):
    if variant['env_name'] == 'humanoid-rllab':
        env = normalize(HumanoidEnv())
    elif variant['env_name'] == 'swimmer-rllab':
        env = normalize(SwimmerEnv())
    else:
        env = normalize(GymEnv(variant['env_name']))

    env = DelayedEnv(env, delay=0.01)

    pool = SimpleReplayBuffer(
        env_spec=env.spec, max_replay_buffer_size=variant['max_pool_size'])

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
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=(M, M))

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

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        full_experiment_name = variant['prefix']
        full_experiment_name += '-' + args.exp_name + '-' + str(i).zfill(2)

        run_sql_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=full_experiment_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=True)


def main():
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator, args)


if __name__ == '__main__':
    main()
