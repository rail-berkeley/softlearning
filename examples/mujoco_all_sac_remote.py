import argparse

from rllab.envs.normalized_env import normalize

from ray import tune
from ray.tune.variant_generator import generate_variants

from softlearning.algorithms import SAC
from softlearning.environments import GymEnv, DelayedEnv
from softlearning.misc.instrument import launch_experiment
from softlearning.misc.utils import timestamp
from softlearning.misc.remote_sampler import RemoteSampler
from softlearning.policies.gmm import GMMPolicy
from softlearning.replay_buffers import SimpleReplayBuffer
from softlearning.value_functions import NNQFunction, NNVFunction

COMMON_PARAMS = {
    "seed": tune.grid_search([1,2,3]),
    "lr": 3E-4,
    "discount": 0.99,
    "tau": 0.01,
    "K": 4,
    "layer_size": 128,
    "batch_size": 128,
    "max_pool_size": 1E6,
    "n_train_repeat": 1,
    "epoch_length": 1000,
    "snapshot_mode": 'gap',
    "snapshot_gap": 100,
    "sync_pkl": True,
}


ENV_PARAMS = {
    'swimmer': { # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'swimmer-rllab',
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
        'n_epochs': 10000,
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
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'scale_reward': 3,
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
    parser.add_argument('--exp_name',type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args


def run_experiment(variant):
    if variant['env_name'] == 'humanoid-rllab':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        env = normalize(HumanoidEnv())
    elif variant['env_name'] == 'swimmer-rllab':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        env = normalize(SwimmerEnv())
    else:
        env = normalize(GymEnv(variant['env_name']))
    env = DelayedEnv(env, delay=0.01)

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=variant['max_pool_size'],
    )

    sampler = RemoteSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size']
    )

    base_kwargs = dict(
        sampler=sampler,
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        env_spec=env.spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )


    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
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

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def launch_experiments(variants, args):
    num_experiments = len(variants)

    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))

        launch_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            # use_cloudpickle=True,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

def main():
    args = parse_args()

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[args.env])
    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments(variants, args)


if __name__ == '__main__':
    main()
