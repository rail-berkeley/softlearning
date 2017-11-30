import argparse

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from sandbox.rocky.tf.envs.base import TfEnv

from sac.algos import SAC
from sac.envs.gym_env import GymEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='')
    parser.add_argument('--exp', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    args = parser.parse_args()

    return args


def get_variants(args):
    params = dict(
        seed=[1, 2, 3],
        lr=3E-4,
        discount=0.99,
        tau=0.01,
        K=4,
        layer_size=128,
        batch_size=128,
        max_pool_size=1E6,
        n_train_repeat=1,
        epoch_length=1000,
        snapshot_mode='gap',
        snapshot_gap=1000,
        sync_pkl=True,
    )

    if args.env == 'swimmer':  # 2 DoF
        params.update(dict(
            prefix='swimmer',
            env_name='swimmer-rllab',
            max_path_length=1000,
            n_epochs=2000,
            scale_reward=100,
        ))
    elif args.env == 'hopper':  # 3 DoF
        params.update(dict(
            prefix='hopper',
            env_name='Hopper-v1',
            max_path_length=1000,
            n_epochs=3000,
            scale_reward=1,
        ))
    elif args.env == 'half-cheetah':  # 6 DoF
        params.update(dict(
            prefix='half-cheetah',
            env_name='HalfCheetah-v1',
            max_path_length=1000,
            n_epochs=10000,
            scale_reward=1,
            max_pool_size=1E7,
        ))
    elif args.env == 'walker':  # 6 DoF
        params.update(dict(
            prefix='walker',
            env_name='Walker2d-v1',
            max_path_length=1000,
            n_epochs=5000,
            scale_reward=3,
        ))
    elif args.env == 'ant':  # 8 DoF
        params.update(dict(
            prefix='ant',
            env_name='Ant-v1',
            max_path_length=1000,
            n_epochs=10000,
            scale_reward=3,
        ))
    elif args.env == 'humanoid':  # 21 DoF
        params.update(dict(
            prefix='humanoid',
            env_name='humanoid-rllab',
            max_path_length=1000,
            n_epochs=20000,
            scale_reward=3,
        ))
    else:
        raise NotImplementedError

    vg = VariantGenerator()
    for key, val in params.items():
        if type(val) is list:
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def run_experiment(v):
    if v['env_name'] == 'humanoid-rllab':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        env = TfEnv(normalize(HumanoidEnv()))
    elif v['env_name'] == 'swimmer-rllab':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        env = normalize(SwimmerEnv())
    else:
        env = normalize(GymEnv(v['env_name']))

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=v['max_pool_size'],
    )

    base_kwargs = dict(
        min_pool_size=v['max_path_length'],
        epoch_length=v['epoch_length'],
        n_epochs=v['n_epochs'],
        max_path_length=v['max_path_length'],
        batch_size=v['batch_size'],
        n_train_repeat=v['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = v['layer_size']
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
        K=v['K'],
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

        lr=v['lr'],
        scale_reward=v['scale_reward'],
        discount=v['discount'],
        tau=v['tau'],

        save_full_state=False,
    )
    algorithm.train()


def launch_experiments(vg):
    for i, v in enumerate(vg.variants()):
        print('Launching {} experiments.'.format(len(vg.variants())))
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=v,
            exp_prefix=v['prefix'] + '/' + args.exp,
            exp_name=v['prefix'] + '-' + args.exp + '-' + str(i).zfill(2),
            n_parallel=1,
            seed=v['seed'],
            terminate_machine=True,
            snapshot_mode=v['snapshot_mode'],
            snapshot_gap=v['snapshot_gap'],
            sync_s3_pkl=v['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
