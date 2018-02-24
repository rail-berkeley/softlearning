import argparse
import joblib
import os

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator

from softqlearning.misc import tf_utils
from softqlearning.misc.instrument import run_sql_experiment
from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc.utils import timestamp
from softqlearning.replay_buffers import UnionBuffer
from softqlearning.value_functions import SumQFunction
from softqlearning.policies import StochasticNNPolicy
from softqlearning.environments.pusher import PusherEnv
from softqlearning.misc.sampler import DummySampler
from softqlearning.misc.utils import PROJECT_PATH

SHARED_PARAMS = {
    'seed': 0,
    'policy_lr': 3E-4,
    'layer_size': 128,
    'batch_size': 128,
    'epoch_length': 100,
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}

ENV_PARAMS = {
    'pusher': {
        'prefix': 'pusher',
        'env_name': 'pusher',
        'max_path_length': 300,
        'n_epochs': 100,
        'goal': (-1, -1),
    }
}
DEFAULT_ENV = 'pusher'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, choices=AVAILABLE_ENVS, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--snapshot1', type=str, default='')
    parser.add_argument('--snapshot2', type=str, default='')
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

    vg.add('snapshot1', (args.snapshot1, ))
    vg.add('snapshot2', (args.snapshot2, ))

    return vg


def load_buffer_and_qf(filename):
    with tf_utils.get_default_session().as_default():
        data = joblib.load(os.path.join(PROJECT_PATH, filename))

    return data['replay_buffer'], data['qf']


def run_experiment(variant):
    env = normalize(PusherEnv(goal=variant.get('goal')))

    buffer1, qf1 = load_buffer_and_qf(variant['snapshot1'])
    buffer2, qf2 = load_buffer_and_qf(variant['snapshot2'])

    sampler = DummySampler(
        batch_size=variant['batch_size'],
        max_path_length=variant['max_path_length'])
    buffer = UnionBuffer(buffers=(buffer1, buffer2))

    qf = SumQFunction(env.spec, q_functions=(qf1, qf2))

    M = variant['layer_size']
    policy = StochasticNNPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M, M),
        name='policy{i}'.format(i=0))

    base_kwargs = dict(
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler)

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=buffer,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=variant['kernel_particles'],
        kernel_update_ratio=variant['kernel_update_ratio'],
        policy_lr=variant['policy_lr'],
        save_full_state=False,
        train_policy=True,
        train_qf=False,
        use_saved_qf=True)

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    print('Launching {} experiments.'.format(len(variants)))

    for i, variant in enumerate(variants):
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
