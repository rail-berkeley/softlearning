import argparse
import joblib
import os

from rllab.envs.normalized_env import normalize

from ray.tune.variant_generator import generate_variants

from softlearning.misc import tf_utils
from softlearning.misc.instrument import launch_experiment
from softlearning.algorithms import SQL
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softlearning.misc.utils import timestamp
from softlearning.replay_pools import UnionPool
from softlearning.value_functions import SumQFunction
from softlearning.policies import StochasticNNPolicy
from softlearning.environments.pusher import PusherEnv
from softlearning.samplers import DummySampler
from softlearning.misc.utils import PROJECT_PATH

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


def load_pool_and_qf(filename):
    with tf_utils.get_default_session().as_default():
        data = joblib.load(os.path.join(PROJECT_PATH, filename))

    return data['replay_pool'], data['qf']


def run_experiment(variant):
    env = normalize(PusherEnv(goal=variant.get('goal')))

    pool1, qf1 = load_pool_and_qf(variant['snapshot1'])
    pool2, qf2 = load_pool_and_qf(variant['snapshot2'])

    sampler = DummySampler(
        batch_size=variant['batch_size'],
        max_path_length=variant['max_path_length'])
    pool = UnionPool(pools=(pool1, pool2))

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
        pool=pool,
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

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def launch_experiments(variants, args):
    print('Launching {} experiments.'.format(len(variants)))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        full_experiment_name = variant['prefix']
        full_experiment_name += '-' + args.exp_name + '-' + str(i).zfill(2)

        launch_experiment(
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

    variant_spec = dict(
        SHARED_PARAMS,
        **ENV_PARAMS[args.env],
        **{'snapshot1', args.snapshot1, 'snapshot_2', args.snapshot2})
    from pdb import set_trace; from pprint import pprint; set_trace()
    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments(variants, args)


if __name__ == '__main__':
    main()
