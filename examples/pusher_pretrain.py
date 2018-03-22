import argparse
import pickle

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator

from softqlearning.environments.pusher import PusherEnv
from softqlearning.misc.instrument import run_sql_experiment
from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc.utils import timestamp
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.value_functions import NNQFunction
from softqlearning.policies import StochasticNNPolicy
from softqlearning.misc.sampler import SimpleSampler

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
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
    'save_full_state': True,
}

ENV_PARAMS = {
    'pusher': {
        'prefix': 'pusher',
        'env_name': 'pusher',
        'max_path_length': 300,
        'n_epochs': 500,
        'reward_scale': 1,
        'goal': [(-1, 'any'), ('any', -1)]
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
    if variant['env_name'] == 'pusher':
        # TODO: assumes `pusher.xml` is located in `rllab/models/` when
        # running on EC2.
        env = normalize(PusherEnv(goal=variant.get('goal')))
    else:
        raise ValueError

    pool = SimpleReplayBuffer(
        env_spec=env.spec, max_replay_buffer_size=variant['max_pool_size'])

    sampler = SimpleSampler(
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

    task_id = abs(pickle.dumps(variant).__hash__())

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=(M, M),
        name='qf_{i}'.format(i=task_id))

    policy = StochasticNNPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M, M),
        name='policy_{i}'.format(i=task_id))

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=variant['kernel_particles'],
        kernel_update_ratio=variant['kernel_update_ratio'],
        value_n_particles=variant['value_n_particles'],
        td_target_update_interval=variant['td_target_update_interval'],
        qf_lr=variant['qf_lr'],
        policy_lr=variant['policy_lr'],
        discount=variant['discount'],
        reward_scale=variant['reward_scale'],
        save_full_state=variant['save_full_state'])

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
            log_dir=args.log_dir + '/' + str(i).zfill(2),
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=True)


def main():
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator, args)


if __name__ == '__main__':
    main()
