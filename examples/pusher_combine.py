import joblib
import os

from ray import tune

from rllab.envs.normalized_env import normalize

from softlearning.misc import tf_utils
from softlearning.algorithms import SQL
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softlearning.replay_pools import UnionPool
from softlearning.value_functions import SumQFunction
from softlearning.policies import StochasticNNPolicy
from softlearning.environments.rllab.pusher import PusherEnv
from softlearning.samplers import DummySampler
from softlearning.misc.utils import PROJECT_PATH
from examples.utils import get_parser, launch_experiments_rllab


COMMON_PARAMS = {
    'seed': tune.grid_search([1]),
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
        'env_name': 'pusher',
        'max_path_length': 300,
        'n_epochs': 100,
        'goal': (-1, -1),
    }
}


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

    qf = SumQFunction(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        q_functions=(qf1, qf2))

    M = variant['layer_size']
    policy = StochasticNNPolicy(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
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


def main():
    parser = get_parser()
    parser.add_argument('--snapshot1', type=str, default='')
    parser.add_argument('--snapshot2', type=str, default='')
    args = parser.parse_args()

    universe, domain, task = 'rllab', 'pusher', 'default'

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[domain],
        **{
            'snapshot1': args.snapshot1,
            'snapshot2': args.snapshot2
        },
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    launch_experiments_rllab(variant_spec, args, run_experiment)


if __name__ == '__main__':
    main()
