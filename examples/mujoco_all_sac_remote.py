from ray import tune
try:
    from ray.tune.variant_generator import generate_variants
except ImportError:
    # TODO(hartikainen): generate_variants has moved in >0.5.0, and some of my
    # stuff uses newer version. Remove this once we bump up the version in
    # requirements.txt
    from ray.tune.suggest.variant_generator import generate_variants


from softlearning.algorithms import SAC
from softlearning.environments.utils import get_environment
from softlearning.samplers import RemoteSampler
from softlearning.policies.gmm import GMMPolicy
from softlearning.environments.rllab import DelayedEnv
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions import NNQFunction, NNVFunction
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)


COMMON_PARAMS = {
    "seed": tune.grid_search([1, 2, 3]),
    "lr": 3E-4,
    "discount": 0.99,
    "tau": 0.01,
    "K": 4,
    "layer_size": 128,
    "batch_size": 128,
    "max_size": 1E6,
    "n_train_repeat": 1,
    "epoch_length": 1000,
    "snapshot_mode": 'gap',
    "snapshot_gap": 100,
    "sync_pkl": True,
}


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'target_entropy': -2.0,
    },
    'hopper': {  # 3 DoF
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'target_entropy': -3.0,
    },
    'half-cheetah': {  # 6 DoF
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'target_entropy': -6.0,
        'max_size': 1E7,
    },
    'walker': {  # 6 DoF
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'target_entropy': -6.0,
    },
    'ant': {  # 8 DoF
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'target_entropy': -8.0,
    },
    'humanoid': {  # 21 DoF
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'target_entropy': -21.0,
    },
}


def run_experiment(variant):
    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']

    env = get_environment(universe, domain, task, env_params={})
    env = DelayedEnv(env, delay=0.01)

    pool = SimpleReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        max_size=variant['max_size'],
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
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        observation_shape=env.observation_space.shape,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
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
        target_entropy=variant['target_entropy'],
        discount=variant['discount'],
        tau=variant['tau'],

        save_full_state=False,
    )

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[args.env],
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments_rllab(variants, args, run_experiment)


if __name__ == '__main__':
    main()
