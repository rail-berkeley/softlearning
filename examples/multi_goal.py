import os

import numpy as np

from softlearning.algorithms import SAC
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.samplers import SimpleSampler
from softlearning.policies.utils import get_policy_from_variant
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions.utils import get_Q_function_from_variant
from examples.utils import get_parser, launch_experiments_ray


def run_experiment(variant, reporter):
    env = get_environment('gym', 'MultiGoal', 'default', {
        'actuation_cost_coeff': 1,
        'distance_cost_coeff': 0.1,
        'goal_reward': 1,
        'init_sigma': 0.1,
    })

    pool = SimpleReplayPool(
        observation_space=env.observation_space,
        action_space=env.action_space,
        max_size=1e6)

    sampler = SimpleSampler(
        max_path_length=30, min_pool_size=100, batch_size=64)

    Qs = get_Q_function_from_variant(variant, env)
    policy = get_policy_from_variant(variant, env, Qs, preprocessor=None)
    plotter = QFPolicyPlotter(
        Q=Qs[0],
        policy=policy,
        obs_lst=np.array(((-2.5, 0.0),
                          (0.0, 0.0),
                          (2.5, 2.5),
                          (-2.5, -2.5))),
        default_action=(np.nan, np.nan),
        n_samples=100)

    algorithm = SAC(
        sampler=sampler,
        reparameterize=True,
        epoch_length=100,
        n_epochs=1000,
        n_train_repeat=1,
        eval_render=True,
        eval_n_episodes=10,
        eval_deterministic=False,

        env=env,
        policy=policy,
        initial_exploration_policy=None,
        pool=pool,
        Qs=Qs,
        plotter=plotter,

        lr=3e-4,
        target_entropy=-2.0,
        discount=0.99,
        tau=1e-4,

        save_full_state=True,
    )

    for train_result in algorithm.train():
        reporter(**train_result)


def main():
    args = get_parser().parse_args()

    universe, domain, task = 'general', 'multigoal', 'default'
    local_dir = os.path.join(
        '~/ray_results', universe, domain, task)

    layer_size = 64
    variant_spec = {
        'seed': 1,

        'universe': universe,
        'domain': domain,
        'task': task,

        'policy': args.policy,
        'local_dir': local_dir,
        'layer_size': layer_size,
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
                'regularization_coeff': 1e-3,
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
            }
        },
    }

    launch_experiments_ray([variant_spec], args, local_dir, run_experiment)


if __name__ == "__main__":
    main()
