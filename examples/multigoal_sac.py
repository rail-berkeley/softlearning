import os

import argparse

import numpy as np
import ray
from ray import tune

from softlearning.algorithms import SAC
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.misc.utils import datetimestamp
from softlearning.samplers import SimpleSampler
from softlearning.policies import GaussianPolicy, GMMPolicy, LatentSpacePolicy
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions.utils import (
    get_Q_function_from_variant,
    get_V_function_from_variant)
from examples.utils import (
    get_parser,
    launch_experiments_rllab,
    launch_experiments_ray)


def run_experiment(variant, reporter=None):
    env = get_environment('rllab', 'multigoal', 'default', {
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

    M = variant['layer_size']
    Qs = get_Q_function_from_variant(variant, env)
    V = get_V_function_from_variant(variant, env)

    if variant['policy'] == 'gaussian':
        policy = GaussianPolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(M, M),
            reparameterize=True,
            reg=1e-3)
    elif variant['policy'] == 'gmm':
        policy = GMMPolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            K=4,
            hidden_layer_sizes=[M, M],
            Q=Qs[0],
            reg=0.001
        )
    elif variant['policy'] == 'lsp':
        bijector_config = {
            "scale_regularization": 0.0,
            "num_coupling_layers": 2,
            "translation_hidden_sizes": (M,),
            "scale_hidden_sizes": (M,),
        }

        policy = LatentSpacePolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            mode="train",
            squash=True,
            bijector_config=bijector_config,
            observations_preprocessor=None,
            Q=Qs[0]
        )

    plotter = QFPolicyPlotter(
        Q=Qs[0],
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5],
                          [-2.5, -2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )

    algorithm = SAC(
        sampler=sampler,
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
        V=V,
        plotter=plotter,

        lr=3e-4,
        target_entropy=-6.0,
        discount=0.99,
        tau=1e-4,

        save_full_state=True,
    )

    for epoch, mean_return in algorithm.train():
        if reporter is not None:
            reporter(timesteps_total=epoch, mean_accuracy=mean_return)


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
        'V_params': {
            'type': 'feedforward_V_function',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
            }
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (layer_size, layer_size),
            }
        },
    }

    if args.mode == 'local':
        launch_experiments_rllab(variant_spec, args, run_experiment)
    elif 'rllab' in args.mode:
        args.mode = args.mode.replace('rllab', '').strip('-')
        launch_experiments_rllab(variant_spec, args, run_experiment)
    else:
        launch_experiments_ray([variant_spec], args, local_dir, run_experiment)


if __name__ == "__main__":
    main()
