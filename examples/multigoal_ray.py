import argparse

import numpy as np
import ray
from ray import tune

from softlearning.algorithms import SAC
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.misc.utils import datetimestamp
from softlearning.samplers import SimpleSampler
from softlearning.policies import GMMPolicy, LatentSpacePolicy
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions import NNQFunction, NNVFunction
from examples.utils import get_parser


def run(variant, reporter):
    env = get_environment('rllab', 'multigoal', 'default', {
        'actuation_cost_coeff': 1,
        'distance_cost_coeff': 0.1,
        'goal_reward': 1,
        'init_sigma': 0.1,
    })

    pool = SimpleReplayPool(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        max_size=1e6)

    sampler = SimpleSampler(
        max_path_length=30, min_pool_size=100, batch_size=64)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': 100,
        'n_epochs': 1000,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10,
        'eval_deterministic': False
    }

    M = 128
    q_functions = tuple(
        NNQFunction(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(M, M),
            name='qf{}'.format(i))
        for i in range(2))
    vf = NNVFunction(
        observation_shape=env.active_observation_shape,
        hidden_layer_sizes=[M, M])

    if variant['policy_type'] == 'gmm':
        policy = GMMPolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            K=4,
            hidden_layer_sizes=[M, M],
            qf=q_functions[0],
            reg=0.001
        )
    elif variant['policy_type'] == 'lsp':
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
            q_function=q_functions[0]
        )

    plotter = QFPolicyPlotter(
        qf=q_functions[0],
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=None,
        pool=pool,
        q_functions=q_functions,
        vf=vf,
        plotter=plotter,

        lr=3e-4,
        target_entropy=-6.0,
        discount=0.99,
        tau=1e-4,

        save_full_state=True
    )

    for epoch, mean_return in algorithm.train():
        reporter(timesteps_total=epoch, mean_accuracy=mean_return)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument(
        '--policy-type', type=str, choices=('gmm', 'lsp'), default='gmm')
    args = parser.parse_args()

    return args


def main():
    args = get_parser().parse_args()
    variants = {
        'policy_type': args.policy
    }

    tune.register_trainable('multigoal-runner', run)
    if args.mode == 'local':
        ray.init()
        local_dir_base = './data/ray/results'
    else:
        ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')
        local_dir_base = '~/ray_results'

    local_dir = '{}/multigoal/default'.format(local_dir_base)
    variants['local_dir'] = local_dir

    tune.run_experiments({
        'multigoal-' + datetimestamp(): {
            'run': 'multigoal-runner',
            'trial_resources': {'cpu': 2},
            'config': variants,
            'local_dir': local_dir
        }
    })


if __name__ == "__main__":
    main()
