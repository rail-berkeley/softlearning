import sys

import numpy as np

from softlearning import algorithms
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.samplers import SimpleSampler
from softlearning import policies
from softlearning.replay_pools import SimpleReplayPool
from softlearning import value_functions
from examples.instrument import run_example_local


def run_experiment(variant, reporter):
    training_environment = (
        get_environment('gym', 'MultiGoal', 'Default-v0', {
            'actuation_cost_coeff': 30,
            'distance_cost_coeff': 1,
            'goal_reward': 10,
            'init_sigma': 0.1,
        }))
    evaluation_environment = training_environment.copy()

    pool = SimpleReplayPool(
        environment=training_environment,
        max_size=1e6)

    sampler = SimpleSampler(max_path_length=30)

    variant['Q_params']['config'].update({
        'input_shapes': (
            training_environment.observation_shape,
            training_environment.action_shape,
        )
    })
    Qs = value_functions.get(variant['Q_params'])

    variant['policy_params']['config'].update({
        'action_range': (training_environment.action_space.low,
                         training_environment.action_space.high),
        'input_shapes': training_environment.observation_shape,
        'output_shape': training_environment.action_shape,
    })
    policy = policies.get(variant['policy_params'])

    plotter = QFPolicyPlotter(
        Q=Qs[0],
        policy=policy,
        obs_lst=np.array(((-2.5, 0.0),
                          (0.0, 0.0),
                          (2.5, 2.5),
                          (-2.5, -2.5))),
        default_action=(np.nan, np.nan),
        n_samples=100)

    variant['algorithm_params']['config'].update({
        'training_environment': training_environment,
        'evaluation_environment': evaluation_environment,
        'policy': policy,
        'Qs': Qs,
        'pool': pool,
        'sampler': sampler,
        'min_pool_size': 100,
        'batch_size': 64,
        'plotter': plotter,
    })
    algorithm = algorithms.get(variant['algorithm_params'])

    for train_result in algorithm.train():
        reporter(**train_result)


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.multi_goal', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
