import sys

import numpy as np

from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.samplers import SimpleSampler
from softlearning.policies.utils import get_policy_from_variant
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import initialize_tf_variables
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

    sampler = SimpleSampler(
        max_path_length=30, min_pool_size=100, batch_size=64)

    Qs = get_Q_function_from_variant(variant, training_environment)
    policy = get_policy_from_variant(variant, training_environment)
    plotter = QFPolicyPlotter(
        Q=Qs[0],
        policy=policy,
        obs_lst=np.array(((-2.5, 0.0),
                          (0.0, 0.0),
                          (2.5, 2.5),
                          (-2.5, -2.5))),
        default_action=(np.nan, np.nan),
        n_samples=100)

    algorithm = get_algorithm_from_variant(
        variant=variant,
        training_environment=training_environment,
        evaluation_environment=evaluation_environment,
        policy=policy,
        Qs=Qs,
        pool=pool,
        sampler=sampler,
        plotter=plotter
    )

    initialize_tf_variables(algorithm._session, only_uninitialized=True)

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
