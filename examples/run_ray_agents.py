import ray
import ray.tune as tune
from ray.tune.registry import register_env

from examples.utils import parse_universe_domain_task, get_parser

from softlearning.environments.utils import get_environment


def main():
    args = get_parser().parse_args()
    universe, domain, task = parse_universe_domain_task(args)
    env_id = '{}-{}-{}'.format(universe, domain, task)

    def env_creator(env_config):
        return get_environment(universe, domain, task, env_config)

    register_env(env_id, env_creator)

    if args.mode == 'local':
        ray.init()
    else:
        ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')

    local_dir_base = '~/ray_results'
    local_dir = local_dir_base

    tune.run_experiments({
        args.exp_name: {
            'run': 'PPO',
            'env': env_id,
            'stop': {},
            # 'trial_resources': {'cpu': 8},
            'config': {
                'env_config': {},

                # Discount factor of the MDP
                'gamma': 0.998,
                # If true, use the Generalized Advantage Estimator (GAE)
                # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
                'use_gae': True,
                # GAE(lambda) parameter
                'lambda': 0.95,
                # Coefficient of the entropy regularizer
                "entropy_coeff": 0.01,
                # PPO clip parameter
                "clip_param": 0.2,

                # Number of SGD iterations in each outer loop
                "num_sgd_iter": 20,
                # Stepsize of SGD
                "sgd_stepsize": 1e-4,
                # Total SGD batch size across all devices for SGD (multi-gpu only)
                # (M?)
                "sgd_batchsize": 8192, # 32768, # 25600,

                # Number of timesteps collected for each SGD round
                # (NT = horizon * num_workers)
                "timesteps_per_batch": 60000,

                # Default sample batch size
                # "sample_batch_size": 512,

                "horizon": 5000,  # 4096,

                'num_workers': 8,
                'num_gpus': 4,
            },
            'local_dir': local_dir,
            'upload_dir': 'gs://sac-ray-test/ray/results'
        },
    })


if __name__ == '__main__':
    main()
