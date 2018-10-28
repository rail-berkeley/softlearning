import argparse
from distutils.util import strtobool
import json
import math
import os

try:
    from ray.tune.variant_generator import generate_variants
except ImportError:
    # TODO(hartikainen): generate_variants has moved in >0.5.0, and some of my
    # stuff uses newer version. Remove this once we bump up the version in
    # requirements.txt
    from ray.tune.suggest.variant_generator import generate_variants

import softlearning.environments.utils as env_utils
from softlearning.misc.utils import datetimestamp, datestamp
from softlearning.misc.instrument import launch_experiment


DEFAULT_UNIVERSE = 'gym'
DEFAULT_TASK = 'default'

TASKS_BY_DOMAIN_BY_UNIVERSE = {
    universe: {
        domain: tuple(tasks.keys())
        for domain, tasks in domains.items()
    }
    for universe, domains in env_utils.ENVIRONMENTS.items()
}

AVAILABLE_TASKS = set(sum(
    [
        tasks
        for universe, domains in TASKS_BY_DOMAIN_BY_UNIVERSE.items()
        for domain, tasks in domains.items()
    ],
    ()))

DOMAINS_BY_UNIVERSE = {
    universe: tuple(domains.keys())
    for universe, domains in env_utils.ENVIRONMENTS.items()
}

AVAILABLE_DOMAINS = set(sum(DOMAINS_BY_UNIVERSE.values(), ()))

UNIVERSES = tuple(env_utils.ENVIRONMENTS.keys())


def parse_universe(env_name):
    universe = next(
        (universe for universe in UNIVERSES if universe in env_name),
        DEFAULT_UNIVERSE)
    return universe


def parse_domain_task(env_name, universe):
    env_name = env_name.replace(universe, '').strip('-')
    domains = DOMAINS_BY_UNIVERSE[universe]
    domain = next(domain for domain in domains if domain in env_name)

    env_name = env_name.replace(domain, '').strip('-')
    tasks = TASKS_BY_DOMAIN_BY_UNIVERSE[universe][domain]
    task = next((task for task in tasks if task == env_name), None)

    if task is None:
        matching_tasks = [task for task in tasks if task in env_name]
        if len(matching_tasks) > 1:
            raise ValueError(
                "Task name cannot be unmbiguously determined: {}."
                " Following task names match: {}"
                "".format(env_name, matching_tasks))
        elif len(matching_tasks) == 1:
            task = matching_tasks[-1]
        else:
            task = DEFAULT_TASK

    return domain, task


def parse_universe_domain_task(args):
    universe, domain, task = args.universe, args.domain, args.task

    if not universe:
        universe = parse_universe(args.env)

    if (not domain) or (not task):
        domain, task = parse_domain_task(args.env, universe)

    return universe, domain, task


def get_parser(allow_policy_list=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--universe',
                        type=str,
                        choices=UNIVERSES,
                        default=None)
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default=DEFAULT_TASK)
    parser.add_argument('--num-samples', type=int, default=1)

    parser.add_argument('--resources', type=json.loads, default=None,
                        help=("Resources to allocate to ray process. Passed"
                              " to `ray.init`."))
    parser.add_argument('--cpus', type=int, default=None,
                        help=("Cpus to allocate to ray process. Passed"
                              " to `ray.init`."))
    parser.add_argument('--gpus', type=int, default=None,
                        help=("Gpus to allocate to ray process. Passed"
                              " to `ray.init`."))

    parser.add_argument('--trial-resources', type=json.loads, default={},
                        help=("Resources to allocate for each trial. Passed"
                              " to `tune.run_experiments`."))
    parser.add_argument('--trial-cpus', type=int, default=None,
                        help=("Resources to allocate for each trial. Passed"
                              " to `tune.run_experiments`."))
    parser.add_argument('--trial-gpus', type=float, default=None,
                        help=("Resources to allocate for each trial. Passed"
                              " to `tune.run_experiments`."))

    if allow_policy_list:
        parser.add_argument('--policy',
                            type=str,
                            nargs='+',
                            choices=('gaussian', 'gmm', 'lsp', 'real-nvp'),
                            default='gaussian')
    else:
        parser.add_argument('--policy',
                            type=str,
                            choices=('gaussian', 'gmm', 'lsp', 'real-nvp'),
                            default='gaussian')
    parser.add_argument('--env', type=str, default='gym-swimmer-default')
    parser.add_argument('--exp_name',
                        type=str,
                        default=datetimestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument("--confirm_remote",
                        type=strtobool,
                        nargs='?',
                        const=True,
                        default=True,
                        help="Whether or not to query yes/no on remote run.")
    parser.add_argument('--log_extra_policy_info', type=strtobool, nargs='?',
                        const=True, default=False,
                        help=(
                            "Stores log pis and raw (unsquashed) actions in the"
                            "replay pool."
                        ))

    return parser


def variant_equals(*keys):
    def get_from_spec(spec):
        # TODO(hartikainen): This may break in some cases. ray.tune seems to
        # add a 'config' key at the top of the spec, whereas `generate_variants`
        # does not.
        node = spec.get('config', spec)
        for key in keys:
            node = node[key]

        return node

    return get_from_spec


def launch_experiments_rllab(variant_spec, args, run_fn):
    variants = [x[1] for x in generate_variants(variant_spec)]
    num_experiments = len(variants)

    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))

        run_params = variant.get('run_params', {})
        snapshot_mode = run_params.get(
            'snapshot_mode', variant.get('snapshot_mode', 'none'))
        snapshot_gap = run_params.get(
            'snapshot_gap', variant.get('snapshot_gap', 0))
        sync_pkl = run_params.get('sync_pkl', variant.get('sync_pkl'))
        sync_png = run_params.get('sync_png', variant.get('sync_png', True))
        sync_log = run_params.get('sync_log', variant.get('sync_log', True))
        seed = run_params.get('seed', variant.get('seed'))

        date_prefix = datestamp()
        experiment_prefix = os.path.join(
            variant.get('prefix', (
                '{}/{}/{}'.format(variant['universe'],
                                  variant['domain'],
                                  variant['task']))),
            '{}-{}'.format(date_prefix, args.exp_name))
        experiment_name = '{exp_name}-{i:0{max_i_len}}'.format(
            exp_name=args.exp_name,
            i=i,
            max_i_len=int(math.ceil(math.log10(num_experiments))))

        mode = args.mode.replace('rllab', '').strip('-')

        launch_experiment(
            run_fn,
            mode=mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=seed,
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            confirm_remote=args.confirm_remote,
            sync_s3_pkl=sync_pkl,
            sync_s3_png=sync_png,
            sync_s3_log=sync_log)


def _normalize_trial_resources(resources, cpu, gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    return resources


def launch_experiments_ray(variant_specs, args, local_dir, experiment_fn):
    import ray
    from ray import tune

    tune.register_trainable('mujoco-runner', experiment_fn)

    if 'local' in args.mode:
        ray.init(
            resources=args.resources,
            num_cpus=args.cpus,
            num_gpus=args.gpus,
        )
    else:
        ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')
        using_new_gcs = os.environ.get('RAY_USE_NEW_GCS', False) == 'on'
        using_xray = os.environ.get('RAY_USE_XRAY', False) == '1'
        if using_new_gcs and using_xray:
            policy = ray.experimental.SimpleGcsFlushPolicy()
            ray.experimental.set_flushing_policy(policy)

    trial_resources = _normalize_trial_resources(
        args.trial_resources, args.trial_cpus, args.trial_gpus)

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, args.exp_name))

    tune.run_experiments({
        "{}-{}".format(experiment_id, i): {
            'run': 'mujoco-runner',
            'trial_resources': trial_resources,
            'config': variant_spec,
            'local_dir': local_dir,
            'num_samples': args.num_samples,
            'upload_dir': 'gs://sac-ray-test/ray/results'
        }
        for i, variant_spec in enumerate(variant_specs)
    })


def launch_experiments_local(*args, **kwargs):
    """Temporary wrapper for local experiment launching.

    TODO(hartikainen): Reimplement this once we get rid of rllab.
    """
    return launch_experiments_rllab(*args, **kwargs)
