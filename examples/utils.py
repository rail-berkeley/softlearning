import multiprocessing
import argparse
from distutils.util import strtobool
import json

import softlearning.environments.utils as env_utils
from softlearning.misc.utils import datetimestamp


DEFAULT_UNIVERSE = 'gym'
DEFAULT_TASK = 'Default'

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

    parser.add_argument('--resources-per-trial', type=json.loads, default={},
                        help=("Resources to allocate for each trial. Passed"
                              " to `tune.run_experiments`."))
    parser.add_argument('--trial-cpus',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help=("Resources to allocate for each trial. Passed"
                              " to `tune.run_experiments`."))
    parser.add_argument('--trial-gpus', type=float, default=None,
                        help=("Resources to allocate for each trial. Passed"
                              " to `tune.run_experiments`."))
    parser.add_argument('--trial-extra-cpus', type=int, default=None,
                        help=(
                            "Extra CPUs to reserve in case the trials need to"
                            " launch additional Ray actors that use CPUs."))
    parser.add_argument('--trial-extra-gpus', type=float, default=None,
                        help=(
                            "Extra GPUs to reserve in case the trials need to"
                            " launch additional Ray actors that use GPUs."))

    parser.add_argument('--checkpoint-frequency',
                        type=int,
                        default=None,
                        help=(
                            "Save the training checkpoint every this many"
                            " epochs. If set, takes precedence over"
                            " variant['run_params']['checkpoint_frequency']."))
    parser.add_argument('--checkpoint-at-end',
                        type=lambda x: bool(strtobool(x)),
                        default=None,
                        help=(
                            "Whether a checkpoint should be saved at the end"
                            " of training. If set, takes precedence over"
                            " variant['run_params']['checkpoint_at_end']."))
    parser.add_argument('--checkpoint-replay-pool',
                        type=lambda x: bool(strtobool(x)),
                        default=None,
                        help=(
                            "Whether a checkpoint should also saved the replay"
                            " pool. If set, takes precedence over"
                            " variant['run_params']['checkpoint_replay_pool']."
                            " Note that the replay pool is saved (and "
                            " constructed) piece by piece so that each"
                            " experience is saved only once."))
    parser.add_argument('--restore',
                        type=str,
                        default=None,
                        help=(
                            "Path to checkpoint. Only makes sense to set if"
                            " running 1 trial. Defaults to None."))

    if allow_policy_list:
        parser.add_argument('--policy',
                            type=str,
                            nargs='+',
                            choices=('gaussian', ),
                            default='gaussian')
    else:
        parser.add_argument('--policy',
                            type=str,
                            choices=('gaussian', ),
                            default='gaussian')
    parser.add_argument('--env', type=str, default='gym-swimmer-default')
    parser.add_argument('--exp-name',
                        type=str,
                        default=datetimestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--upload-dir', type=str, default='',
                        help=("Optional URI to sync training results to (e.g."
                              " s3://<bucket> or gs://<bucket>)."))

    parser.add_argument("--confirm-remote",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Whether or not to query yes/no on remote run.")

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


def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu

    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu

    return resources


def launch_experiments_ray(variant_specs,
                           args,
                           local_dir,
                           experiment_fn,
                           scheduler=None):
    import ray
    from ray import tune

    tune.register_trainable('mujoco-runner', experiment_fn)

    resources_per_trial = _normalize_trial_resources(
        args.resources_per_trial,
        args.trial_cpus,
        args.trial_gpus,
        args.trial_extra_cpus,
        args.trial_extra_gpus)

    if 'local' in args.mode or 'debug' in args.mode:
        resources = args.resources or {}

        if 'debug' in args.mode:
            # Require a debug resource for each trial, so that we never run
            # more than one trial at a time. This makes debugging easier, since
            # the debugger stdout behaves more reasonably with single process.
            # TODO(hartikainen): Change this from 'extra_gpu' to
            # 'debug-resource' once tune supports custom resources.
            # See: https://github.com/ray-project/ray/pull/2979.
            resources['extra_gpu'] = 1
            resources_per_trial['extra_gpu'] = 1

        ray.init(
            resources=resources,
            num_cpus=args.cpus,
            num_gpus=args.gpus)
    else:
        ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, args.exp_name))

    tune.run_experiments(
        {
            "{}-{}".format(experiment_id, i): {
                'run': 'mujoco-runner',
                'resources_per_trial': resources_per_trial,
                'config': variant_spec,
                'local_dir': local_dir,
                'num_samples': args.num_samples,
                'upload_dir': args.upload_dir,
                'checkpoint_freq': (
                    args.checkpoint_frequency
                    if args.checkpoint_frequency is not None
                    else variant_spec['run_params'].get('checkpoint_frequency', 0)
                ),
                'checkpoint_at_end': (
                    args.checkpoint_at_end
                    if args.checkpoint_at_end is not None
                    else variant_spec['run_params'].get('checkpoint_at_end', True)
                ),
                'restore': args.restore,  # Defaults to None
            }
            for i, variant_spec in enumerate(variant_specs)
        },
        scheduler=scheduler,
    )
