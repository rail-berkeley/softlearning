import multiprocessing
import argparse
from distutils.util import strtobool
import json

import softlearning.algorithms.utils as alg_utils
import softlearning.environments.utils as env_utils
from softlearning.misc.utils import datetimestamp
from .instrument import (
    run_experiments_dry,
    run_experiments_local,
    run_experiments_debug,
    launch_experiments_gce,
    launch_experiments_ec2,
    run_experiments_cluster)

DEFAULT_UNIVERSE = 'gym'
DEFAULT_DOMAIN = 'Swimmer'
DEFAULT_TASK = 'Default'
DEFAULT_ALGORITHM = 'SAC'

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

AVAILABLE_ALGORITHMS = set(alg_utils.ALGORITHM_CLASSES.keys())


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


def add_ray_init_args(parser):

    def init_help_string(help_string):
        return help_string + " Passed to `ray.init`."

    parser.add_argument(
        '--cpus',
        type=int,
        default=None,
        help=init_help_string("Cpus to allocate to ray process."))
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help=init_help_string("Gpus to allocate to ray process."))
    parser.add_argument(
        '--resources',
        type=json.loads,
        default=None,
        help=init_help_string("Resources to allocate to ray process."))
    parser.add_argument(
        '--include-webui',
        type=str,
        default=True,
        help=init_help_string("Boolean flag indicating whether to start the"
                              "web UI, which is a Jupyter notebook."))
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help=init_help_string("If provided, it will specify the root temporary"
                              " directory for the Ray process."))

    return parser


def add_ray_tune_args(parser):

    def tune_help_string(help_string):
        return help_string + " Passed to `tune.run_experiments`."

    parser.add_argument(
        '--resources-per-trial',
        type=json.loads,
        default={},
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--trial-gpus',
        type=float,
        default=None,
        help=("Resources to allocate for each trial. Passed"
              " to `tune.run_experiments`."))
    parser.add_argument(
        '--trial-extra-cpus',
        type=int,
        default=None,
        help=("Extra CPUs to reserve in case the trials need to"
              " launch additional Ray actors that use CPUs."))
    parser.add_argument(
        '--trial-extra-gpus',
        type=float,
        default=None,
        help=("Extra GPUs to reserve in case the trials need to"
              " launch additional Ray actors that use GPUs."))
    parser.add_argument(
        '--num-samples',
        default=1,
        type=int,
        help=tune_help_string("Number of times to repeat each trial."))
    parser.add_argument(
        '--upload-dir',
        type=str,
        default='',
        help=tune_help_string("Optional URI to sync training results to (e.g."
                              " s3://<bucket> or gs://<bucket>)."))
    parser.add_argument(
        '--trial-name-creator',
        default=None,
        help=tune_help_string(
            "Optional creator function for the trial string, used in "
            "generating a trial directory."))
    parser.add_argument(
        '--trial-cpus',
        type=int,
        default=multiprocessing.cpu_count(),
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=None,
        help=tune_help_string(
            "How many training iterations between checkpoints."
            " A value of 0 (default) disables checkpointing. If set,"
            " takes precedence over variant['run_params']"
            "['checkpoint_frequency']."))
    parser.add_argument(
        '--checkpoint-at-end',
        type=lambda x: bool(strtobool(x)),
        default=None,
        help=tune_help_string(
            "Whether to checkpoint at the end of the experiment. If set,"
            " takes precedence over variant['run_params']"
            "['checkpoint_at_end']."))
    parser.add_argument(
        '--max-failures',
        default=3,
        type=int,
        help=tune_help_string(
            "Try to recover a trial from its last checkpoint at least this "
            "many times. Only applies if checkpointing is enabled."))
    parser.add_argument(
        '--restore',
        type=str,
        default=None,
        help=tune_help_string(
            "Path to checkpoint. Only makes sense to set if running 1 trial."
            " Defaults to None."))
    parser.add_argument(
        '--with-server',
        type=str,
        default=False,
        help=tune_help_string("Starts a background Tune server. Needed for"
                              " using the Client API."))

    return parser


# class AutoscalerConfigFileType(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         if values is None and namespace.mode in ('gce', 'ec2'):
#             values = DEFAULT_AUTOSCALER_CONFIG_PATHS[namespace.mode]

#         from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
#         setattr(namespace, self.dest, values)


def add_ray_autoscaler_exec_args(parser):
    parser.add_argument(
        '--autoscaler-config-file',
        type=str)
    parser.add_argument(
        '--autoscaler-tmux',
        type=lambda x: bool(strtobool(x)),
        default=True)
    parser.add_argument(
        '--autoscaler-screen',
        type=lambda x: bool(strtobool(x)),
        default=False)
    parser.add_argument(
        '--autoscaler-start',
        type=lambda x: bool(strtobool(x)),
        default=True)
    parser.add_argument(
        '--autoscaler-stop',
        type=lambda x: bool(strtobool(x)),
        default=True)
    parser.add_argument(
        '--autoscaler-override-cluster-name',
        type=str,
        default=None)
    parser.add_argument(
        '--autoscaler-port-forward',
        type=int,
        default=None)

    return parser


def get_parser(allow_policy_list=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--universe', type=str, choices=UNIVERSES, default=None)
    parser.add_argument(
        '--domain', type=str, choices=AVAILABLE_DOMAINS, default=None)
    parser.add_argument(
        '--task', type=str, choices=AVAILABLE_TASKS, default=DEFAULT_TASK)

    parser.add_argument(
        '--checkpoint-replay-pool',
        type=lambda x: bool(strtobool(x)),
        default=None,
        help=("Whether a checkpoint should also saved the replay"
              " pool. If set, takes precedence over"
              " variant['run_params']['checkpoint_replay_pool']."
              " Note that the replay pool is saved (and "
              " constructed) piece by piece so that each"
              " experience is saved only once."))

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=AVAILABLE_ALGORITHMS,
        default=DEFAULT_ALGORITHM)
    if allow_policy_list:
        parser.add_argument(
            '--policy',
            type=str,
            nargs='+',
            choices=('gaussian', ),
            default='gaussian')
    else:
        parser.add_argument(
            '--policy',
            type=str,
            choices=('gaussian', ),
            default='gaussian')

    parser.add_argument(
        '--exp-name',
        type=str,
        default=datetimestamp())
    parser.add_argument(
        '--mode', type=str, default='local')
    parser.add_argument(
        '--confirm-remote',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help="Whether or not to query yes/no on remote run.")

    parser = add_ray_init_args(parser)
    parser = add_ray_tune_args(parser)
    parser = add_ray_autoscaler_exec_args(parser)

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


def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_freq': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    variant_spec['restore'] = command_line_args.restore

    return variant_spec


def add_command_line_args_to_variant_specs(variant_specs, command_line_args):
    variant_specs = [
        add_command_line_args_to_variant_spec(variant_spec, command_line_args)
        for variant_spec in variant_specs
    ]

    return variant_specs


def launch_experiments_ray(variant_specs,
                           command_line_args,
                           local_dir,
                           experiment_fn,
                           *args,
                           **kwargs):
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, command_line_args.exp_name))

    variant_specs = add_command_line_args_to_variant_specs(
        variant_specs, command_line_args)

    experiments = {
        "{}-{}".format(experiment_id, i): {
            'run': experiment_fn,
            'resources_per_trial': resources_per_trial,
            'config': variant_spec,
            'local_dir': local_dir,
            'num_samples': command_line_args.num_samples,
            'upload_dir': command_line_args.upload_dir,
            'checkpoint_freq': (
                variant_spec['run_params']['checkpoint_frequency']),
            'checkpoint_at_end': (
                variant_spec['run_params']['checkpoint_at_end']),
            'restore': command_line_args.restore,  # Defaults to None
        }
        for i, variant_spec in enumerate(variant_specs)
    }

    if command_line_args.mode == 'dry':
        run_experiments_dry(experiments, command_line_args, *args, **kwargs)
    elif command_line_args.mode == 'debug':
        run_experiments_debug(experiments, command_line_args, *args, **kwargs)
    elif command_line_args.mode == 'local':
        run_experiments_local(experiments, command_line_args, *args, **kwargs)
    elif command_line_args.mode == 'gce':
        launch_experiments_gce(experiments, command_line_args, *args, **kwargs)
    elif command_line_args.mode == 'ec2':
        launch_experiments_ec2(experiments, command_line_args, *args, **kwargs)
    elif command_line_args.mode == 'cluster':
        run_experiments_cluster(experiments, command_line_args, *args, **kwargs)
    else:
        raise ValueError(command_line_args.mode)
