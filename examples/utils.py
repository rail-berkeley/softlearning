import os
import math
import argparse
from distutils.util import strtobool

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


def get_parser():
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
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian', 'gmm', 'lsp'),
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


DEFAULT_SNAPSHOT_MODE = 'none'
DEFAULT_SNAPSHOT_GAP = 1000


def setup_rllab_logger(variant):
    """Temporary setup for rllab logger previously handled by run_experiment.

    TODO.hartikainen: Remove this once we have gotten rid of rllab logger.
    """

    from rllab.misc import logger

    run_params = variant['run_params']

    ray_log_dir = os.getcwd()
    log_dir = os.path.join(ray_log_dir, 'rllab-logger')

    tabular_log_file = os.path.join(log_dir, 'progress.csv')
    text_log_file = os.path.join(log_dir, 'debug.log')
    variant_log_file = os.path.join(log_dir, 'variant.json')

    logger.log_variant(variant_log_file, variant)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(
        run_params.get('snapshot_mode', DEFAULT_SNAPSHOT_MODE))
    logger.set_snapshot_gap(
        run_params.get('snapshot_gap', DEFAULT_SNAPSHOT_GAP))
    logger.set_log_tabular_only(False)

    # TODO.hartikainen: need to remove something, or push_prefix, pop_prefix?
    # logger.push_prefix("[%s] " % args.exp_name)


def launch_experiments_rllab(variants, args, run_fn):
    num_experiments = len(variants)

    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))

        run_params = variant.get('run_params', {})
        snapshot_mode = run_params.get(
            'snapshot_mode', variant.get('snapshot_mode'))
        snapshot_gap = run_params.get(
            'snapshot_gap', variant.get('snapshot_gap'))
        sync_pkl = run_params.get('sync_pkl', variant.get('sync_pkl'))
        seed = run_params.get('seed', variant.get('seed'))

        date_prefix = datestamp()
        experiment_prefix = os.path.join(
            variant['prefix'],
            '{}-{}'.format(date_prefix, args.exp_name))
        experiment_name = '{exp_name}-{i:0{max_i_len}}'.format(
            exp_name=args.exp_name,
            i=i,
            max_i_len=int(math.ceil(math.log10(num_experiments))))

        launch_experiment(
            run_fn,
            mode=args.mode,
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
            sync_s3_pkl=sync_pkl)
