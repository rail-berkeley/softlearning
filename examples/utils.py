import multiprocessing
import argparse
from distutils.util import strtobool
import json

from ray.tune import sample_from

import softlearning.environments.utils as env_utils
from softlearning.utils.times import datetimestamp


DEFAULT_UNIVERSE = 'gym'
DEFAULT_DOMAIN = 'Pendulum'
DEFAULT_TASK = 'v0'

AVAILABLE_UNIVERSES = tuple(env_utils.UNIVERSES)


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
        '--include-dashboard',
        type=str,
        default=False,
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
        return help_string + " Passed to `tune.run`."

    parser.add_argument(
        '--resources-per-trial',
        type=json.loads,
        default={},
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--trial-cpus',
        type=int,
        default=multiprocessing.cpu_count(),
        help=tune_help_string(
            "CPUs to allocate for each trial. Note: this is only used for"
            " Ray's internal scheduling bookkeeping, and is not an actual hard"
            " limit for CPUs."))
    parser.add_argument(
        '--trial-gpus',
        type=float,
        default=None,
        help=tune_help_string(
            "GPUs to allocate for each trial. Note: this is only used for"
            " Ray's internal scheduling bookkeeping, and is not an actual hard"
            " limit for GPUs."))
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
        '--trial-name-template',
        type=str,
        default='id={trial.trial_id}-seed={trial.config[run_params][seed]}',
        help=tune_help_string(
            "Optional string template for trial name. For example:"
            " '{trial.trial_id}-seed={trial.config[run_params][seed]}'"))
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
        '--fail-fast',
        type=lambda x: bool(strtobool(x)),
        default=False,
        help=tune_help_string("Finishes as soon as a trial fails if True."))
    parser.add_argument(
        '--server-port',
        type=lambda value: int(value) if value else None,
        default=None,
        help=tune_help_string("Port number for launching TuneServer."))

    return parser


def get_parser(allow_policy_list=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--universe',
        type=str,
        choices=AVAILABLE_UNIVERSES,
        default=DEFAULT_UNIVERSE)
    parser.add_argument(
        '--domain',
        type=str,
        default=DEFAULT_DOMAIN)
    parser.add_argument(
        '--task', type=str, default=DEFAULT_TASK)

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

    parser.add_argument('--algorithm', type=str)
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
        '--run-eagerly',
        type=lambda x: bool(strtobool(x)),
        help="Whether to run tensorflow in eager mode.")
    parser.add_argument(
        '--local-dir',
        type=str,
        default='~/ray_results',
        help='Destination local folder to save training results.')

    parser.add_argument(
        '--confirm-remote',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help="Whether or not to query yes/no on remote run.")

    parser.add_argument(
        '--video-save-frequency',
        type=int,
        default=None,
        help="Save frequency for videos.")

    parser = add_ray_init_args(parser)
    parser = add_ray_tune_args(parser)

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

    return sample_from(get_from_spec)
