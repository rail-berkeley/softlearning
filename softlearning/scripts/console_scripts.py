"""A command line interface that exposes softlearning examples to user.

This package exposes the modules in examples package to the user through a cli,
which allows seamless runs of examples in different modes (e.g. locally, in
google compute engine, or ec2).


There are two types of cli commands in this file:
1. run_example_* methods, which run the experiments by invoking
    `tune.run_experiments` function.
2. launch_example_* methods, which are helpers function to submit an
    example to be run in the cloud. In practice, these launch a cluster,
    and then run the `run_example_cluster` method with the provided
    arguments and options.

TODO(hartikainen): consider moving some of the functionality in examples
module, as that seems a bit more logical place for these functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from pprint import pformat
import logging

import click
import ray
from ray import tune
from ray.scripts.scripts import exec_cmd

from examples.utils import (
    get_parser as get_example_parser,
    generate_experiment)

from examples.instrument import (
    unique_cluster_name,
    get_experiments_info,
    AUTOSCALER_DEFAULT_CONFIG_FILE_GCE,
    AUTOSCALER_DEFAULT_CONFIG_FILE_EC2)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command(
    name='run_example_dry',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_dry(example_module_name, example_argv):
    """Print the variant spec and related information of an example."""
    example_args = get_example_parser().parse_args(example_argv)
    example_module = importlib.import_module(example_module_name)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)

    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)

    experiments = {experiment_id: experiment}

    experiments_info = get_experiments_info(experiments)
    number_of_trials = experiments_info["number_of_trials"]
    total_number_of_trials = experiments_info["total_number_of_trials"]

    experiments_info_text = f"""
Dry run.

Experiment specs:
{pformat(experiments, indent=2)}

Number of trials:
{pformat(number_of_trials, indent=2)}

Number of total trials (including samples/seeds): {total_number_of_trials}
"""

    print(experiments_info_text)


@cli.command(
    name='run_example_local',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_local(example_module_name, example_argv):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    example_args = get_example_parser().parse_args(example_argv)
    example_module = importlib.import_module(example_module_name)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)

    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)
    experiments = {experiment_id: experiment}

    ray.init(
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        resources=example_args.resources or {},
        # Tune doesn't currently support local mode
        local_mode=False,
        include_webui=example_args.include_webui,
        temp_dir=example_args.temp_dir)

    tune.run_experiments(
        experiments,
        with_server=example_args.with_server,
        server_port=4321,
        scheduler=None)


@cli.command(
    name='run_example_debug',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_debug(example_module_name, example_argv):
    """The debug mode limits tune trial runs to enable use of debugger.

    TODO(hartikainen): The debug mode should allow easy switch between
    parallelized andnon-parallelized runs such that the debugger can be
    reasonably used when running the code. This could be implemented for
    example by requiring a custom resource (e.g. 'debug-resource') that
    limits the number of parallel runs to one. For this to work, tune needs to
    merge the support for custom resources:
    https://github.com/ray-project/ray/pull/2979. Alternatively, this could be
    implemented using the 'local_mode' argument for ray.init(), once tune
    supports it.
    """
    raise NotImplementedError


@cli.command(
    name='run_example_cluster',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_cluster(example_module_name, example_argv):
    """Run example on cluster mode.

    This functions is very similar to the local mode, except that it
    correctly sets the redis address to make ray/tune work on a cluster.
    """
    example_args = get_example_parser().parse_args(example_argv)
    example_module = importlib.import_module(example_module_name)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)

    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)
    experiments = {experiment_id: experiment}

    resources = example_args.resources or {}
    redis_address = ray.services.get_node_ip_address() + ':6379'

    ray.init(
        redis_address=redis_address,
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        resources=resources,
        # Tune doesn't currently support local mode
        local_mode=False,
        include_webui=example_args.include_webui,
        temp_dir=example_args.temp_dir)

    tune.run_experiments(
        experiments,
        with_server=example_args.with_server,
        server_port=4321,
        scheduler=None)


def confirm_yes_no(prompt):
    # raw_input returns the empty string for "enter"
    yes = {'yes', 'y', 'ye'}
    no = {'no', 'n'}

    choice = input(prompt).lower()
    while True:
        if choice in yes:
            return True
        elif choice in no:
            exit(0)
        else:
            print("Please respond with 'yes' or 'no'")
        choice = input().lower()


def add_options(options):
    def decorator(f):
        for option in options[::-1]:
            click.decorators._param_memo(f, option)
        return f
    return decorator


@cli.command(
    name='launch_example_cluster',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True
    })
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--cluster_config_file",
    required=False,
    type=str)
@click.option(
    "--stop",
    is_flag=True,
    default=True,
    help="Stop the cluster after the command finishes running.")
@click.option(
    "--start",
    is_flag=True,
    default=True,
    help="Start the cluster if needed.")
@click.option(
    "--screen",
    is_flag=True,
    default=False,
    help="Run the command in a screen.")
@click.option(
    "--tmux",
    is_flag=True,
    default=True,
    help="Run the command in tmux.")
@click.option(
    "--cluster-name",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--port-forward", required=False, type=int, help="Port to forward.")
@click.pass_context
def launch_example_cluster(ctx,
                           example_module_name,
                           example_argv,
                           cluster_config_file,
                           screen,
                           tmux,
                           stop,
                           start,
                           cluster_name,
                           port_forward):
    """Launches the example on autoscaled ray cluster through ray exec_cmd.

    This handles basic validation and sanity checks for the experiment, and
    then executes the command on autoscaled ray cluster. If necessary, it will
    also fill in more useful defaults for our workflow (i.e. for tmux and
    cluster_name).
    """
    example_args = get_example_parser().parse_args(example_argv)
    example_module = importlib.import_module(example_module_name)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)

    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)
    experiments = {experiment_id: experiment}

    experiments_info = get_experiments_info(experiments)
    total_number_of_trials = experiments_info['total_number_of_trials']

    if not example_args.upload_dir:
        confirm_yes_no(
            "`upload_dir` is empty. No results will be uploaded to cloud"
            " storage. Use `--upload-dir` argument to set upload dir."
            " Continue without upload directory?\n(yes/no) ")

    confirm_yes_no(f"Launch {total_number_of_trials} trials?\n(yes/no) ")

    cluster_name = cluster_name or unique_cluster_name(example_args)

    cluster_command_parts = (
        'softlearning',
        'run_example_cluster',
        *example_argv)
    cluster_command = ' '.join(cluster_command_parts)

    ctx.invoke(
        exec_cmd,
        cluster_config_file=cluster_config_file,
        cmd=cluster_command,
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=start,
        cluster_name=cluster_name,
        port_forward=port_forward)


@cli.command(
    name='launch_example_gce',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True
    })
@add_options(launch_example_cluster.params)
@click.pass_context
def launch_example_gce(ctx, *args, cluster_config_file, **kwargs):
    """Forwards call to `launch_example_cluster` after adding gce defaults.

    This optionally sets the ray autoscaler configuration file to the default
    gce configuration file, and then calls `launch_example_cluster` to
    execute the original command on autoscaled gce cluster by parsing the args.

    See `launch_example_cluster` for further details.
    """
    cluster_config_file = (
        cluster_config_file or AUTOSCALER_DEFAULT_CONFIG_FILE_GCE)

    ctx.invoke(
        launch_example_cluster,
        **kwargs,
        cluster_config_file=cluster_config_file)


@cli.command(
    name='launch_example_ec2',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True
    })
@add_options(launch_example_cluster.params)
@click.pass_context
def launch_example_ec2(ctx, *args, cluster_config_file, **kwargs):
    """Forwards call to `launch_example_cluster` after adding ec2 defaults.

    This optionally sets the ray autoscaler configuration file to the default
    ec2 configuration file, and then calls `launch_example_cluster` to
    execute the original command on autoscaled ec2 cluster by parsing the args.

    See `launch_example_cluster` for further details.
    """
    cluster_config_file = (
        cluster_config_file or AUTOSCALER_DEFAULT_CONFIG_FILE_EC2)

    ctx.invoke(
        launch_example_cluster,
        **kwargs,
        cluster_config_file=cluster_config_file)


cli.add_command(run_example_local)
cli.add_command(run_example_dry)
cli.add_command(run_example_cluster)

cli.add_command(launch_example_cluster)
cli.add_command(launch_example_gce)
cli.add_command(launch_example_ec2)


def main():
    return cli()


if __name__ == "__main__":
    main()
