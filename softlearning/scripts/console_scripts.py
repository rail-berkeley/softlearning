"""A command line interface that exposes softlearning examples to user.

This package exposes the functions in examples.instrument module to the user
through a cli, which allows seamless runs of examples in different modes (e.g.
locally, in google compute engine, or ec2).


There are two types of cli commands in this file (each have their corresponding
function in examples.instrument):
1. run_example_* methods, which run the experiments by invoking `tune.run`
    function.
2. launch_example_* methods, which are helpers function to submit an
    example to be run in the cloud. In practice, these launch a cluster,
    and then run the `run_example_cluster` method with the provided
    arguments and options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import click

from examples.instrument import (
    run_example_dry,
    run_example_local,
    run_example_debug,
    run_example_cluster,
    launch_example_cluster,
    launch_example_gce,
    launch_example_ec2)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_options(options):
    def decorator(f):
        for option in options[::-1]:
            click.decorators._param_memo(f, option)
        return f
    return decorator


@click.group()
def cli():
    pass


@cli.command(
    name='run_example_dry',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_dry_cmd(example_module_name, example_argv):
    """Print the variant spec and related information of an example."""
    example_argv = (*example_argv, '--mode=dry')
    return run_example_dry(example_module_name, example_argv)


@cli.command(
    name='run_example_local',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_local_cmd(example_module_name, example_argv):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    example_argv = (*example_argv, '--mode=local')
    return run_example_local(example_module_name, example_argv)


@cli.command(
    name='run_example_debug',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_debug_cmd(example_module_name, example_argv):
    """The debug mode limits tune trial runs to enable use of debugger."""
    example_argv = (*example_argv, '--mode=debug')
    return run_example_debug(example_module_name, example_argv)


@cli.command(
    name='run_example_cluster',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_cluster_cmd(example_module_name, example_argv):
    """Run example on cluster mode.

    This functions is very similar to the local mode, except that it
    correctly sets the ray address to make ray/tune work on a cluster.
    """
    run_example_cluster(example_module_name, example_argv)


@cli.command(
    name='launch_example_cluster',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True
    })
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--config_file",
    required=False,
    type=str)
@click.option(
    "--stop/--no-stop",
    is_flag=True,
    default=True,
    help="Stop the cluster after the command finishes running.")
@click.option(
    "--start/--no-start",
    is_flag=True,
    default=True,
    help="Start the cluster if needed.")
@click.option(
    "--screen/--no-screen",
    is_flag=True,
    default=False,
    help="Run the command in a screen.")
@click.option(
    "--tmux/--no-tmux",
    is_flag=True,
    default=True,
    help="Run the command in tmux.")
@click.option(
    "--override-cluster-name",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--port-forward", required=False, type=int, help="Port to forward.")
def launch_example_cluster_cmd(*args, **kwargs):
    """Launches the example on autoscaled ray cluster through ray exec_cmd.

    This handles basic validation and sanity checks for the experiment, and
    then executes the command on autoscaled ray cluster. If necessary, it will
    also fill in more useful defaults for our workflow (i.e. for tmux and
    override_cluster_name).
    """
    return launch_example_cluster(*args, **kwargs)


@cli.command(
    name='launch_example_gce',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True
    })
@add_options(launch_example_cluster_cmd.params)
def launch_example_gce_cmd(*args, example_argv=(), **kwargs):
    """Forwards call to `launch_example_cluster` after adding gce defaults.

    This optionally sets the ray autoscaler configuration file to the default
    gce configuration file, and then calls `launch_example_cluster` to
    execute the original command on autoscaled gce cluster by parsing the args.

    See `launch_example_cluster` for further details.
    """
    example_argv = (*example_argv, '--mode=gce')
    return launch_example_gce(*args, example_argv=example_argv, **kwargs)


@cli.command(
    name='launch_example_ec2',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True
    })
@add_options(launch_example_cluster_cmd.params)
def launch_example_ec2_cmd(*args, example_argv=(), **kwargs):
    """Forwards call to `launch_example_cluster` after adding ec2 defaults.

    This optionally sets the ray autoscaler configuration file to the default
    ec2 configuration file, and then calls `launch_example_cluster` to
    execute the original command on autoscaled ec2 cluster by parsing the args.

    See `launch_example_cluster` for further details.
    """
    example_argv = (*example_argv, '--mode=gce')
    return launch_example_ec2(*args, example_argv=example_argv, **kwargs)


cli.add_command(run_example_local_cmd)
cli.add_command(run_example_dry_cmd)
cli.add_command(run_example_cluster_cmd)

# Alias for run_example_local
cli.add_command(run_example_local_cmd, name='launch_example_local')
# Alias for run_example_dry
cli.add_command(run_example_dry_cmd, name='launch_example_dry')
# Alias for run_example_debug
cli.add_command(run_example_debug_cmd, name='launch_example_debug')
cli.add_command(launch_example_cluster_cmd)
cli.add_command(launch_example_gce_cmd)
cli.add_command(launch_example_ec2_cmd)


def main():
    return cli()


if __name__ == "__main__":
    main()
