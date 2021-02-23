"""Functions for instrumenting and running softlearning examples.

This package contains functions, which allow seamless runs of examples in
different modes (e.g. locally, in google compute engine, or ec2).


There are two types of functions in this file:
1. run_example_* methods, which run the experiments by invoking
    `tune.run` function.
2. launch_example_* methods, which are helpers function to submit an
    example to be run in the cloud. In practice, these launch a cluster,
    and then run the `run_example_cluster` method with the provided
    arguments and options.
"""

import copy
import importlib
import multiprocessing
import os
import uuid
from pprint import pformat

import ray
from ray import tune
from ray.autoscaler._private.commands import exec_cluster

from softlearning.utils.times import datetimestamp
from softlearning.utils.misc import PROJECT_PATH


AUTOSCALER_DEFAULT_CONFIG_FILE_GCE = os.path.join(
    PROJECT_PATH, 'config', 'ray-autoscaler-gce.yaml')
AUTOSCALER_DEFAULT_CONFIG_FILE_EC2 = os.path.join(
    PROJECT_PATH, 'config', 'ray-autoscaler-ec2.yaml')


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
        'checkpoint_frequency': (
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

    if (command_line_args.mode == 'debug'
        and ('run_eagerly' not in command_line_args
             or command_line_args.run_eagerly is None)):
        variant_spec['run_params']['run_eagerly'] = True
    elif 'run_eagerly' in command_line_args:
        variant_spec['run_params']['run_eagerly'] = (
            command_line_args.run_eagerly)

    variant_spec['restore'] = command_line_args.restore

    return variant_spec


def generate_experiment_kwargs(variant_spec, command_line_args):
    local_dir = command_line_args.local_dir
    if command_line_args.mode == 'debug':
        local_dir = os.path.join(local_dir, 'debug')
    local_dir = os.path.join(
        local_dir,
        command_line_args.universe,
        command_line_args.domain,
        command_line_args.task)
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)
    upload_dir = (
        os.path.join(
            command_line_args.upload_dir,
            command_line_args.universe,
            command_line_args.domain,
            command_line_args.task)
        if command_line_args.upload_dir
        else None)

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, command_line_args.exp_name))

    variant_spec = add_command_line_args_to_variant_spec(
        variant_spec, command_line_args)

    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['config']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    def create_trial_name_creator(trial_name_template=None):
        if not trial_name_template:
            return None

        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)

        return trial_name_creator

    experiment_kwargs = {
        'name': experiment_id,
        'resources_per_trial': resources_per_trial,
        'config': variant_spec,
        'local_dir': local_dir,
        'num_samples': command_line_args.num_samples,
        'upload_dir': upload_dir,
        'checkpoint_freq': (
            variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end': (
            variant_spec['run_params']['checkpoint_at_end']),
        'max_failures': command_line_args.max_failures,
        'trial_name_creator': create_trial_name_creator(
            command_line_args.trial_name_template),
        'restore': command_line_args.restore,  # Defaults to None
    }

    return experiment_kwargs


def unique_cluster_name(args):
    cluster_name_parts = (
        datetimestamp(''),
        str(uuid.uuid4())[:6],
        args.domain,
        args.task
    )
    cluster_name = "-".join(cluster_name_parts).lower()
    return cluster_name


def get_experiments_info(experiments):
    number_of_trials = {
        experiment_kwargs['name']: len(list(
            tune.suggest.variant_generator.generate_variants(
                experiment_kwargs['config'])
        )) * experiment_kwargs['num_samples']
        for experiment_kwargs in experiments
    }
    total_number_of_trials = sum(number_of_trials.values())

    experiments_info = {
        "number_of_trials": number_of_trials,
        "total_number_of_trials": total_number_of_trials,
    }

    return experiments_info


def confirm_yes_no(prompt):
    # raw_input returns the empty string for "enter"
    yes = {'yes', 'ye', 'y'}
    no = {'no', 'n'}

    choice = input(prompt).lower()
    while True:
        if choice in yes:
            return True
        elif choice in no:
            exit(0)
        else:
            print("Please respond with 'yes' or 'no'.\n(yes/no)")
        choice = input().lower()


def run_example_dry(example_module_name, example_argv):
    """Print the variant spec and related information of an example."""
    example_module = importlib.import_module(example_module_name)

    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)

    experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

    experiments_info = get_experiments_info([experiment_kwargs])
    number_of_trials = experiments_info["number_of_trials"]
    total_number_of_trials = experiments_info["total_number_of_trials"]

    experiments_info_text = f"""
Dry run.

Experiment specs:
{pformat(experiment_kwargs, indent=2)}

Number of trials:
{pformat(number_of_trials, indent=2)}

Number of total trials (including samples/seeds): {total_number_of_trials}
"""

    print(experiments_info_text)


def run_example_local(example_module_name, example_argv, local_mode=False):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    example_module = importlib.import_module(example_module_name)

    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)

    experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

    ray.init(
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        resources=example_args.resources or {},
        local_mode=local_mode,
        include_dashboard=example_args.include_dashboard,
        _temp_dir=example_args.temp_dir)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        server_port=example_args.server_port,
        fail_fast=example_args.fail_fast,
        scheduler=None,
        reuse_actors=True)


def run_example_debug(example_module_name, example_argv):
    """The debug mode sets runs up in order to enable use of debugger.

    The debug mode should allow easy switch from parallelized to
    non-parallelized runs such that the debugger can be reasonably used when
    running the code. In practice, we default to running tensorflow in eager
    mode (i.e. `tf.config.experimental_run_functions_eagerly(True)`) and
    set initialize ray with `local_mode=True`.

    TODO(hartikainen): This probably doesn't need to allocate any resources
    anymore. If it does, it should allocate a custom "debug_resource" instead
    of all cpus once ray local mode supports custom resources.
    """

    example_module = importlib.import_module(example_module_name)
    example_args = example_module.get_parser().parse_args(example_argv)

    debug_args = copy.copy(example_args)

    if 'trial_cpus' in debug_args:
        available_cpus = multiprocessing.cpu_count()
        debug_args.trial_cpus = available_cpus

    if 'server_port' in debug_args:
        print(f"Ignoring 'server_port' due to debug mode.")
    debug_args.server_port = None

    if 'max_failures' in debug_args:
        print(f"Ignoring 'max_failures' due to debug mode.")
    debug_args.max_failures = 0

    if 'upload_dir' in debug_args:
        print(f"Ignoring 'upload_dir' due to debug mode.")
    debug_args.upload_dir = None

    debug_args.fail_fast = True


    variant_spec = example_module.get_variant_spec(debug_args)
    trainable_class = example_module.get_trainable_class(debug_args)

    experiment_kwargs = generate_experiment_kwargs(variant_spec, debug_args)

    ray.init(
        num_cpus=debug_args.cpus,
        num_gpus=debug_args.gpus,
        resources=debug_args.resources or {},
        local_mode=True,
        include_dashboard=debug_args.include_dashboard,
        _temp_dir=debug_args.temp_dir)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        server_port=debug_args.server_port,
        fail_fast=debug_args.fail_fast,
        scheduler=None,
        reuse_actors=True)


def run_example_cluster(example_module_name, example_argv):
    """Run example on cluster mode.

    This functions is very similar to the local mode, except that it
    correctly sets the ray address to make ray/tune work on a cluster.
    """
    example_module = importlib.import_module(example_module_name)

    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)

    experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

    address = ray.services.get_node_ip_address() + ':6379'

    ray.init(
        address=address,
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        local_mode=False,
        include_dashboard=example_args.include_dashboard,
        _temp_dir=example_args.temp_dir)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        server_port=example_args.server_port,
        fail_fast=example_args.fail_fast,
        scheduler=None,
        queue_trials=True,
        reuse_actors=True)


def launch_example_cluster(example_module_name,
                           example_argv,
                           config_file,
                           screen,
                           tmux,
                           stop,
                           start,
                           override_cluster_name,
                           port_forward):
    """Launches the example on autoscaled ray cluster through ray exec_cmd.

    This handles basic validation and sanity checks for the experiment, and
    then executes the command on autoscaled ray cluster. If necessary, it will
    also fill in more useful defaults for our workflow (i.e. for tmux and
    cluster_name).
    """
    example_module = importlib.import_module(example_module_name)

    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)

    experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

    experiments_info = get_experiments_info([experiment_kwargs])
    total_number_of_trials = experiments_info['total_number_of_trials']

    if not example_args.upload_dir:
        confirm_yes_no(
            "`upload_dir` is empty. No results will be uploaded to cloud"
            " storage. Use `--upload-dir` argument to set upload dir."
            " Continue without upload directory?\n(yes/no) ")

    confirm_yes_no(f"Launch {total_number_of_trials} trials?\n(yes/no) ")

    override_cluster_name = override_cluster_name or unique_cluster_name(
        example_args)

    cluster_command_parts = (
        'softlearning',
        'run_example_cluster',
        example_module_name,
        *example_argv)
    cluster_command = ' '.join(cluster_command_parts)

    return exec_cluster(
        config_file=config_file,
        cmd=cluster_command,
        docker=False,
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=start,
        override_cluster_name=override_cluster_name,
        port_forward=port_forward)


def launch_example_gce(*args, config_file, **kwargs):
    """Forwards call to `launch_example_cluster` after adding gce defaults.

    This optionally sets the ray autoscaler configuration file to the default
    gce configuration file, and then calls `launch_example_cluster` to
    execute the original command on autoscaled gce cluster by parsing the args.

    See `launch_example_cluster` for further details.
    """
    config_file = config_file or AUTOSCALER_DEFAULT_CONFIG_FILE_GCE

    return launch_example_cluster(
        *args,
        config_file=config_file,
        **kwargs)


def launch_example_ec2(*args, config_file, **kwargs):
    """Forwards call to `launch_example_cluster` after adding ec2 defaults.

    This optionally sets the ray autoscaler configuration file to the default
    ec2 configuration file, and then calls `launch_example_cluster` to
    execute the original command on autoscaled ec2 cluster by parsing the args.

    See `launch_example_cluster` for further details.
    """
    config_file = config_file or AUTOSCALER_DEFAULT_CONFIG_FILE_EC2

    launch_example_cluster(
        *args,
        config_file=config_file,
        **kwargs)
