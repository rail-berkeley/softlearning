import os
import uuid
import re
import sys
from pprint import pformat

import ray

from ray import tune
from ray.autoscaler.commands import exec_cluster

from softlearning.misc.utils import datetimestamp, PROJECT_PATH


def get_experiments_info(experiments):
    number_of_trials = {
        experiment_id: len(list(
            tune.suggest.variant_generator.generate_variants(
                experiment_spec['config'])
        )) * experiment_spec['num_samples']
        for experiment_id, experiment_spec in experiments.items()
    }
    total_number_of_trials = sum(number_of_trials.values())

    experiments_info = {
        "number_of_trials": number_of_trials,
        "total_number_of_trials": total_number_of_trials,
    }

    return experiments_info


def run_experiments_dry(experiments, args):
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

    return


def run_experiments_debug(experiments, args):
    """The debug limits tune trial runs to enable use of debugger.

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


def run_experiments_local(experiments, args, scheduler=None):
    resources = args.resources or {}

    ray.init(
        num_cpus=args.cpus,
        num_gpus=args.gpus,
        resources=resources,
        # Tune doesn't currently support local mode
        local_mode=False,
        include_webui=args.include_webui,
        temp_dir=args.temp_dir,
    )

    tune.run_experiments(
        experiments,
        with_server=args.with_server,
        server_port=4321,
        scheduler=scheduler)


def run_experiments_cluster(experiments, args, scheduler=None):
    resources = args.resources or {}
    redis_address = ray.services.get_node_ip_address() + ':6379'

    ray.init(
        redis_address=redis_address,
        num_cpus=args.cpus,
        num_gpus=args.gpus,
        resources=resources,
        # Tune doesn't currently support local mode
        local_mode=False,
        include_webui=args.include_webui,
        temp_dir=args.temp_dir)

    tune.run_experiments(
        experiments,
        with_server=args.with_server,
        server_port=4321,
        scheduler=scheduler)


AUTOSCALER_DEFAULT_CONFIG_PATH_GCE = os.path.join(
    PROJECT_PATH, 'config', 'gcp-ray-autoscaler-mujoco.yaml')
AUTOSCALER_DEFAULT_CONFIG_PATH_EC2 = os.path.join(
    PROJECT_PATH, 'config', 'aws-ray-autoscaler-mujoco.yaml')


def confirm_yes_no(prompt):
    # raw_input returns the empty string for "enter"
    yes = {'yes', 'y'}
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


def unique_cluster_name(args):
    cluster_name_parts = (
        datetimestamp(''),
        str(uuid.uuid4())[:6],
        args.domain,
        args.task
    )
    cluster_name = "-".join(cluster_name_parts).lower()
    return cluster_name


MODE_GCE_REGEX = r'--mode(?:=?| *)gce'
MODE_EC2_REGEX = r'--mode(?:=?| *)ec2'
MODE_CLUSTER_REGEX = r'--mode(?:=?| *)cluster'


def launch_experiments_cluster(experiments, args, cluster_command, config_file):
    """Executes the command on autoscaled cluster through ray exec_cluster.

    This handles basic validation sanity check for the experiment run, then
    converts the command line argument Namespace back to command line argument
    string, and finally executes this command on autoscaled ray cluster.
    """
    assert args.mode == 'cluster', args.mode
    assert re.search(MODE_CLUSTER_REGEX, cluster_command) is not None, (
        cluster_command)

    if not args.upload_dir:
        confirm_yes_no(
            "`upload_dir` is empty. No results will be uploaded to cloud"
            " storage. Use `--upload-dir` argument to set upload dir."
            " Continue without upload directory?\n(yes/no) ")

    experiments_info = get_experiments_info(experiments)
    total_number_of_trials = experiments_info['total_number_of_trials']

    confirm_yes_no(f"Launch {total_number_of_trials} trials?\n(yes/no) ")

    override_cluster_name = (
        args.autoscaler_override_cluster_name
        or unique_cluster_name(args))

    exec_cluster(
        config_file=config_file,
        cmd=cluster_command,
        screen=args.autoscaler_screen,
        tmux=args.autoscaler_tmux,
        stop=args.autoscaler_stop,
        start=args.autoscaler_start,
        override_cluster_name=override_cluster_name,
        port_forward=args.autoscaler_port_forward)


def launch_experiments_gce(experiments, args, gce_command=None):
    """Executes the run command on gce cluster using ray autoscaler api.

    This optionally sets the ray autoscaler configuration file to the default
    ec2 configuration file, and then calls `launch_experiments_cluster` to
    execute the original command on autoscaled gce cluster by parsing the args.
    """
    # Sanity checks for the command
    assert args.mode == 'gce', args.mode
    args.mode = 'cluster'

    gce_command = gce_command or ' '.join(sys.argv)
    assert re.search(MODE_GCE_REGEX, gce_command) is not None, gce_command
    cluster_command = re.sub(MODE_GCE_REGEX, "--mode=cluster", gce_command)

    config_file = (
        args.autoscaler_config_file
        or AUTOSCALER_DEFAULT_CONFIG_PATH_GCE)

    return launch_experiments_cluster(
        experiments, args, cluster_command, config_file)


def launch_experiments_ec2(experiments, args, ec2_command=None):
    """Executes the run command on ec2 cluster using ray autoscaler api.

    This optionally sets the ray autoscaler configuration file to the default
    ec2 configuration file, and then calls `launch_experiments_cluster` to
    execute the original command on autoscaled ec2 cluster by parsing the args.
    """
    # Sanity checks for the command
    assert args.mode == 'ec2', args.mode
    args.mode = 'cluster'

    ec2_command = ec2_command or ' '.join(sys.argv)
    assert re.search(MODE_EC2_REGEX, ec2_command) is not None, ec2_command
    cluster_command = re.sub(MODE_EC2_REGEX, "--mode=cluster", ec2_command)

    config_file = (
        args.autoscaler_config_file
        or AUTOSCALER_DEFAULT_CONFIG_PATH_EC2)

    return launch_experiments_cluster(
        experiments, args, cluster_command, config_file)
