import os
import uuid

from ray import tune

from softlearning.misc.utils import datetimestamp, PROJECT_PATH


AUTOSCALER_DEFAULT_CONFIG_FILE_GCE = os.path.join(
    PROJECT_PATH, 'config', 'gcp-ray-autoscaler-mujoco.yaml')
AUTOSCALER_DEFAULT_CONFIG_FILE_EC2 = os.path.join(
    PROJECT_PATH, 'config', 'aws-ray-autoscaler-mujoco.yaml')


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
