import glob
import os
from pprint import pprint
import re
import shutil


RESULT_FILE_REGEXES = (
    "^result.json$",
    "^progress.csv$",
    "^events.out.tfevents.\\d+.\\w$",
)


PARAMS_FILE_REGEXES = (
    "^params.json$",
    "^params.pkl$",
)

CHECKPOINT_DIRECTORY_REGEXES = (
    "^checkpoint_\\d+$"
)


def is_result_file(filename):
    return any(
        re.match(result_file_regex, filename)
        for result_file_regex in RESULT_FILE_REGEXES)


def is_params_file(filename):
    return any(
        re.match(params_file_regex, filename)
        for params_file_regex in PARAMS_FILE_REGEXES)


def is_checkpoint_directory(dirname):
    # TODO(hartikainen): might want to check the contents of this directory.
    # e.g. check `.tune_metadata`, etc.
    return any(
        re.match(checkpoint_directory_regex, dirname)
        for checkpoint_directory_regex in CHECKPOINT_DIRECTORY_REGEXES)


def is_trial_directory(root_dir):
    if not os.path.isdir(root_dir):
        return False

    root, directories, files = next(os.walk(root_dir))
    # json logger: params.json, result.json, params.pkl
    # csv logger: progress.csv
    # tf logger: events.out.tfevents.1562394433.ray-hopp-2-head-4ba37bcf
    # log_syncxurz09ic.log

    result_files = [
        filename
        for filename in files
        if is_result_file(filename)
    ]

    params_files = [
        filename
        for filename in files
        if is_params_file(filename)
    ]

    # TODO(hartikainen): checkpoint_directories are currently unused here
    checkpoint_directories = [
        directory
        for directory in directories
        if is_checkpoint_directory(os.path.join(root, directory))
    ]

    # TODO(hartikainen): might want to check if "^log_sync\\d{8}.log$" exists

    return result_files and params_files


def is_experiment_directory(root_dir):
    if not os.path.isdir(root_dir):
        return False

    root, directories, files = next(os.walk(root_dir))
    # 1) experiment_state.json exists -> is experiment
    experiment_state_paths = glob.glob(
        os.path.join(root, "experiment_state*.json"))

    if experiment_state_paths:
        # TODO(hartikainen): This needs to be fixed. In general, a directory
        # can have multiple experiment state files. Softlearning experiment
        # directories shouldn't though.
        assert len(experiment_state_paths) == 1, experiment_state_paths
        return True

    # 2) All the subfolders are trials -> is experiment
    if directories and all(
            is_trial_directory(os.path.join(root, directory))
            for directory in directories):
        return True

    return False


def find_all_experiment_directories(root_dir):
    """Given a directory path, recursively find all experiment directories in it.

    TODO(hartikainen): Should maybe have an option for recursive=False?
    """

    root_dir = os.path.expanduser(root_dir)

    if is_experiment_directory(root_dir):
        return (root_dir, )

    directories = next(os.walk(root_dir))[1]
    all_experiment_directories = sum((
        find_all_experiment_directories(os.path.join(root_dir, directory))
        for directory in directories
    ), ())

    return all_experiment_directories


def find_all_trial_directories(experiment_dir):
    """Given a path to experiment, find all trial directories in it.

    Raises an error if given experiment path is not actually an experiment
    path.
    """

    assert is_experiment_directory(experiment_dir), experiment_dir

    experiment_dir = os.path.expanduser(experiment_dir)
    directories = next(os.walk(experiment_dir))[1]

    all_trial_directories = [
        os.path.join(experiment_dir, directory)
        for directory in directories
        if is_trial_directory(os.path.join(experiment_dir, directory))
    ]

    return all_trial_directories
