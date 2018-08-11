import os
import uuid

from rllab.misc.instrument import run_experiment_lite
from softlearning.misc.utils import datetimestamp, PROJECT_PATH

DEFAULT_LOG_DIR = PROJECT_PATH + "/data"


def _create_symlink(folder):
    """Create a symbolic link that points to the softlearning folder."""

    # Unique filename for the symlink.
    include_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(include_path)

    os.symlink(
        os.path.join(PROJECT_PATH, folder), os.path.join(include_path, folder))

    return include_path


def launch_experiment(main,
                      mode,
                      include_folders=None,
                      log_dir=None,
                      exp_prefix="experiment",
                      exp_name=None,
                      **kwargs):
    if exp_name is None:
        exp_name = datetimestamp()

    if include_folders is None:
        include_folders = []

    if mode == 'ec2':
        include_folders += ['softlearning', 'models', 'snapshots']
        all_symlinks = []

        for folder in include_folders:
            all_symlinks.append(_create_symlink(folder))

        kwargs.update(added_project_directories=all_symlinks)

    print("\nlog_dir={}\nexp_prefix={}\n".format(log_dir, exp_prefix))

    run_experiment_lite(
        stub_method_call=main,
        mode=mode,
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        log_dir=log_dir,
        **kwargs)
