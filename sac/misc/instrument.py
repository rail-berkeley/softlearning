import os
import uuid

from rllab.misc.instrument import run_experiment_lite

from sac.misc.utils import PROJECT_PATH


def _create_symlink(folder):
    # Create a symbolic link that points to the sac folder and include it
    # in the tarball.

    # Unique filename for the symlink.
    include_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(include_path)

    os.symlink(os.path.join(PROJECT_PATH, folder),
               os.path.join(include_path, folder))

    return include_path


def run_sac_experiment(main, mode, include_folders=None, **kwargs):
    if include_folders is None:
        include_folders = list()

    if mode == 'ec2':
        include_folders.append('sac')
        all_symlinks = list()

        for folder in include_folders:
            all_symlinks.append(_create_symlink(folder))

        kwargs.update(added_project_directories=all_symlinks)

    run_experiment_lite(
        stub_method_call=main,
        mode=mode,
        **kwargs,
    )
