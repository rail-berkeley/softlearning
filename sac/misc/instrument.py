import os
import uuid

from rllab.misc.instrument import run_experiment_lite

SAC_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..', 'sac')))


def run_sac_experiment(*args, mode, **kwargs):

    if mode == 'ec2':
        # Create a symbolic link that points to the sac folder and include it
        # in the tarball.

        # Unique filename for the symlink.
        sac_path = os.path.join('/tmp/', str(uuid.uuid4()))
        os.makedirs(sac_path)
        os.symlink(SAC_PATH, os.path.join(sac_path, 'sac'))

        kwargs.update(added_project_directories=[sac_path])

    run_experiment_lite(
        mode=mode,
        *args,
        **kwargs,
    )
