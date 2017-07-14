from rllab import config
import subprocess
import os.path as osp


def _rllab_root():
    import rllab
    return osp.abspath(osp.dirname(osp.dirname(rllab.__file__)))


def exec_cmd(cmd):
    print("Exec -- " + " ".join(cmd))
    subprocess.check_call(cmd)


def docker_build(docker_image=None):
    root = _rllab_root()
    if docker_image is None:
        docker_image = config.DOCKER_IMAGE
    dockerfile_fullpath = osp.join(root, config.DOCKERFILE_PATH)
    exec_cmd(["docker", "build", "-t=%s" % docker_image, "-f=%s" % dockerfile_fullpath, root])


def docker_push(docker_image=None):
    if docker_image is None:
        docker_image = config.DOCKER_IMAGE
    exec_cmd(["docker", "push", docker_image])


def docker_run(docker_image=None):
    if docker_image is None:
        docker_image = config.DOCKER_IMAGE
    exec_cmd(["docker", "run", "-it", docker_image, "/bin/bash"])
