#!/usr/bin/env python3

import argparse
from distutils.version import LooseVersion
import os
import subprocess
import sys


KNOWN_PLATFORMS = ('linux', 'darwin')
DEFAULT_MUJOCO_PATH = '~/.mujoco'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mujoco-path', type=str, default=DEFAULT_MUJOCO_PATH)
    parser.add_argument('--versions',
                        type=str,
                        nargs='+',
                        default=('2.00', ))
    return parser


def get_mujoco_zip_name(platform, version):
    past_150 = LooseVersion(version) > LooseVersion("1.50")
    basename = "mujoco" if past_150 else "mjpro"

    if platform == 'darwin':
        platform_id = 'macos' if past_150 else 'osx'
    elif platform == 'linux':
        platform_id = 'linux'
    else:
        raise ValueError(platform)

    # For example: "mujoco200_linux.zip"
    zip_name = f"{basename}{version.replace('.', '')}_{platform_id}.zip"
    return zip_name


def install_mujoco(platform, version, mujoco_path):
    print(f"Installing MuJoCo version {version} to {mujoco_path}")

    mujoco_zip_name = get_mujoco_zip_name(platform, version)
    mujoco_dir_name = os.path.splitext(mujoco_zip_name)[0]
    if os.path.exists(os.path.join(mujoco_path, mujoco_dir_name)):
        print(f"MuJoCo {platform}, {version} already installed.")
        return

    mujoco_zip_url = f"https://www.roboti.us/download/{mujoco_zip_name}"

    if subprocess.call(["command", "-v", "wget"], shell=True) == 0:
        subprocess.check_call([
            "wget",
            "--progress=bar:force",
            "--show-progress",
            "--timestamping",
            "--directory-prefix",
            mujoco_path,
            mujoco_zip_url])
    elif subprocess.call(["command", "-v", "curl"], shell=True) == 0:
        subprocess.check_call([
            "curl",
            "--location",
            "--show-error",
            "--output",
            os.path.join(mujoco_path, mujoco_zip_name),
            mujoco_zip_url])
    else:
        raise ValueError("Need either `wget` or `curl` to download mujoco.")

    subprocess.call([
        "unzip",
        "-n",
        os.path.join(mujoco_path, mujoco_zip_name),
        "-d",
        mujoco_path])
    subprocess.call(["rm", os.path.join(mujoco_path, mujoco_zip_name)])

    if LooseVersion(version) == LooseVersion('2.0'):
        subprocess.call([
            "ln",
            "-s",
            os.path.join(mujoco_path, mujoco_dir_name),
            os.path.join(mujoco_path, "mujoco200"),
        ])


def main():
    parser = get_parser()
    args = parser.parse_args()
    mujoco_path = os.path.expanduser(args.mujoco_path)

    if not os.path.exists(mujoco_path):
        os.makedirs(mujoco_path)

    platform = sys.platform
    assert platform in KNOWN_PLATFORMS, (platform, KNOWN_PLATFORMS)

    for version in args.versions:
        install_mujoco(platform, version, mujoco_path)


if __name__ == '__main__':
    main()
