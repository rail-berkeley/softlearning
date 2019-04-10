#!/usr/bin/env python3


import argparse
from distutils.version import LooseVersion
import os
import sys
from pprint import pprint


KNOWN_PLATFORMS = ('linux', 'darwin')
DEFAULT_MUJOCO_PATH = '~/.mujoco'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mujoco-path', type=str, default=DEFAULT_MUJOCO_PATH)
    parser.add_argument('--versions',
                        type=str,
                        nargs='+',
                        default=('1.50', '2.00'))
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

    mujoco_zip_url = f"https://www.roboti.us/download/{mujoco_zip_name}"
    os.system(f"wget -N -P {mujoco_path} {mujoco_zip_url}")
    os.system(f"unzip -n {mujoco_path}/{mujoco_zip_name} -d {mujoco_path}")
    os.system(f"rm {mujoco_path}/{mujoco_zip_name}")


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
