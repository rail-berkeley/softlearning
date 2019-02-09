#!/usr/bin/python

import argparse
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'sync_path', type=str, default=None, nargs='?')
    parser.add_argument(
        '--sync-checkpoints', action='store_true', default=False)
    parser.add_argument(
        '--dry', action='store_true', default=False)
    args = parser.parse_args()

    return args


def sync_gs(args):
    """Sync files from google cloud storage bucket to local machine.

    TODO(hartikainen): Refactor this to use project config instead of
        environment variables (e.g. `SAC_GS_BUCKET`).
    """
    if 'SAC_GS_BUCKET' not in os.environ:
        raise ValueError(
            "'SAC_GS_BUCKET' environment variable needs to be set.")

    bucket = os.environ['SAC_GS_BUCKET']

    remote_gs_parts = [bucket, 'ray', 'results']
    local_gs_parts = [os.path.expanduser('~/ray_results/gs/')]

    if args.sync_path is not None:
        remote_gs_parts.append(args.sync_path)
        local_gs_parts.append(args.sync_path)

    remote_gs_path = os.path.join(*remote_gs_parts)
    local_gs_path = os.path.join(*local_gs_parts)

    if not os.path.exists(local_gs_path):
        os.makedirs(local_gs_path)

    command_parts = ['gsutil', '-m', 'rsync', '-r']

    if not args.sync_checkpoints:
        command_parts += ['-x', '".*./checkpoint_.*./.*"']

    if args.dry:
        command_parts += ["-n"]

    command_parts += [shlex.quote(remote_gs_path), shlex.quote(local_gs_path)]

    command = " ".join(command_parts)

    subprocess.call(command, shell=True)


def main():
    args = parse_args()
    sync_gs(args)


if __name__ == '__main__':
    main()
