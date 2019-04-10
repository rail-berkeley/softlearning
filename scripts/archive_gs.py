#!/usr/bin/python

import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_path', type=str, default=None, nargs='?')
    parser.add_argument('--unarchive', action='store_true', default=False)
    parser.add_argument('--dry', action='store_true', default=False)
    args = parser.parse_args()

    return args


def archive_gs(args):
    """Archive files in google cloud storage bucket.

    Moves files from `<bucket>/ray/results` to `<bucket>/archive/ray/results`.

    TODO(hartikainen): Refactor this to use project config instead of
        environment variables (e.g. `SAC_GS_BUCKET`).
    """
    if 'SAC_GS_BUCKET' not in os.environ:
        raise ValueError(
            "'SAC_GS_BUCKET' environment variable needs to be set.")

    bucket = os.environ['SAC_GS_BUCKET']
    fresh_results_path = os.path.join(bucket, 'ray', 'results')
    archive_results_path = os.path.join(bucket, 'archive', 'ray', 'results')

    fresh_url = os.path.join(fresh_results_path, args.archive_path)
    archive_url = os.path.join(archive_results_path, args.archive_path)

    src_url, dst_url = (
        (archive_url, fresh_url)
        if args.unarchive
        else (fresh_url, archive_url))

    command_parts = ['gsutil', '-m', 'mv', src_url, dst_url]
    command = " ".join(command_parts)

    if args.dry:
        print(command)
        return

    subprocess.call(command, shell=True)


def main():
    args = parse_args()
    archive_gs(args)


if __name__ == '__main__':
    main()
