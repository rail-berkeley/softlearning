import os
import argparse
import pickle
import glob
import json
from collections import defaultdict

import numpy as np

from sac.misc.visualization import plot_visitations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_glob',
                        type=str,
                        help='Glob pattern for experiments to visualize')

    args = parser.parse_args()

    return args

def visitation_plots(args):
    rollout_paths = glob.glob(args.rollout_glob)
    np.unique([os.path.dirname(x) for x in rollout_paths])
    for rollout_path in glob.iglob(args.rollout_glob):
        runs = {}
        rollout_dir = os.path.dirname(rollout_path)

        save_file = (os.path.splitext(os.path.basename(rollout_path))[0]
                     + '_visitations.png')
        rollout_path.replace('/Users/kristian/code/rllab/data/s3/', '')
        VISUALIZATION_FOLDER = '/Users/kristian/code/softqlearning-private/vis/visitations/'
        save_dir = os.path.join(
            VISUALIZATION_FOLDER,
            os.path.dirname(rollout_path.replace('/Users/kristian/code/rllab/data/s3/', '')))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, save_file)

        variant_path = os.path.join(rollout_dir, "variant.json")
        with open(variant_path, "r") as f:
            variant = json.load(f)

        try:
            print('opening: {}'.format(rollout_path))
            if os.path.isfile(save_path):
                print("visualization {} already exists".format(save_path))
                continue

            with open(rollout_path, "rb") as f:
                run_data = pickle.load(f)
        except Exception as e:
            from nose.tools import set_trace; from pprint import pprint; set_trace()
            pass

        run_id = variant["exp_name"].split("-")[-1]
        runs[run_id] = run_data

        plot_visitations(runs, variant=variant, save_path=save_path)


if __name__ == "__main__":
    args = parse_args()
    visitation_plots(args)
