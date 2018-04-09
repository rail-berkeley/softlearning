import os
import argparse
import pickle
import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from sac.misc.visualization import plot_hierarchy_visitations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_glob',
                        type=str,
                        help='Glob to rollout paths to visualize')

    args = parser.parse_args()

    return args

def visitation_plots(args):
    for rollout_path in glob.iglob(args.rollout_glob):
        rollout_dir = os.path.dirname(rollout_path)

        variant_path = os.path.join(rollout_dir, "variant.json")
        with open(variant_path, "r") as f:
            variant = json.load(f)

        with open(rollout_path, "rb") as f:
            run_data = pickle.load(f)

        run_id = variant["exp_name"].split("-")[-1]

        save_file = (os.path.splitext(os.path.basename(rollout_path))[0]
                     + '_visitations.png')
        rollout_path.replace('/Users/kristian/code/rllab/data/s3/', '')
        VISUALIZATION_FOLDER = '/Users/kristian/code/softqlearning-private/vis/hierarchy_visitations/'
        save_dir = os.path.join(
            VISUALIZATION_FOLDER,
            os.path.dirname(rollout_path.replace('/Users/kristian/code/rllab/data/s3/', '')))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, save_file)
        plot_hierarchy_visitations(
            run_data, variant, save_path=save_path)

if __name__ == "__main__":
    args = parse_args()
    visitation_plots(args)
