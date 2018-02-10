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
    parser.add_argument('--run_path',
                        type=str,
                        help='Path for experiment to visualize')

    args = parser.parse_args()

    return args

def visitation_plots(args):
    run_path = args.run_path

    path_files = glob.iglob(os.path.join(run_path, "itr_*_path.pkl"))
    sorted_path_files = list(sorted(path_files))
    if not sorted_path_files:
        path_files = glob.iglob(os.path.join(run_path, "params_path.pkl"))
        sorted_path_files = list(sorted(path_files))

    path_file = sorted_path_files[-1] # os.path.join(run_path, "itr_500_path.pkl")
    variant_file = os.path.join(run_path, "variant.json")

    with open(variant_file, "r") as f:
        variant = json.load(f)

    with open(path_file, "rb") as f:
        run_data = pickle.load(f)

    run_id = variant["exp_name"].split("-")[-1]

    plot_hierarchy_visitations(run_data)

if __name__ == "__main__":
    args = parse_args()
    visitation_plots(args)
