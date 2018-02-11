import os
import argparse
import pickle
import glob
import json
from collections import defaultdict

from sac.misc.visualization import plot_visitations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_glob',
                        type=str,
                        help='Glob pattern for experiments to visualize')

    args = parser.parse_args()

    return args

def visitation_plots(args):
    runs = defaultdict(dict)
    for rollout_path in glob.iglob(args.rollout_glob):
        rollout_dir = os.path.dirname(rollout_path)

        variant_path = os.path.join(rollout_dir, "variant.json")
        with open(variant_path, "r") as f:
            variant = json.load(f)

        with open(rollout_path, "rb") as f:
            run_data = pickle.load(f)

        run_id = variant["exp_name"].split("-")[-1]
        key = None
        runs[key][run_id] = run_data

    for key, variant_runs in runs.items():
        print(key)
        plot_visitations(variant_runs, suptitle=key)

if __name__ == "__main__":
    args = parse_args()
    visitation_plots(args)
