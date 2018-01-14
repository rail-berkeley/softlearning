import os
import argparse
import pickle
import glob
import json
from collections import defaultdict

from sac.misc.visualization import plot_visitations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_glob',
                        type=str,
                        help='Glob pattern for experiments to visualize')

    args = parser.parse_args()

    return args

def visitation_plots(args):
    runs = defaultdict(dict)
    for run_path in glob.iglob(args.run_glob):
        path_files = glob.iglob(os.path.join(run_path, "itr_*_path.pkl"))
        path_file = list(sorted(path_files))[-1] # os.path.join(run_path, "itr_500_path.pkl")
        variant_file = os.path.join(run_path, "variant.json")
        with open(variant_file, "r") as f:
            variant = json.load(f)

        with open(path_file, "rb") as f:
            run_data = pickle.load(f)

        run_id = variant["exp_name"].split("-")[-1]
        key = (variant["tau"], variant["scale_reward"])
        runs[key][run_id] = run_data

    for key, variant_runs in runs.items():
        print(key)
        plot_visitations(variant_runs, suptitle=key)

if __name__ == "__main__":
    args = parse_args()
    visitation_plots(args)
