import argparse

import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--policy_h', type=int)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    return args

def simulate_policy(args):
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']

        with policy.deterministic(args.deterministic):
            while True:
                path = rollout(env, policy,
                               max_path_length=args.max_path_length,
                               animated=True, speedup=args.speedup)
if __name__ == "__main__":
    args = parse_args()
    simulate_policy(args)
