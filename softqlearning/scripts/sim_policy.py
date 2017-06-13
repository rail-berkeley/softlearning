import argparse
import joblib

import tensorflow as tf
import numpy as np

from softqlearning.misc.sampler import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    # Note: max-path-length might be overridden by the environment.
    parser.add_argument('--max-path-length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Fixed random seed')
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)
    args = parser.parse_args()

    policy = None
    env = None

    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']

        while True:
            path = rollout(env, policy, path_length=args.max_path_length,
                           render=args.render, speedup=args.speedup)
            print('Total reward:', np.sum(path["rewards"]))
