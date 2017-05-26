import argparse

import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout
from rllab.misc.ext import set_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Fixed random seed')
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)
    args = parser.parse_args()

    policy = None
    env = None

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    # while True:
    if args.seed >= 0:
        set_seed(args.seed)
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if "algo" in data:
            policy = data["algo"].policy
            env = data["algo"].env
        else:
            policy = data['policy']
            env = data['env']
        # while True:
        for i in range(8):
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=args.render, speedup=args.speedup, itr=i)
            print("latent " + str(i) + ":",
                  path["rewards"][-1],
                  path["mean_velocity"],
                  path["successful"])
