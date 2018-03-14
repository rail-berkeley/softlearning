import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import joblib
import tensorflow as tf
import os
from sac.misc import utils
from sac.policies.hierarchical_policy import FixedOptionPolicy
from sac.misc.sampler import rollouts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=100)
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--dim_0', type=int, default=0)
    parser.add_argument('--dim_1', type=int, default=1)
    parser.add_argument('--use_qpos', type=bool, default=False)
    parser.add_argument('--use_action', type=bool, default=False)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()
    filename = '{}_{}_{}_trace.png'.format(os.path.splitext(args.file)[0],
                                           args.dim_0, args.dim_1)

    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim

        plt.figure(figsize=(6, 6))
        palette = sns.color_palette('hls', num_skills)
        with policy.deterministic(args.deterministic):
            for z in range(num_skills):
                fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
                for path_index in range(args.n_paths):
                    obs = env.reset()
                    if args.use_qpos:
                        qpos = env.wrapped_env.env.model.data.qpos[:, 0]
                        obs_vec = [qpos]
                    else:
                        obs_vec = [obs]
                    for t in range(args.max_path_length):
                        action, _ = fixed_z_policy.get_action(obs)
                        (obs, _, _, _) = env.step(action)
                        if args.use_qpos:
                            qpos = env.wrapped_env.env.model.data.qpos[:, 0]
                            obs_vec.append(qpos)
                        elif args.use_action:
                            obs_vec.append(action)
                        else:
                            obs_vec.append(obs)

                    obs_vec = np.array(obs_vec)
                    x = obs_vec[:, args.dim_0]
                    y = obs_vec[:, args.dim_1]
                    plt.plot(x, y, c=palette[z])

        plt.savefig(filename)
        plt.close()
