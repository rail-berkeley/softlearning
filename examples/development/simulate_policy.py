import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.misc.utils import save_video


DEFAULT_RENDER_KWARGS = {
    'mode': 'human',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-kwargs', '-r',
                        type=json.loads,
                        default='{}',
                        help="Kwargs for rollouts renderer.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args


def simulate_policy(args):
    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    evaluation_environment = get_environment_from_params(environment_params)

    policy = (
        get_policy_from_variant(variant, evaluation_environment))
    policy.set_weights(picklable['policy_weights'])

    render_kwargs = {**DEFAULT_RENDER_KWARGS, **args.render_kwargs}

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(args.num_rollouts,
                         evaluation_environment,
                         policy,
                         path_length=args.max_path_length,
                         render_kwargs=render_kwargs)

    if args.render_kwargs.get('mode') == 'rgb_array':
        fps = 1 // getattr(evaluation_environment, 'dt', 1/30)
        for i, path in enumerate(paths):
            video_save_dir = os.path.expanduser('/tmp/simulate_policy/')
            video_save_path = os.path.join(video_save_dir, f'episode_{i}.mp4')
            save_video(path['images'], video_save_path, fps=fps)

    return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
