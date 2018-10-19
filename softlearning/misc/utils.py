import collections
import datetime
import os
import random

import tensorflow as tf
import numpy as np


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


DEFAULT_SNAPSHOT_MODE = 'none'
DEFAULT_SNAPSHOT_GAP = 1000


def setup_rllab_logger(variant):
    """Temporary setup for rllab logger previously handled by run_experiment.

    TODO.hartikainen: Remove this once we have gotten rid of rllab logger.
    """

    from rllab.misc import logger

    run_params = variant['run_params']

    ray_log_dir = os.getcwd()
    log_dir = os.path.join(ray_log_dir, 'rllab-logger')

    tabular_log_file = os.path.join(log_dir, 'progress.csv')
    text_log_file = os.path.join(log_dir, 'debug.log')
    variant_log_file = os.path.join(log_dir, 'variant.json')

    logger.log_variant(variant_log_file, variant)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(
        run_params.get('snapshot_mode', DEFAULT_SNAPSHOT_MODE))
    logger.set_snapshot_gap(
        run_params.get('snapshot_gap', DEFAULT_SNAPSHOT_GAP))
    logger.set_log_tabular_only(False)

    # TODO.hartikainen: need to remove something, or push_prefix, pop_prefix?
    # logger.push_prefix("[%s] " % args.exp_name)


def set_seed(seed):
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print("Using seed {}".format(seed))


def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)


def datestamp(divider=''):
    return datetime.date.today().isoformat().replace('-', divider)


def timestamp(divider=''):
    now = datetime.datetime.now()
    time_now = datetime.datetime.time(now)
    return time_now.strftime('%H-%M-%S-%Z').replace('-', divider)


def concat_obs_z(obs, z, num_skills):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_one_hot = np.zeros(num_skills)
    z_one_hot[z] = 1
    return np.hstack([obs, z_one_hot])


def split_aug_obs(aug_obs, num_skills):
    """Splits an augmented observation into the observation and Z."""
    (obs, z_one_hot) = (aug_obs[:-num_skills], aug_obs[-num_skills:])
    z = np.where(z_one_hot == 1)[0][0]
    return (obs, z)


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def _save_video(paths, filename):
    import cv2
    assert all(['ims' in path for path in paths])
    ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


def _softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)


def deep_update(d, *us):
    d = d.copy()

    for u in us:
        u = u.copy()
        for k, v in u.items():
            d[k] = (
                deep_update(d.get(k, {}), v)
                if isinstance(v, collections.Mapping)
                else v)

    return d


def get_git_rev():
    try:
        import git
    except ImportError:
        print(
            "Warning: gitpython not installed."
            " Unable to log git rev."
            " Run `pip install gitpython` if you want git revs to be logged.")
        return None

    try:
        repo = git.Repo(os.getcwd())
        git_rev = repo.active_branch.commit.name_rev
    except TypeError:
        git_rev = repo.head.object.name_rev

    return git_rev


def flatten(unflattened, parent_key='', separator='.'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result
