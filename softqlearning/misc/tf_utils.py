import tensorflow as tf
from rllab import config


def get_default_session():
    return tf.get_default_session() or create_session()


def create_session(**kwargs):
    """ Creates a new tensorflow session with a given configuration. """
    if "config" not in kwargs:
        kwargs["config"] = get_configuration()
    return tf.InteractiveSession(**kwargs)


def get_configuration():
    """ Returns personal tensorflow configuration. """
    config_args = dict(
        gpu_options=tf.GPUOptions(
            allow_growth=config.TF_GPU_ALLOW_GROWTH,
            per_process_gpu_memory_fraction=config.TF_GPU_MEM_FRAC,
        ),
        log_device_placement=config.TF_LOG_DEVICE_PLACEMENT,
    )
    if not config.TF_USE_GPU:
        config_args["device_count"] = {'GPU': 0}
    return tf.ConfigProto(**config_args)
