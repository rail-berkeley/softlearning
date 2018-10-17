"""Replay pool that also stores images from the environment.

TODO.hartikainen: This is still work-in-progress until I can figure out a
reasonable apis between the sampler, pool, observations (images) preprocessor,
and the policy.
"""

import os
from PIL import Image

import numpy as np

from serializable import Serializable

from .replay_pool import ReplayPool
from .simple_replay_pool import SimpleReplayPool


class ImageReplayPool(SimpleReplayPool, Serializable):
    def __init__(self, image_shape, *args, **kwargs):
        self._Serializable__initialize(locals())
        super(ImageReplayPool, self).__init__(*args, **kwargs)

        self.image_shape = image_shape

        fields = {
            'images': {
                'shape': image_shape,
                'dtype': 'uint8'
            },
        }

        self.add_fields(fields)
