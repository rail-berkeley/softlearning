import os.path as osp

import numpy as np
from skimage.transform import resize

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.mujoco_py import MjViewer

from softlearning.misc.utils import PROJECT_PATH
from .pusher import PusherEnv


class ImagePusherEnv(PusherEnv):
    def __init__(self, image_size, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.image_size = image_size
        PusherEnv.__init__(self, *args, **kwargs)
        self.viewer_setup()

    def get_current_obs(self):
        self.viewer_setup()
        image = self.render(mode='rgb_array')

        # image = resize(image, output_shape=(16,16,3), preserve_range=True)
        # show_image = True
        # if show_image:
        #     from PIL import Image
        #     Image.fromarray(image.astype('uint8')).show()
        #     # .save(
        #     #     '/Users/kristian/code/softqlearning-private/tmp/{}_real.png'.format(self.i)
        #     # )
        #     # Image.fromarray(resized_image.astype('uint8')).show()
        #     # .save(
        #     #     '/Users/kristian/code/softqlearning-private/tmp/{}_resized.png'.format(self.i)
        #     # )
        #     # setattr(self, 'i', getattr(self, 'i', 0) + 1)
        #     from pdb import set_trace; from pprint import pprint; set_trace()

        return np.concatenate([
            image.reshape(-1),
            self.model.data.qpos.flat[self.JOINT_INDS],
            self.model.data.qvel.flat[self.JOINT_INDS],
        ]).reshape(-1)

    def get_viewer(self):
        if self.viewer is None:
            width, height = self.image_size[:2]
            self.viewer = MjViewer(
                visible=False, init_width=width, init_height=height)
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def viewer_setup(self):
        viewer = self.get_viewer()
        viewer.cam.trackbodyid = 0
        cam_dist = 3.5
        cam_pos = np.array([0, 0, 0, cam_dist, -90, 0])
        viewer.cam.lookat[:3] = [0,0,0]
        viewer.cam.distance = cam_dist
        viewer.cam.elevation = -90
        viewer.cam.azimuth = 0
        viewer.cam.trackbodyid = -1

    def render(self, *args, **kwargs):
        self.viewer_setup()
        return super(ImagePusherEnv, self).render(*args, **kwargs)
