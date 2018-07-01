import os.path as osp

import numpy as np
from skimage.transform import resize

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from softlearning.misc.utils import PROJECT_PATH
from .pusher import PusherEnv


class ImagePusherEnv(PusherEnv):
    @overrides
    def get_current_obs(self):
        image = self.render(mode='rgb_array',)
        resized_image = resize(
            image,
            output_shape=(32,32,3),
            preserve_range=True,
            anti_aliasing=True)

        show_image = False
        if show_image:
            from PIL import Image
            Image.fromarray(image.astype('uint8')).show()
            # .save(
            #     '/Users/kristian/code/softqlearning-private/tmp/{}_real.png'.format(self.i)
            # )
            Image.fromarray(resized_image.astype('uint8')).show()
            # .save(
            #     '/Users/kristian/code/softqlearning-private/tmp/{}_resized.png'.format(self.i)
            # )
            # setattr(self, 'i', getattr(self, 'i', 0) + 1)
            from pdb import set_trace; from pprint import pprint; set_trace()

        return np.concatenate([
            resized_image.reshape(-1),
            self.model.data.qpos.flat[self.JOINT_INDS],
            self.model.data.qvel.flat[self.JOINT_INDS],
        ]).reshape(-1)

    def viewer_setup(self):
        viewer = self.get_viewer()
        viewer.cam.trackbodyid = 0
        viewer.cam.distance = 4.0
        cam_dist = 3.5
        cam_pos = np.array([0, 0, 0, cam_dist, -90, 0])
        viewer.cam.lookat[:3] = cam_pos[:3]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = -1

    def render(self, *args, **kwargs):
        self.viewer_setup()
        result = super(PusherEnv, self).render(*args, **kwargs)

        return result
