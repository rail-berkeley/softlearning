"""Sampler that saves images from env to ImagePool.

TODO.hartikainen: This is still work-in-progress until I can figure out a
reasonable apis between the sampler, pool, observations (images) preprocessor,
and the policy.
"""

import numpy as np
from skimage.transform import resize

from .simple_sampler import SimpleSampler


class ImageSampler(SimpleSampler):
    def __init__(self, resize_kwargs, *args, **kwargs):
        self.resize_kwargs = resize_kwargs
        super(ImageSampler, self).__init__(*args, **kwargs)

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        (action, _, _), _ = self.policy.get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)

        image_id = np.array([self._n_episodes, self._path_length])
        image = self.env.render(mode='rgb_array',)
        resized_image = resize(image, **self.resize_kwargs)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observations=self._current_observation,
            actions=action,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation,
            images=resized_image
        )

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation
