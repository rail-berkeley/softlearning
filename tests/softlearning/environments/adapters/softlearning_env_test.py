import numpy as np
from gym import spaces


class AdapterTestClass(object):
    ENVIRONMENTS = []

    def test_observation_space(self):
        env = self.create_adapter()
        observation_space = env.observation_space
        self.assertTrue(
            isinstance(observation_space, (spaces.Box, spaces.Dict)))
        # TODO(hartikainen): Test actual conversion of dimensions and types of
        # inside items; not just outside type.

    def test_action_space(self):
        env = self.create_adapter()
        action_space = env.action_space
        self.assertTrue(
            isinstance(action_space, spaces.Box))

    def test_step(self):
        env = self.create_adapter()
        env.reset()
        step = env.step(env.action_space.sample())
        self.assertTrue(isinstance(step, tuple))
        self.assertEqual(len(step), 4)

        observation, reward, done, info = step
        self.assertIsInstance(observation, dict)
        self.assertIsInstance(reward, np.float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_reset(self):
        env = self.create_adapter()
        observation = env.reset()
        self.assertIsInstance(observation, dict)

    def test_render_rgb_array(self):
        env = self.create_adapter()
        result = env.render(mode='rgb_array')
        self.assertIsInstance(result, np.ndarray)
        env.close()

    def test_render_human(self):
        env = self.create_adapter()
        result = env.render(mode='human')
        self.assertIsNone(result)
        env.close()

    def test_close(self):
        env = self.create_adapter()
        env.close()
