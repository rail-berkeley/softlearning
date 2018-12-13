import unittest
import numpy as np

from gym import spaces


class AdapterTestClass(object):
    ENVIRONMENTS = []

    def test_environments(self):
        # Make sure that all the environments are creatable
        SKIP_ENVIRONMENTS = (
            ('pusher-2d', 'image-default'),
            ('pusher-2d', 'image-reach'),
            ('pusher-2d', 'blind-reach'),
        )
        environments = [
            str(self.create_adapter(domain=domain, task=task))
            for domain, tasks in self.ENVIRONMENTS.items()
            for task in tasks
            if (domain, task) not in SKIP_ENVIRONMENTS
        ]

        self.assertEqual(
            environments, self.EXPECTED_ENVIRONMENTS)

    def test_observation_space(self):
        env = self.create_adapter()
        observation_space = env.observation_space
        self.assertTrue(
            isinstance(observation_space, spaces.box.Box))

    def test_action_space(self):
        env = self.create_adapter()
        action_space = env.action_space
        self.assertTrue(
            isinstance(action_space, spaces.box.Box))

    def test_step(self):
        env = self.create_adapter()
        step = env.step(env.action_space.sample())

        self.assertTrue(isinstance(step, tuple))
        self.assertEqual(len(step), 4)

        observation, reward, done, info = step

        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(reward, np.float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_reset(self):
        env = self.create_adapter()
        observation = env.reset()
        self.assertIsInstance(observation, np.ndarray)

    @unittest.skip("The test annoyingly opens a glfw window.")
    def test_render(self):
        env = self.create_adapter()
        result = env.render(mode='rgb_array')
        self.assertIsInstance(result, np.ndarray)

    def test_close(self):
        env = self.create_adapter()
        env.close()
