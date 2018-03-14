from sac.misc.utils import concat_obs_z
import numpy as np
import tensorflow as tf

class FixedOptionPolicy(object):
    def __init__(self, base_policy, num_skills, z):
        self._z = z
        self._base_policy = base_policy
        self._num_skills = num_skills

    def reset(self):
        pass

    def get_action(self, obs):
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)

    def get_distribution_for(self, obs_t, reuse=False):
        shape = [tf.shape(obs_t)[0]]
        z = tf.tile([self._z], shape)
        z_one_hot = tf.one_hot(z, self._num_skills, dtype=obs_t.dtype)
        aug_obs_t = tf.concat([obs_t, z_one_hot], axis=1)
        return self._base_policy.get_distribution_for(aug_obs_t, reuse=reuse)

class ScheduledOptionPolicy(object):
    def __init__(self, base_policy, num_skills, z_vec):
        self._z_vec = z_vec
        self._base_policy = base_policy
        self._num_skills = num_skills
        self._t = 0

    def reset(self):
        pass

    def get_action(self, obs):
        assert self._t < len(self._z_vec)
        z = self._z_vec[self._t]
        aug_obs = concat_obs_z(obs, z, self._num_skills)
        self._t += 1
        return self._base_policy.get_action(aug_obs)


class RandomOptionPolicy(object):

    def __init__(self, base_policy, num_skills, steps_per_option):
        self._num_skills = num_skills
        self._steps_per_option = steps_per_option
        self._base_policy = base_policy
        self.reset()

    def reset(self):
        self._z = np.random.choice(self._num_skills)

    def get_action(self, obs):
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)

class HierarchicalPolicy(object):
    def __init__(self, base_policy, num_skills, meta_policy, steps_per_option):
        self._steps_per_option = steps_per_option
        self._meta_policy = meta_policy
        self._base_policy = base_policy
        self._num_skills = num_skills
        self.reset()

    def reset(self):
        self._t = 0
        self._z = None

    def get_action(self, obs):
        # Choose a skill if necessary
        if self._t % self._steps_per_option == 0:
            (self._z, _) = self._meta_policy.get_action(obs)
        self._t += 1
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)

class RandomHierarchicalPolicy(object):

    def __init__(self, base_policy, num_skills, steps_per_option):
        self._steps_per_option = steps_per_option
        self._base_policy = base_policy
        self._num_skills = num_skills
        self.reset()

    def reset(self):
        self._t = 0
        self._z = None

    def get_action(self, obs):
        # Choose a skill if necessary
        if self._t % self._steps_per_option == 0:
            self._z = np.random.choice(self._num_skills)
        self._t += 1
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)
