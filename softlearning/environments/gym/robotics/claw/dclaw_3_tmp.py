import numpy as np

try:
    from sac_envs.envs.dclaw.dclaw3_screw_v11 import DClaw3ScrewV11
except ModuleNotFoundError as e:
    def raise_on_use(*args, **kwargs):
        raise ValueError('Could not import DClaw')
    DClaw3ScrewV11 = raise_on_use


class DClaw3TMPScrewV11(DClaw3ScrewV11):
    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase

        # apply actions and step
        self.robot.step(
            self,
            a[:self.robot.nJnt],
            step_duration=self.skip * self.model.opt.timestep,
            sim_override=False)

        done = False
        obs = self._get_obs()

        object_velocity = self.data.get_joint_qvel('OBJRx')

        reward = object_velocity
        r_dict = {
            'reward': reward,
            'object_velocity': object_velocity
        }

        return obs, reward, done, {'rewards': r_dict}
