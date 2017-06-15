import numpy as np

# TODO(vpong): unittest this
def split_paths(paths):
    """
    Plit paths from rllab's rollout function into rewards, terminals, obs
    actions, and next_obs
    terminals
    :param paths:
    :return:
    """
    rewards = [path["rewards"] for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = []
    terminals = []
    for path in paths:
        next_obs_i = path["observations"][1:, :]
        next_obs_i = np.vstack((next_obs_i,
                                np.zeros_like(path["observations"][0:1])))
        next_obs.append(next_obs_i)

        terminal_i = np.zeros_like(path["rewards"])
        terminal_i[-1] = 1
        terminals.append(terminal_i)
    rewards = np.hstack(rewards)
    terminals = np.hstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    return rewards, terminals, obs, actions, next_obs


def get_action_dim(**kwargs):
    env_spec = kwargs.get('env_spec', None)
    action_dim = kwargs.get('action_dim', None)
    assert env_spec or action_dim
    return action_dim or env_spec.action_space.flat_dim


def get_observation_dim(**kwargs):
    env_spec = kwargs.get('env_spec', None)
    observation_dim = kwargs.get('observation_dim', None)
    assert env_spec or observation_dim
    return observation_dim or env_spec.observation_space.flat_dim
