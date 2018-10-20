from .nn_policy import NNPolicy
from .stochastic_policy import StochasticNNPolicy
from .gmm import GMMPolicy
from .latent_space_policy import LatentSpacePolicy
from .uniform_policy import UniformPolicy
from .gaussian_policy import GaussianPolicy


def get_gaussian_policy(env, Q, preprocessor, **kwargs):
    policy = GaussianPolicy(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        **kwargs)

    return policy


def get_gmm_policy(env, Q, **kwargs):
    policy = GMMPolicy(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        **kwargs)

    return policy


def get_latent_space_policy(env, Q, preprocessor, **kwargs):
    policy = LatentSpacePolicy(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        Q=Q,
        observations_preprocessor=preprocessor,
        **kwargs)

    return policy


def get_uniform_policy(env, *args, **kwargs):
    policy = UniformPolicy(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape)

    return policy


POLICY_FUNCTIONS = {
    'GaussianPolicy': get_gaussian_policy,
    'GMMPolicy': get_gmm_policy,
    'LatentSpacePolicy': get_latent_space_policy,
    'UniformPolicy': get_uniform_policy,
}


def get_policy(policy_type, env, **kwargs):
    return POLICY_FUNCTIONS[policy_type](env, **kwargs)


def get_policy_from_variant(variant, env, Qs, preprocessor):
    policy_params = variant['policy_params'].copy()
    policy_type = policy_params.pop('type')

    if policy_type == 'GMMPolicy':
        assert not policy_params['reparameterize'], (
            "GMMPolicy cannot be reparameterized")

    policy = get_policy(
        policy_type,
        env,
        Q=Qs[0],
        preprocessor=preprocessor,
        **policy_params)

    return policy
