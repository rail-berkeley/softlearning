import pytest
import numpy as np
from dm_control import suite

from action_scale import Wrapper as ActionScaleWrapper


def test_scale_action():
    seed = 0
    unwrapped_env = suite.load(
        domain_name="quadruped", task_name="run",
        task_kwargs={"random": seed})
    assert np.any(np.not_equal(unwrapped_env.action_spec().minimum, -1.0))
    assert np.any(np.not_equal(unwrapped_env.action_spec().maximum, 1.0))

    wrapped_env = ActionScaleWrapper(
        suite.load(
            domain_name="quadruped",
            task_name="run",
            task_kwargs={"random": seed}),
        minimum=-1,
        maximum=1)
    assert np.all(np.equal(wrapped_env.action_spec().minimum, -1.0))
    assert np.all(np.equal(wrapped_env.action_spec().maximum, 1.0))

    timestep_unwrapped = unwrapped_env.reset()
    timestep_wrapped = wrapped_env.reset()

    assert (set(timestep_unwrapped.observation.keys())
            == set(timestep_wrapped.observation.keys()))

    for key in timestep_unwrapped.observation.keys():
        np.testing.assert_allclose(
            timestep_unwrapped.observation[key],
            timestep_wrapped.observation[key])

    timestep_unwrapped = unwrapped_env.step(
        unwrapped_env.action_spec().maximum)

    assert np.any(
        wrapped_env.action_spec().maximum < unwrapped_env.action_spec().maximum)
    with pytest.raises(AssertionError):
        wrapped_env.step(unwrapped_env.action_spec().maximum)

    timestep_wrapped = wrapped_env.step(
        np.ones_like(unwrapped_env.action_spec().maximum))

    for key in timestep_unwrapped.observation.keys():
        np.testing.assert_allclose(
            timestep_unwrapped.observation[key],
            timestep_wrapped.observation[key])

    assert np.allclose(timestep_wrapped.reward, timestep_unwrapped.reward)
