from .simple_sampler import SimpleSampler


class GoalSampler(SimpleSampler):
    @property
    def _policy_input(self):
        observation = super(GoalSampler, self)._action_input
        goal = {
            key: self._current_observation[key]
            for key in self.policy.goal_keys
        }

        return (observation, goal)

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        full_observation = observation.copy()
        observation = {
            key: full_observation[key]
            for key, value in self.policy.observation_keys
        }
        goal = {
            key: full_observation[key]
            for key, value in self.policy.goal_keys
        }
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'goals': goal,
            'infos': info,
        }

        return processed_observation
