import numpy as np

from .simple_replay_pool import SimpleReplayPool
from flatten_dict import flatten, unflatten


def random_int_with_variable_range(mins, maxs):
    result = np.floor(np.random.uniform(mins, maxs)).astype(int)
    return result


class HindsightExperienceReplayPool(SimpleReplayPool):
    def __init__(self,
                 *args,
                 resample_fields=None,
                 her_strategy=None,
                 terminal_epsilon=0,
                 **kwargs):
        self._resample_fields = resample_fields
        # self._resample_fields_flat = flatten(resample_fields_flat)
        self._her_strategy = her_strategy
        self._terminal_epsilon = terminal_epsilon
        super(HindsightExperienceReplayPool, self).__init__(*args, **kwargs)

    def _relabel_batch(self, batch, indices, her_strategy):
        batch_size = indices.size
        batch['goal_resample_distances'] = np.full(
            (batch_size, 1), np.float('inf'))
        batch['resampled'] = np.zeros((batch_size, 1), dtype='bool')

        if her_strategy:
            her_strategy_type = self._her_strategy['type']
            goal_resampling_probability = self._her_strategy[
                'resampling_probability']

            resample_mask = (
                np.random.rand(batch_size) < goal_resampling_probability)
            where_resampled = np.where(resample_mask)
            resample_indices = indices[where_resampled]
            num_resamples = np.sum(resample_mask)

            # Everything here is relative to the original goal
            episode_first_distances = -1 * batch['episode_index_forwards'][
                where_resampled]
            episode_last_distances = batch['episode_index_backwards'][
                where_resampled]
            episode_first_indices = (
                resample_indices + episode_first_distances[..., 0]).astype(int)
            episode_last_indices = (
                resample_indices + episode_last_distances[..., 0]).astype(int)

            if her_strategy_type == 'random':
                goal_resample_indices = self.random_indices(num_resamples)
                goal_resample_distances = np.full(
                    (num_resamples, 1), np.float('inf'))

                in_same_episodes = np.logical_and(
                    episode_first_indices <= goal_resample_indices,
                    goal_resample_indices < episode_last_indices)
                where_same_episode = np.where(in_same_episodes)
                goal_resample_distances[
                    where_same_episode
                ] = (
                    goal_resample_indices[where_same_episode]
                    - resample_indices[where_same_episode]
                )[..., None]
            else:
                if her_strategy_type == 'final':
                    goal_resample_indices = episode_last_indices
                    goal_resample_distances = episode_last_distances
                elif her_strategy_type == 'episode':
                    goal_resample_distances = random_int_with_variable_range(
                        episode_first_distances, episode_last_distances)
                    goal_resample_indices = (
                        resample_indices + goal_resample_distances[..., 0])
                elif her_strategy_type == 'future':
                    goal_resample_distances = random_int_with_variable_range(
                        0, episode_last_distances)
                    goal_resample_indices = (
                        resample_indices + goal_resample_distances[..., 0])

            goal_resample_indices %= self._max_size
            goals_batch_flat = flatten(
                super(HindsightExperienceReplayPool, self)
                .batch_by_indices(
                    indices=goal_resample_indices,
                    field_name_filter=None))

            batch_flat = flatten(batch)
            for key in goals_batch_flat.keys():
                if key not in self._resample_fields: continue
                batch_flat[key][where_resampled] = goals_batch_flat[key]

            batch = unflatten(batch_flat)
            batch['goal_resample_distances'][where_resampled] = (
                goal_resample_distances)
            batch['resampled'][where_resampled] = True

        return batch

    def batch_by_indices(self, indices, *args, relabel=True, **kwargs):
        batch = super(HindsightExperienceReplayPool, self).batch_by_indices(
            indices, *args, **kwargs)
        if relabel:
            batch = self._relabel_batch(
                batch, indices, her_strategy=self._her_strategy)
        return batch
