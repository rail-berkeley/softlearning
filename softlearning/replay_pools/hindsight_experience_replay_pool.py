import numpy as np

from .goal_replay_pool import GoalReplayPool
from flatten_dict import flatten, unflatten


def random_int_with_variable_range(mins, maxs):
    result = np.floor(np.random.uniform(mins, maxs)).astype(int)
    return result


class ResamplingReplayPool(GoalReplayPool):
    def _resample_indices(self,
                          indices,
                          episode_first_distances,
                          episode_last_distances,
                          resampling_strategy):
        """Compute resampled indices for given indices.

        Given indices of a batch (`indices`), and distances to the
          extremes of the corresponding episodes
          (`episode_{first,last}_distances`) compute new resampled indices
          using the given `resampling_strategy`.

        Args:
          indices: absolute indices of the samples we wish to resample batch
            for.
          episode_first_distances: distance (non-positive integer) to the
            first episode observation present in the pool for each index.
          episode_last_distances: distance (positive integer) to the last
            episode observation present in the pool for each index.
          resampling_strategy: One of:
            random: Sample randomly from the whole pool.
            final: For each index, sample the last observation from the
              corresponding episode.
            episode: For each index, sample any observation from the
              corresponding episode (could be from the past or the future).
            future: For each index, sample any observation from the
              corresponding episode's future.

        Returns:
          resample_indices: indices that can be used to fetch the resampled
            data using `self.batch_by_indices`.
          resample_distances: distances between the given indices and the
            resampled indices.
              If 0: the resampled index is the same as the original.
              If positive integer: the resampled index is from the future of
                the same episode as the original.
              If negative integer: the resample index is from the past of the
                same episode as the original.
              If inf: the resampled index is from other episode than the
                original.
        """
        num_resamples = indices.size
        episode_first_indices = (
            indices + episode_first_distances[..., 0]).astype(int)
        episode_last_indices = (
            indices + episode_last_distances[..., 0]).astype(int)

        if resampling_strategy == 'random':
            resample_indices = self.random_indices(num_resamples)
            resample_distances = np.full(
                (num_resamples, 1), np.float('inf'))

            in_same_episodes = np.logical_and(
                episode_first_indices <= resample_indices,
                resample_indices < episode_last_indices)
            where_same_episode = np.flatnonzero(in_same_episodes)
            resample_distances[
                where_same_episode
            ] = (
                resample_indices[where_same_episode]
                - indices[where_same_episode]
            )[..., None]
        else:
            if resampling_strategy == 'final':
                resample_indices = episode_last_indices
                resample_distances = episode_last_distances
            elif resampling_strategy == 'episode':
                resample_distances = random_int_with_variable_range(
                    episode_first_distances, episode_last_distances)
                resample_indices = (
                    indices + resample_distances[..., 0])
            elif resampling_strategy == 'future':
                resample_distances = random_int_with_variable_range(
                    0, episode_last_distances)
                resample_indices = (
                    indices + resample_distances[..., 0])

        resample_indices %= self._size

        return resample_indices, resample_distances


def REPLACE_FULL_OBSERVATION(original_batch,
                             resampled_batch,
                             where_resampled,
                             environment):
    batch_flat = flatten(original_batch)
    resampled_batch_flat = flatten(resampled_batch)
    goal_keys = [
        key for key in batch_flat.keys()
        if key[0] == 'goals'
    ]
    for key in goal_keys:
        assert (batch_flat[key][where_resampled].shape
                == resampled_batch_flat[key].shape)
        batch_flat[key][where_resampled] = (
            resampled_batch_flat[key])

    return unflatten(batch_flat)


class HindsightExperienceReplayPool(ResamplingReplayPool):
    def __init__(self,
                 *args,
                 her_strategy=None,
                 update_batch_fn=REPLACE_FULL_OBSERVATION,
                 reward_fn=None,
                 terminal_fn=None,
                 **kwargs):
        self._her_strategy = her_strategy
        self._update_batch_fn = update_batch_fn
        self._reward_fn = reward_fn or (
            lambda original_batch, *args: original_batch['rewards'])
        self._terminal_fn = terminal_fn or (
            lambda original_batch, *args: original_batch['terminals'])
        super(HindsightExperienceReplayPool, self).__init__(*args, **kwargs)

    def _relabel_batch(self, batch, indices, her_strategy):
        batch_size = indices.size
        batch['resampled_distances'] = np.full(
            (batch_size, 1), np.float('inf'))
        batch['resampled'] = np.zeros((batch_size, 1), dtype='bool')

        if her_strategy:
            her_strategy_type = self._her_strategy['type']
            goal_resampling_probability = self._her_strategy[
                'resampling_probability']

            to_resample_mask = (
                np.random.rand(batch_size) < goal_resampling_probability)
            where_resampled = np.flatnonzero(to_resample_mask)
            to_resample_indices = indices[where_resampled]

            episode_first_distances = -1 * batch['episode_index_forwards'][
                where_resampled]
            episode_last_distances = batch['episode_index_backwards'][
                where_resampled]

            resampled_indices, resampled_distances = self._resample_indices(
                to_resample_indices,
                episode_first_distances,
                episode_last_distances,
                her_strategy_type)

            resampled_batch = super(
                HindsightExperienceReplayPool, self
            ).batch_by_indices(
                indices=resampled_indices,
                field_name_filter=None)

            batch['resampled_distances'][where_resampled] = (
                resampled_distances)
            batch['resampled'][where_resampled] = True

            batch = self._update_batch_fn(
                batch, resampled_batch, where_resampled, self._environment)

            if self._reward_fn:
                self._reward_fn(
                    batch, resampled_batch, where_resampled, self._environment)
            if self._terminal_fn:
                self._terminal_fn(
                    batch, resampled_batch, where_resampled, self._environment)

        return batch

    def batch_by_indices(self, indices, *args, relabel=True, **kwargs):
        batch = super(HindsightExperienceReplayPool, self).batch_by_indices(
            indices, *args, **kwargs)
        if relabel:
            batch = self._relabel_batch(
                batch, indices, her_strategy=self._her_strategy)
        return batch

    def last_n_batch(self, *args, relabel=False, **kwargs):
        return super(HindsightExperienceReplayPool, self).last_n_batch(
            *args, relabel=False, **kwargs)
