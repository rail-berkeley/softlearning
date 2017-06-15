from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides


class Bptt(RLAlgorithm):
    """
    Back propogation through time
    """
    def __init__(
            self,
    ):
        self.num_batches = None
        self.episode_length = None
        self.num_episodes_per_iteration = None
        self.replay_buffer = None
        self.env = None
        self.policy = None

    def _sample(self):
        for k in range(self.num_episodes_per_iteration):
            last_observation = self.env.reset()
            done = False
            for t in range(self.episode_length):
                action = self.policy.act(last_observation)
                observation, reward, done, _ = self.env.step(action)
                self.replay_buffer.add_sample(observation, action, reward, done)
                if t < self.episode_length - 1:
                    assert not done
                assert done
            assert done
            self.replay_buffer.finish_episode()

    @overrides
    def train(self):
        self._sample()
        batches = self.replay_buffer.sample(self.num_batches)
        self._bptt(batches)

    def _bptt(self, batches):
        # A numpy array holding the state of LSTM after each batch of words.
        total_loss = 0.0
        for current_batch_of_words in words_in_dataset:
            numpy_state, current_loss = session.run([final_state, loss],
                                                    # Initialize the LSTM state from the previous iteration.
                                                    feed_dict={
                                                        initial_state: numpy_state,
                                                        words: current_batch_of_words})
            total_loss += current_loss

