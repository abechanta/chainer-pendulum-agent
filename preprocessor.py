import numpy as np


class Preprocessor():
    NB_STATE_HISTORY = 4

    def __init__(self, dim):
        self.state = np.zeros((Preprocessor.NB_STATE_HISTORY, dim.input // Preprocessor.NB_STATE_HISTORY))

    def reset(self, observation):
        self.state = np.zeros(self.state.shape)
        for _ in range(Preprocessor.NB_STATE_HISTORY):
            self.store_state(observation)

    def get_state(self):
        return self.state.copy()

    def store_state(self, observation):
        self.state[1:] = self.state[:-1]
        self.state[0] = self._state_from_obs(observation)
        return self.get_state()

    def _state_from_obs(self, observation):
        return observation
