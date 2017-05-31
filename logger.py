from collections import deque
import numpy as np
import os
import pickle


class Logger():
    def __init__(self, log_file):
        self.best_reward = -1000000.000
        self.log = deque()
        self.log_file = log_file

    def __call__(self, data):
        episode, frame, _, _, episode_done = data
        if frame == len(self.log):
            self.log.append(data)

        if episode_done:
            total_reward = np.sum([l[3] for l in self.log])     # data.reward
            if self.best_reward < total_reward:
                print("episode {} achieves total reward {:.4f}.".format(episode, total_reward))
                self.best_reward = total_reward
                self.save(episode)
            self.log = deque()

    def save(self, episode):
        log_path = os.path.dirname(self.log_file)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        log_name = self.log_file.format(episode)
        with open(log_name, "wb") as lf:
            pickle.dump(self.log, lf)
