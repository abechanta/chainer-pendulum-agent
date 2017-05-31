from collections import deque
from collections import namedtuple
import gym
from gym import wrappers
import time


class PendulumEnvironment():
    def __init__(self, render=False, debug=False, record_path=""):
        self.env = gym.make("Pendulum-v0")
        if record_path:
            self.env = wrappers.Monitor(self.env, record_path)
        self.actions = [[0.001], self.env.action_space.low, self.env.action_space.high]
        if not debug:
            self.actions = self.actions[1:]
        print("actions=", self.actions)
        self.render = render
        self.debug = debug
        self.prev_frame_tick = time.time()
        self.frames = deque()

    def get_dim(self, nb_history=1):
        Dim = namedtuple("Dim", "input, output")
        return Dim(input=self.env.observation_space.shape[0] * nb_history, output=len(self.actions))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.render:
            self.env.render()
        obs, reward, episode_done, info = self.env.step(self.actions[action])
        if self.render:
            self._wait(0.500) if episode_done else self._sync()
        return obs, reward, episode_done, info

    def _wait(self, wait_time):
        start_time = time.time()
        while time.time() - start_time < wait_time:
            pass
        self.prev_frame_tick = time.time()

    def _sync(self, sec_per_frame=1.000/30):
        while time.time() - self.prev_frame_tick < sec_per_frame:
            pass
        if self.debug:
            self.frames.append(time.time() - self.prev_frame_tick)
            if len(self.frames) > 60:
                self.frames.popleft()
                # print("{:.4f}fps".format(len(self.frames) / sum(self.frames, 0.001)))
        self.prev_frame_tick = time.time()
