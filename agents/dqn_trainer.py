from trainer import Trainer

import chainer
from chainer import Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import deque
import copy
import numpy as np


REPLAY_CAPACITY = 2048
MINI_BATCH_SIZE = 64
GAMMA = 0.990


def epsilon(epoch):
    return 1 / (1 + np.log(1 + epoch / 1000.00))


def sigmoid(x):
    return 1 / (1 + np.exp(-x / 100) + 0.001)


def tanh(x):
    ex1 = np.exp(+x / 100)
    ex2 = np.exp(-x / 100)
    return (ex1 - ex2) / (ex1 + ex2)


class DqnTrainer(Trainer):
    def __init__(self, agent, steps_per_update, episodes_per_record=10):
        self.agent = agent
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(agent.model)
        self.target_model = copy.deepcopy(self.agent.model)
        self.steps_per_update_model = steps_per_update[0]
        self.steps_per_update_target = steps_per_update[1]
        self.episodes_per_record = episodes_per_record
        self.replay_memory = deque()
        self.epoch = self.agent.epoch
        self.episode = 0
        self.step = 1

    def begin(self):
        # self.agent.set_epsilon(epsilon(self.epoch))
        pass

    def end(self, total_reward):
        self.episode += 1
        if self.episode % self.episodes_per_record == 0:
            self.agent.save_model(self.epoch)

    def train(self, state, action, reward, episode_done, state_next):
        if self.step % self.steps_per_update_model == 0:
            if self._experience(state, action, reward, episode_done, state_next):
                self._update_model()
                # self.agent.set_epsilon(epsilon(self.epoch))
        if self.step % self.steps_per_update_target == 0:
            self._update_target()
        self.step += 1

    def _experience(self, state, action, reward, episode_done, state_next):
        self.replay_memory.append((state, action, reward, episode_done, state_next))
        if len(self.replay_memory) > REPLAY_CAPACITY:
            self.replay_memory.popleft()
        return len(self.replay_memory) >= MINI_BATCH_SIZE

    def _update_model(self):
        batch = np.random.permutation(np.array(self.replay_memory))
        for i in range(0, len(batch), MINI_BATCH_SIZE):
            if i + MINI_BATCH_SIZE > len(batch):
                break
            self.agent.model.zerograds()
            mini_batch = batch[i : i + MINI_BATCH_SIZE]
            loss = self._forward(mini_batch)
            loss.backward()
            self.optimizer.update()
        self.epoch += 1

    def _forward(self, batch):
        state = *map(np.array, batch[:,0]),
        predicted_qv = self.agent.model.forward(Variable(np.array(state).astype(np.float32)))
        target_qv = predicted_qv.data.copy()
        state_next = *map(np.array, batch[:,4]),
        qv = self.target_model.forward(Variable(np.array(state_next).astype(np.float32)))
        qv = np.max(qv.data, axis=1)
        for i, action, reward, episode_done in zip(
            range(len(batch)), batch[:,1], tanh(batch[:,2].astype(np.float32)), batch[:,3]
        ):
            target_qv[i, action] = reward + (0 if episode_done else GAMMA * qv[i])
        loss = F.mean_squared_error(predicted_qv, Variable(target_qv))
        return loss

    def _update_target(self):
        self.target_model = copy.deepcopy(self.agent.model)
