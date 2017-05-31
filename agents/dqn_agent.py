from agent import Agent
from agents.dqn_trainer import DqnTrainer

import chainer
from chainer import Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np
import os


class QNetwork(Chain):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__(
            l1=L.Linear(input_dim, hidden_dim),
            l2=L.Linear(hidden_dim, output_dim),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.forward(x), y)

    def forward(self, x):
        h = F.relu(self.l1(x))
        predicted_y = self.l2(h)
        return predicted_y

    def hash(self):
        input_dim = self.l1.W.shape[1]
        return self.forward(Variable(np.zeros((1, input_dim), dtype=np.float32), volatile="on")).data


class DqnAgent(Agent):
    def __init__(self, dim, model_file, greedy=False):
        self.model = QNetwork(dim.input, 64, dim.output)
        self.dim = dim
        self.greedy = greedy
        self.epsilon = 0.05
        self.model_file = model_file
        self.epoch = 0
        self.load_model()

    def get_trainer(self):
        return DqnTrainer(self, steps_per_update=(10, 30))

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        print("epsilon={:.4f}".format(self.epsilon))

    def act(self, state):
        if (not self.greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.dim.output)
        x = Variable(state.reshape(1, -1).astype(np.float32))
        q = self.model.forward(x)
        return np.argmax(q.data[0])

    def load_model(self):
        model_path = os.path.dirname(self.model_file)
        if not os.path.exists(model_path):
            return False
        model_base = os.path.basename(self.model_file)
        *model_names, = filter(lambda fn: fn.startswith(model_base.partition(".")[0]), os.listdir(model_path))
        *epochs, = map(lambda fn: int(fn.partition(".")[2].partition(".")[0]), model_names)
        if len(epochs) == 0:
            return False
        epochs.sort()
        self.epoch = epochs[-1]     # extract largest epoch#

        model_name = self.model_file.format(self.epoch)
        serializers.load_npz(model_name, self.model)
        print("model {} loaded. hash={}".format(model_name, self.model.hash()))
        return True

    def save_model(self, epoch):
        model_path = os.path.dirname(self.model_file)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.epoch = epoch          # assign epoch#

        model_name = self.model_file.format(self.epoch)
        serializers.save_npz(model_name, self.model)
        print("model {} saved. hash={}".format(model_name, self.model.hash()))
