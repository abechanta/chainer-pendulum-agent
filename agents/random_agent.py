from agent import Agent

import numpy as np


class RandomAgent(Agent):
	def __init__(self, dim):
		self.actions = dim.output

	def act(self, state):
		return np.random.randint(self.actions)
