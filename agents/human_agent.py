from agent import Agent

from collections import namedtuple
import pygame


Dpad = namedtuple("Dpad", "d, u, l, r")
Buttons = namedtuple("Buttons", "a, b, x, y, lb, rb, lt, rt")


class HumanAgent(Agent):
	def __init__(self, dim, idx=0):
		self.actions = dim.output
		self.dpad = Dpad._make([False] * 4)
		self.buttons = Buttons._make([False] * 8)
		pygame.init()
		pygame.joystick.init()
		try:
			pad = pygame.joystick.Joystick(idx)
			pad.init()
			self.pad = pad
			print('enabled gamepad: ', pad.get_name())
		except pygame.error:
			print('no gamepad found.')

	def act(self, state):
		events = [e.type for e in pygame.event.get()]

		if pygame.JOYAXISMOTION in events:
			tolerance = 0.300
			self.dpad = Dpad._make(sum([[i < -tolerance, i > tolerance] for i in map(self.pad.get_axis, range(2))], []))
			# print(self.dpad, [i for i in map(self.pad.get_axis, range(2))])

		if pygame.JOYBUTTONDOWN in events or pygame.JOYBUTTONUP in events:
			self.buttons = Buttons._make([self.pad.get_button(i) > 0 for i in range(8)])
			# print(self.buttons)

		# TODO: joystick.get_hat() not working for my legacy gamepad.
		# return 0 if self.dpad.l == self.dpad.r else 1 if self.dpad.r else 2
		return 0 if self.buttons.lb == self.buttons.rb else 1 if self.buttons.rb else 2
