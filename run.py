from agent import Agent
from agents.dqn_agent import DqnAgent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from environment import PendulumEnvironment
from logger import Logger
from preprocessor import Preprocessor
from simulator import Simulator

import argparse
import gym
import os
import sys


MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.trained/pendulum.{}.npz")
LOG_FILE = os.path.join(os.path.dirname(__file__), "model.log/pendulum.{}.pkl")
REC_FILE = os.path.join(os.path.dirname(__file__), "model.log/pendulum-experiment-1")


def train(render, episodes):
    env = PendulumEnvironment(render=render, record_path=REC_FILE)
    agent = DqnAgent(env.get_dim(Preprocessor.NB_STATE_HISTORY), model_file=MODEL_FILE)
    simulator = Simulator(env, agent, train=True)

    logger = Logger(log_file=LOG_FILE)
    for e, f, action, reward, episode_done in simulator.run(episodes):
        logger((e, f, action, reward, episode_done))


def test(render, episodes):
    env = PendulumEnvironment(render=render)
    agent = DqnAgent(env.get_dim(Preprocessor.NB_STATE_HISTORY), model_file=MODEL_FILE, greedy=True)
    simulator = Simulator(env, agent, train=False)
    # env = PendulumEnvironment(render=render, debug=True)
    # agent = HumanAgent(env.get_dim())
    # # agent = RandomAgent(env.get_dim())
    # simulator = Simulator(env, agent, train=False)

    episode, total_reward, best_reward = 0, 0.000, -1000000
    for e, f, action, reward, episode_done in simulator.run(episodes, frames_per_episode=5*60*60):
        total_reward += reward
        if episode_done:
            print("episode {} achieves total reward {:.4f}.".format(episode, total_reward))
            episode, total_reward, best_reward = episode + 1, 0.000, max(total_reward, best_reward)
    print("best reward {:.4f}.".format(best_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pendulum Agent with DQN")
    parser.add_argument("--episodes", type=int, default=10, help="number of episodes to run")
    parser.add_argument("--render", action="store_const", const=True, default=False, help="render or not")
    parser.add_argument("--train", action="store_const", const=True, default=False, help="train or not")
    args = parser.parse_args()
    if args.train:
        train(args.render, args.episodes)
    else:
        test(args.render, args.episodes)
