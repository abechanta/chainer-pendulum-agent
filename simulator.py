from preprocessor import Preprocessor


class Simulator():
    def __init__(self, environment, agent, train, action_freq=1):
        self.env = environment
        self.agent = agent
        self.prep = Preprocessor(self.env.get_dim(Preprocessor.NB_STATE_HISTORY))
        self.trainer = agent.get_trainer() if train else None
        self.action_freq = action_freq

    def _begin(self):
        obs = self.env.reset()
        if self.trainer:
            self.trainer.begin()
        self.prep.reset(obs)

    def _end(self, total_reward):
        if self.trainer:
            self.trainer.end(total_reward)

    def _tick(self, action):
        state = self.prep.get_state()
        if not action:
            action = self.agent.act(state)
        obs, reward, episode_done, info = self.env.step(action)
        state_next = self.prep.store_state(obs)
        if self.trainer:
            self.trainer.train(state, action, reward, episode_done, state_next)
        return action, obs, reward, episode_done, info

    def run(self, episodes, frames_per_episode=300):
        for e in range(episodes):
            self._begin()
            total_reward = 0.000
            for f in range(frames_per_episode):
                action = None if f % self.action_freq == 0 else action
                action, obs, reward, episode_done, info = self._tick(action)
                total_reward += reward
                if episode_done or (f == frames_per_episode - 1):
                    if episode_done:
                        print("episode finished after {} frames.".format(f + 1))
                    break
                yield e, f, action, reward, episode_done
            self._end(total_reward)
            yield e, f, action, reward, episode_done or (f == frames_per_episode - 1)
