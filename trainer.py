from time import sleep


class GymTrainer:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.test = self.evaluate

    def train(self, episode, render=False, adjust_render_fps=True):
        _delay = self._get_delay(adjust_render_fps)
        #
        env, agent = self.env, self.agent
        result_reward = [0] * episode
        result_steps = [0] * episode
        for i in range(episode):
            state = env.reset()
            action = agent.start_episode(state)
            while True:
                if render:
                    env.render()
                    sleep(_delay)
                state, reward, done, info = env.step(action)
                result_reward[i] += reward
                result_steps[i] += 1
                if not done:
                    action = agent.step(state, reward)
                    continue
                else:
                    agent.end_episode(state, reward)
                    break
        return {'reward' : result_reward, 'steps' : result_steps}

    def evaluate(self, episode, render=False, adjust_render_fps=True):
        _delay = self._get_delay(adjust_render_fps)
        env, agent = self.env, self.agent
        result_reward = [0] * episode
        result_steps = [0] * episode
        for i in range(episode):
            state = env.reset()
            while True:
                if render:
                    env.render()
                    sleep(_delay)
                action = agent.select_best_action(state)
                state, reward, done, info = env.step(action)
                result_reward[i] += reward
                result_steps[i] += 1
                if done:
                    break
        return {'reward' : result_reward, 'steps' : result_steps}

    def _get_delay(self, adjust_render_fps):
        if type(adjust_render_fps) == int:
            return 1 / adjust_render_fps
        elif adjust_render_fps == True:
            return 1 / self.env.metadata.get('video.frames_per_second', 60)
        else:
            return 0

