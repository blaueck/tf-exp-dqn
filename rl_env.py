import retro
import numpy as np
import cv2


class Env:

    def __init__(self, name, n_skip=4, is_pooling=True, **kwarg):
        self.n_skip = n_skip
        self.env = retro.make(
            name, use_restricted_actions=retro.Actions.DISCRETE,
            **kwarg)
        self.action_space = self.env.action_space
        self.n_action = self.env.action_space.n
        self.is_pooling = is_pooling

        self.name = name

    def preprocess(self, pre_frame, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self.is_pooling and pre_frame is not None:
            pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_RGB2GRAY)
            np.maximum(frame, pre_frame, out=frame)

        frame = cv2.resize(frame, (84, 84))
        return frame

    def step(self, action):
        reward_sum = 0
        pre_ob, ob = None, None
        for _ in range(self.n_skip):
            pre_ob = ob
            ob, r, done, info = self.env.step(action)
            reward_sum += r
            if done:
                break
        ob = self.preprocess(pre_ob, ob)
        return ob, reward_sum, done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def reset(self):
        return self.preprocess(None, self.env.reset())

    def close(self):
        self.env.close()


class StateRecorder:

    def __init__(self):
        self.pre_state = None
        self.state = np.zeros([4, 84, 84], dtype='uint8')

    def reset(self):
        self.state[:] = 0

    def add(self, ob):
        self.pre_state = self.state
        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1, ...] = ob
