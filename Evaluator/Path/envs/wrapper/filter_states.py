import gym

from envs.state import ImageState


class ObsStateTmp(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        super(ObsStateTmp, self).__init__(env)

    def observation(self, states: ImageState):

        return [states.sensor_maps, states.vector_states, states.ped_maps]


class ObsLaserStateTmp(gym.Wrapper):
    def __init__(self, env, cfg):
        super(ObsLaserStateTmp, self).__init__(env)

    def reset(self, **kwargs):
        state, reward, done, info = self.env.reset(**kwargs)
        return self.observation(state), reward, done, info
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.observation(state), reward, done, info

    def observation(self, states: ImageState):
        return [states.lasers, states.vector_states]
