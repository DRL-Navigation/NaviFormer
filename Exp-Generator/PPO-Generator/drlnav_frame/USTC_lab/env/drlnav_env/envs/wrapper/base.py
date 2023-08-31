import logging
import gym
import numpy as np
import math
import time
import sys
import struct
import pymongo
import random

from typing import *
from collections import deque
from copy import deepcopy


from envs.state import ImageState
from envs.action import *
from envs.utils import BagRecorder


class StatePedVectorWrapper(gym.ObservationWrapper):
    avg = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0])
    std = np.array([6.0, 6.0, 0.6, 0.9, 0.50, 0.5, 6.0])

    def __init__(self, env, cfg=None):
        super(StatePedVectorWrapper, self).__init__(env)

    def observation(self, state: ImageState):
        self._normalize_ped_state(state.ped_vector_states)
        return state

    def _normalize_ped_state(self, peds):

        for robot_i_peds in peds:
            for j in range(int(robot_i_peds[0])): # j: ped index
                robot_i_peds[1 + j * 7:1 + (j + 1) * 7] = (robot_i_peds[1 + j * 7:1 + (j + 1) * 7] - self.avg) / self.std


class VelActionWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(VelActionWrapper, self).__init__(env)
        if cfg['discrete_action']:
            self.actions: DiscreteActions = DiscreteActions(cfg['discrete_actions'])

            self.f = lambda x: self.actions[int(x)] if np.isscalar(x) else ContinuousAction(*x)
        else:
            clip_range = cfg['continuous_actions']

            def tmp_f(x):
                y = []
                for i in range(len(x)):
                    y.append(np.clip(x[i], clip_range[i][0], clip_range[i][1]))
                return ContinuousAction(*y)
            # self.f = lambda x: ContinuousAction(*x)
            self.f = tmp_f

    def step(self, action: np.ndarray):
        action = self.action(action)
        state, reward, done, info = self.env.step(action)
        info['speeds'] = np.array([a.reverse()[:2] for a in action])
        return state, reward, done, info

    def action(self, actions: np.ndarray) -> List[ContinuousAction]:
        return list(map(self.f, actions))

    def reverse_action(self, actions):

        return actions


class MultiRobotCleanWrapper(gym.Wrapper):
    is_clean : list
    def __init__(self, env, cfg):
        super(MultiRobotCleanWrapper, self).__init__(env)
        self.is_clean = np.array([True] * cfg['agent_num_per_env'])

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        info['is_clean'] = deepcopy(self.is_clean)
        reward[~info['is_clean']] = 0
        info['speeds'][~info['is_clean']] = np.zeros(2)
        # for i in range(len(done)):
        #     if done[i]:
        #         self.is_clean[i]=False
        self.is_clean = np.where(done>0, False, self.is_clean)
        return state, reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.is_clean = np.array([True] * len(self.is_clean))
        return state



class StateBatchWrapper(gym.Wrapper):
    batch_dict: np.ndarray

    def __init__(self, env, cfg):
        print(cfg,flush=True)
        super(StateBatchWrapper, self).__init__(env)
        self.q_sensor_maps = deque([], maxlen=cfg['image_batch']) if cfg['image_batch']>0 else None
        self.q_vector_states = deque([], maxlen=cfg['state_batch']) if cfg['state_batch']>0 else None
        self.q_lasers = deque([], maxlen=cfg['laser_batch']) if cfg['laser_batch']>0 else None
        self.batch_dict = {
            "sensor_maps": self.q_sensor_maps,
            "vector_states": self.q_vector_states,
            "lasers": self.q_lasers,
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.batch_state(state), reward, done, info

    def _concate(self, b: str, t: np.ndarray):
        q = self.batch_dict[b]
        if q is None:
            return t
        else:
            t = np.expand_dims(t, axis=1)
        # start situation
        while len(q) < q.maxlen:
            q.append(np.zeros_like(t))
        q.append(t)
        #  [n(Robot), k(batch), 84, 84]
        return np.concatenate(list(q), axis=1)

    def batch_state(self, state):
        # TODO transpose. print
        state.sensor_maps = self._concate("sensor_maps", state.sensor_maps)
        # print('sensor_maps shape; ', state.sensor_maps.shape)

        # [n(robot), k(batch), state_dim] -> [n(robot), k(batch) * state_dim]
        tmp_ = self._concate("vector_states", state.vector_states)
        state.vector_states = tmp_.reshape(tmp_.shape[0], tmp_.shape[1] * tmp_.shape[2])
        # print("vector_states shape", state.vector_states.shape)
        state.lasers = self._concate("lasers", state.lasers)
        # print("lasers shape:", state.lasers.shape)
        return state

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.batch_state(state)


class SensorsPaperRewardWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(SensorsPaperRewardWrapper, self).__init__(env)

        self.laser_norm = cfg['laser_norm']
        self.laser_max = cfg['laser_max']
        self.cfg = cfg
        self.min_dist = []
        self.closer = []

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        self.min_dist = []
        self.closer = []
        for vs in states.vector_states:
            self.min_dist.append(math.sqrt(vs[0]**2+vs[1]**2))
            self.closer.append(False)
        return states

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        return states, self.reward(reward, states), done, info

    def _each_r(self, states: ImageState, index: int):
        collision_reward = reach_reward = distance_reward = explore_reward = 0.0
        distance_reward_factor = 400.0
        collision_reward_factor = -100.0

        min_dist = self.min_dist[index]
        closer = self.closer[index]
        vector_state = states.vector_states[index]
        dist_to_obs = min(list(states.lasers[index].reshape(-1))) * (self.laser_max if self.laser_norm else 1.0)
        is_collision = states.is_collisions[index]
        is_arrive = states.is_arrives[index]

        if is_collision > 0:
            collision_reward = -1000
        else:
            if dist_to_obs < 0.5+0.17:
                collision_reward = collision_reward_factor * (0.5+0.17-dist_to_obs)**2
            dist = math.sqrt(vector_state[0] ** 2 + vector_state[1] ** 2)
            if dist < 0.17 or is_arrive:
                reach_reward = 1000
            if dist < min_dist:
                distance_reward = distance_reward_factor * (min_dist - dist)**2
                self.min_dist[index] = dist
                self.closer[index] = True
            else:
                if closer == True:
                    explore_reward = -25
                self.closer[index] = False
            
        reward = collision_reward + reach_reward + distance_reward + explore_reward
        return reward

    def reward(self, reward, states):
        rewards = np.zeros(len(states))
        for i in range(len(states)):
            rewards[i] = self._each_r(states, i)

        return rewards


class NeverStopWrapper(gym.Wrapper):
    """
        NOTE !!!!!!!!!!!
        put this in last wrapper.
    """
    def __init__(self, env, cfg):
        super(NeverStopWrapper, self).__init__(env)

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        if info['all_down'][0]:
            states = self.env.reset(**info)

        return states, reward, done, info


# time limit
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeLimitWrapper, self).__init__(env)
        self._max_episode_steps = cfg['time_max']
        robot_total = cfg['robot']['total']
        self._elapsed_steps = np.zeros(robot_total, dtype=np.uint8)

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        done = np.where(self._elapsed_steps > self._max_episode_steps, 1, done)
        info['dones_info'] = np.where(self._elapsed_steps > self._max_episode_steps, 10, info['dones_info'])
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)###


class InfoLogWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(InfoLogWrapper, self).__init__(env)
        self.robot_total = cfg['robot']['total']
        self.tmp = np.zeros(self.robot_total, dtype=np.uint8)
        self.ped: bool = cfg['ped_sim']['total'] > 0 and cfg['env_type'] == 'robot_nav'

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        info['arrive'] = states.is_arrives
        info['collision'] = states.is_collisions

        info['dones_info'] = np.where(states.is_collisions > 0, states.is_collisions, info['dones_info'])
        info['dones_info'] = np.where(states.is_arrives == 1, 5, info['dones_info'])
        info['all_down'] = self.tmp + sum(np.where(done>0, 1, 0)) == len(done)

        if self.ped:
            # when robot get close to human, record their speeds.
            info['bool_get_close_to_human'] = np.where(states.ped_min_dists < 1 , 1 , 0)

        return states, reward, done, info


class BagRecordWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(BagRecordWrapper, self).__init__(env)
        self.bag_recorder = BagRecorder(cfg["bag_record_output_name"])
        self.record_epochs = int(cfg['bag_record_epochs'])
        self.episode_res_topic = "/" + cfg['env_name'] + str(cfg['node_id']) + "/episode_res"
        print("epi_res_topic", self.episode_res_topic, flush=True)
        self.cur_record_epoch = 0

        self.bag_recorder.record(self.episode_res_topic)

    def _trans2string(self, dones_info):
        o: List[str] = []
        for int_done in dones_info:
            if int_done == 10:
                o.append("stuck")
            elif int_done == 5:
                o.append("arrive")
            elif 0 < int_done < 4:
                o.append("collision")
            else:
                raise ValueError
        print(o, flush=True)
        return o

    def reset(self, **kwargs):
        if self.cur_record_epoch == self.record_epochs:
            time.sleep(10)
            self.bag_recorder.stop()
        if kwargs.get('dones_info') is not None: # first reset not need
            self.env.end_ep(self._trans2string(kwargs['dones_info']))
            self.cur_record_epoch += 1
        """
                done info:
                10: timeout
                5:arrive
                1: get collision with static obstacle
                2: get collision with ped
                3: get collision with other robot
                """
        print(self.cur_record_epoch, flush=True)
        return self.env.reset(**kwargs)


class TimeControlWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeControlWrapper, self).__init__(env)
        self.dt = cfg['control_hz']

    def step(self, action):
        start_time = time.time()
        states, reward, done, info = self.env.step(action)
        while time.time() - start_time < self.dt:
            time.sleep(0.02)
        return states, reward, done, info

class ExpCollectWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(ExpCollectWrapper, self).__init__(env)
        self.node_id = cfg['node_id']
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.database = self.client['DataBase']
        self.collection = self.database['Collection']
        self.num = 0
        self.sum = cfg['exp_sum']
        self.exp_id_bias = cfg['exp_id_bias']
        self.exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        self.finish = False

    @classmethod
    def __decode_data(cls, bytes_data: bytes) -> List[np.ndarray]:
        type_dict = {
            1 : (1, np.uint8),
            2 : (2, np.float16),
            3 : (4, np.float32),
            4 : (8, np.float64),
        }
        index, length = 0, len(bytes_data)
        list_np_data = []
        while index < length:
            c_type = struct.unpack(">h", bytes_data[index: index+2])[0]
            tb, t = type_dict[c_type]
            count, shape_dim = struct.unpack(">II", bytes_data[index+2: index+10])
            shape = struct.unpack(">" + "I" * shape_dim,
                                  bytes_data[index + 10: index + 10 + shape_dim * 4])
            index = index + 10 + shape_dim * 4
            each_bytes_data = bytes_data[index: index + count*tb]
            list_np_data.append(np.frombuffer(each_bytes_data, dtype=t).reshape(*shape))
            index += count*tb
        return list_np_data

    @classmethod
    def __encode_data(cls, np_list_data: List[np.ndarray]) -> bytes:
        def type_encode(data_dtype: np.dtype):
            if data_dtype == np.uint8:
                return 1
            elif data_dtype == np.float16:
                return 2
            elif data_dtype == np.float32:
                return 3
            elif data_dtype == np.float64:
                return 4
            else:
                logging.log(logging.ERROR, "Match data type error !")
                raise ValueError
        bytes_out = b""
        for np_data in np_list_data:
            shape = np_data.shape
            shape_dim = len(shape)
            count = 1
            for shape_i in shape:
                count *= shape_i
            bytes_out += struct.pack(">h", type_encode(np_data.dtype))
            bytes_out += struct.pack(">II", count, shape_dim)
            bytes_out += struct.pack(">" + "I" * shape_dim, *shape)
            bytes_out += np_data.tobytes()
        return bytes_out

    def reset(self, **kwargs):
        if self.finish == False:
            if self.sum > self.num:
                if len(self.exp['action']) > 0:
                    self.exp['action'].append(np.zeros_like(self.exp['action'][-1]))
                    self.exp['reward'].append(np.zeros_like(self.exp['reward'][-1]))
                    self.collection.insert_one({'_id' : int(self.node_id*self.sum+self.num+self.exp_id_bias),
                                                'laser' : ExpCollectWrapper.__encode_data(self.exp['laser']),
                                                'vector' : ExpCollectWrapper.__encode_data(self.exp['vector']),
                                                'action' : ExpCollectWrapper.__encode_data(self.exp['action']),
                                                'reward' : ExpCollectWrapper.__encode_data(self.exp['reward'])
                                                })
                    self.num += 1
                    if self.num % 10000 == 0:
                        print("[{}]Got {}W Exp from Node {}".format(time.strftime('%H:%M:%S',time.localtime(time.time())), int(self.num/10000), self.node_id), flush=True)
            else:
                self.client.close()
                self.database = self.collection = None
                self.finish = True
                print("[{}]Node {} Finish Exp Generate .".format(self.node_id, time.strftime('%H:%M:%S',time.localtime(time.time()))), flush=True)
        states = self.env.reset(**kwargs)
        self.exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        self.exp['laser'].append(states[0])
        self.exp['vector'].append(states[1])
        return states

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        if self.finish == False:
            self.exp['action'].append(info['feedback_action'])
            self.exp['reward'].append(reward.reshape(reward.shape+(1,)))
            self.exp['laser'].append(states[0])
            self.exp['vector'].append(states[1])
        return states, reward, done, info

class ActionRandomWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(ActionRandomWrapper, self).__init__(env)
        self.range = cfg["discrete_actions"]
        self.turn_on = cfg["random_action"]
        self.rate = cfg["random_rate"]

    def random_action(self, robot_action):
        random_action = self.range[random.randint(0, len(self.range)-1)]
        return ContinuousAction(*random_action)

    @classmethod
    def unzip_action(cls, vector_states):
        return [vector_states[-2], vector_states[-1]]

    def step(self, action):
        if self.turn_on:
            if random.random() < self.rate:
                action = list(map(self.random_action, action))
        state, reward, done, info = self.env.step(action)
        info['feedback_action'] = np.array(list(map(ActionRandomWrapper.unzip_action, state.vector_states.tolist())))
        return state, reward, done, info