import numpy as np
import math
from copy import deepcopy

class DWA_Pursuit:
    def __init__(self, cfg):
        self.v_window = [0.0, 1.0]
        self.w_window = [-0.9, 0.9]
        self.acc_v = 5.0
        self.acc_w = 4.0
        self.resol_v = 0.1
        self.resol_w = 0.1
        self.dt = cfg['control_hz']

        self.robot_r = cfg['robot_radius']
        self.inf_r = 0.5

        self.v_weight = 0.2
        self.g_weight = 1.0
        self.o_weight = 1.0

        self.last_v = 0.0
        self.last_w = 0.0

    @classmethod
    def vector_to_location(cls, vector):
        a = vector[0][0]
        b = vector[0][1]
        yaw = vector[0][2]
        x = -a*math.cos(yaw)-b*math.sin(yaw)
        y = a*math.sin(yaw)-b*math.cos(yaw)
        return [x, y, (-yaw)%(2*math.pi)]

    @classmethod
    def distance(cls, location1, location2 = [0., 0.]):
        return math.sqrt((location1[0]-location2[0])**2+(location1[1]-location2[1])**2)

    def reset(self):
        self.last_v = self.last_w = 0.0

    def scan_to_obs(self, location, scan, ped_map):
        obs_list = []
        range_total = scan.shape[0] // 10
        angle_step = math.pi / range_total
        cur_angle = -math.pi/2
        for i in range(range_total):
            j = i * 10
            if scan[j] < 1.0:
                _x = scan[j] * math.cos(cur_angle)
                _y = scan[j] * math.sin(cur_angle)
                x = location[0] + _x*math.cos(location[2]) - _y*math.sin(location[2])
                y = location[1] + _x*math.sin(location[2]) + _y*math.cos(location[2])
                obs_list.append([x, y, 0])
            cur_angle += angle_step

        indices = np.argwhere(ped_map == 1)
        for ij in indices:
            _x = 3 - (ij[0]+1)*0.125
            _y = 3 - (ij[1]+1)*0.125
            if DWA_Pursuit.distance([_x, _y]) > 2: continue
            x = location[0] + _x*math.cos(location[2]) - _y*math.sin(location[2])
            y = location[1] + _x*math.sin(location[2]) + _y*math.cos(location[2])
            obs_list.append([x, y, 1])
        return obs_list

    def calc_dynamic_window(self):
        vs = self.v_window
        vs.extend(self.w_window)
        vd = [self.last_v - self.acc_v * self.dt,
              self.last_v + self.acc_v * self.dt,
              self.last_w - self.acc_w * self.dt,
              self.last_w + self.acc_w * self.dt]
        vr = [math.ceil(max(vs[0], vd[0])/self.resol_v)*self.resol_v, 
              math.floor(min(vs[1], vd[1])/self.resol_v)*self.resol_v,
              math.ceil(max(vs[2], vd[2])/self.resol_w)*self.resol_w, 
              math.floor(min(vs[3], vd[3])/self.resol_w)*self.resol_w]
        return vr

    def calc_reach(self, location, vt, wt):
        reach = deepcopy(location)
        t = wt * self.dt / 2
        if wt != 0.0: 
            r = 2 * vt / wt * math.sin(t)
            _x = r*math.cos(t)
            _y = r*math.sin(t)
        else:
            _x = vt * self.dt
            _y = 0.0
        reach[0] = reach[0] + _x*math.cos(reach[2]) - _y*math.sin(reach[2])
        reach[1] = reach[1] + _x*math.sin(reach[2]) + _y*math.cos(reach[2])
        reach[2] = (reach[2] + (2*t)) % (2*math.pi)
        return reach

    def cost_v_w(self, vt, wt):
        return abs(vt-self.last_v) + 0.2*abs(wt-self.last_w)

    def cost_goal(self, location, reach, path):
        if DWA_Pursuit.distance(path[-1], path[0]) > 2*DWA_Pursuit.distance(path[0]):
            pathi = [0.0, 0.0, path[0][2]]
        else:
            pathi = path[0]
        dist = DWA_Pursuit.distance(reach, pathi)
        long = min(DWA_Pursuit.distance(reach, location), DWA_Pursuit.distance(pathi, location))
        yaw = abs(reach[2]-pathi[2])
        return dist - 2*long + 0.2*yaw
    
    def cost_obs(self, reach, obs_list):
        dist_func = lambda obs : DWA_Pursuit.distance(reach, obs) + (0.0 if obs[2] == 0 else -0.25)
        dist = list(map(dist_func, obs_list))
        cost_obs = -self.inf_r
        if len(dist) > 0:
            dist = min(dist)
            if dist <= self.robot_r:
                cost_obs = math.inf
            elif dist <= self.robot_r + self.inf_r:
                cost_obs = self.robot_r-dist
        return 2*cost_obs

    def sample_choice(self, vr, location, path, scan, ped_map):
        choice = []
        for vt in np.arange(vr[0], vr[1]+0.001, self.resol_v):
            vt = round(vt, 2)
            for wt in np.arange(vr[2], vr[3]+0.001, self.resol_w):
                wt = round(wt, 2)
                reach = self.calc_reach(location, vt, wt)
                cost_v = self.cost_v_w(vt, wt)
                cost_goal = self.cost_goal(location, reach, path)
                cost_obs = self.cost_obs(reach, self.scan_to_obs(location, scan, ped_map))
                choice.append([vt, wt, cost_v, cost_goal, cost_obs])
        return choice

    def action(self, path, state, laser_norm=10.0):
        location = DWA_Pursuit.vector_to_location(state[1])
        choice = self.sample_choice(self.calc_dynamic_window(), location, path, state[0][0][0]*laser_norm, state[2][0][0])
        score_list = []
        score = np.array(choice)
        for ii in range(0, len(score[:, 0])):
            weights = np.mat([self.v_weight, self.g_weight, self.o_weight])
            scoretemp = weights * (np.mat(score[ii, 2:5])).T
            score_list.append(scoretemp)
        max_score_id = np.argmin(score_list)
        action = list(score[max_score_id, 0:2])
        self.last_v = action[0]
        self.last_w = action[1]
        return np.array([[self.last_v, self.last_w]])
