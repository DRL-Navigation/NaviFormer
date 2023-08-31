from envs import make_env, read_yaml
from nn import NaviGPT, GPT2Config
from pursuit import Pursuit
from copy import deepcopy

import torch, numpy, sys, os, math

def vector_to_location(vector):
    a = vector[0][0]
    b = vector[0][1]
    yaw = vector[0][2]
    x = -a*math.cos(yaw)-b*math.sin(yaw)
    y = a*math.sin(yaw)-b*math.cos(yaw)
    return numpy.array([[x, y, (-yaw)%(2*math.pi)]])

def path_rtg_estimate(states, laser_norm=10.0):
    def angle_from_negative_pi_to_pi(angle):
        if angle > math.pi:
            angle = angle - 2*math.pi*math.ceil(angle/(2*math.pi))
        elif angle < -math.pi:
            angle = angle - 2*math.pi*math.floor(angle/(2*math.pi))
        return angle

    location = vector_to_location(states[1])[0]
    theta = math.atan2(location[1], location[0]) % (2*math.pi)
    theta = angle_from_negative_pi_to_pi((theta - location[2] + math.pi) % (2*math.pi))
    rtg = 0.0
    if math.fabs(theta) > math.pi/4:
        rtg = 0.0
    else:
        if theta >= 0:
            theta = max(0.0, theta - 0.9*0.5)
        else:
            theta = min(0.0, theta + 0.9*0.5)
        rtg = 4 * (math.cos(theta)/4)**2 * 400.0
        obs_dist = min(list(states[0][0][0]*laser_norm))
        if obs_dist < 0.5+0.17:
            rtg -= 4 * (0.5+0.17-obs_dist)**2 * 100.0
        if location[0]**2+location[1]**2 < 1.0**2:
            rtg += 1000.0
    return numpy.array([[rtg]])

def seq_to_torch(location_seq, laser_seq, rtg_seq, path_seq):
    location_torch = torch.from_numpy(numpy.concatenate(location_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    laser_torch = torch.from_numpy(numpy.concatenate(laser_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    rtg_torch = torch.from_numpy(numpy.concatenate(rtg_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    path_torch = torch.from_numpy(numpy.concatenate(path_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    return location_torch, laser_torch, rtg_torch, path_torch

def seq_reset(states):
    location_seq, laser_seq = [vector_to_location(states[1])], [states[0]]
    rtg_seq = [path_rtg_estimate(states)]
    path_seq = [numpy.zeros(shape=(1, 4), dtype=float)]
    return location_seq, laser_seq, rtg_seq, path_seq

def index_to_path(index, location):
    index = list(index)
    l = [location[0][0], location[0][1], location[0][2]]
    path = []
    for i in index:
        t = (i % 19)*0.0125+(-0.9*0.25*0.5)
        r = math.floor(i / 19) * 0.025
        _x = r*math.cos(t)
        _y = r*math.sin(t)
        l[0] = l[0] + _x*math.cos(l[2]) - _y*math.sin(l[2])
        l[1] = l[1] + _x*math.sin(l[2]) + _y*math.cos(l[2])
        l[2] = (l[2] + (2*t)) % (2*math.pi)
        path.append(deepcopy(l))
    return path

def get_obs(state, laser_norm=10.0):
    obs_list = []
    location = vector_to_location(state[1])[0]
    scan = state[0][0][0]*laser_norm
    ped_map = state[2][0][0]

    range_total = len(scan) // 10
    angle_step = math.pi / range_total
    cur_angle = -math.pi / 2
    for i in range(range_total):
        j = i * 10
        if scan[j] < 1.0:
            _x = scan[j] * math.cos(cur_angle)
            _y = scan[j] * math.sin(cur_angle)
            x = location[0] + _x*math.cos(location[2]) - _y*math.sin(location[2])
            y = location[1] + _x*math.sin(location[2]) + _y*math.cos(location[2])
            obs_list.append([x, y, 0])
        cur_angle += angle_step

    # indices = numpy.argwhere(ped_map == 1)
    # for ij in indices:
    #     _x = 3 - (ij[0]+1)*0.125
    #     if _x < 0: continue
    #     _y = 3 - (ij[1]+1)*0.125
    #     x = location[0] + _x*math.cos(location[2]) - _y*math.sin(location[2])
    #     y = location[1] + _x*math.sin(location[2]) + _y*math.cos(location[2])
    #     obs_list.append([x, y, 1])

    return obs_list

def safe_check(path, obs_list, safe_dist=0.20):
    def distance(location1, location2 = [0., 0.]):
        return math.sqrt((location1[0]-location2[0])**2+(location1[1]-location2[1])**2)

    def min_distance_from_obs(point, obs_list):
        dist_func = lambda obs : distance(point, obs)
            # if obs[2] == 0 else distance(point, obs)/2
        dist_list = list(map(dist_func, obs_list))
        return min(dist_list) if len(dist_list) != 0 else 10.0
    
    if path == None: return False
    if min_distance_from_obs(path[0], obs_list) <= safe_dist or \
        min_distance_from_obs(path[1], obs_list) <= safe_dist or \
        min_distance_from_obs(path[2], obs_list) <= safe_dist or \
        min_distance_from_obs(path[3], obs_list) <= safe_dist:
        return False
    return True


output = './output/'
logname = '10obs-5ped-138M.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = read_yaml('envs/cfg/test.yaml')
max_len = cfg['max_len']
env = make_env(cfg)
GPT_cfg = GPT2Config(n_embd=cfg['token_dim'], n_layer=cfg['nlayer'], n_head=cfg['nhead'], n_inner=cfg['ninner'], resid_pdrop=cfg['dropout'])
net = NaviGPT(GPT_cfg).cuda().float().eval()
net.load_state_dict(torch.load('../../Trainer/NaviFormer/log/138M/last_model.pt'))
pursuit = Pursuit(cfg)

states = env.reset()
while True:
    location_seq, laser_seq, rtg_seq, path_seq = seq_reset(states)
    pursuit.reset()
    
    while True:
        location_torch, laser_torch, rtg_torch, path_torch = seq_to_torch(location_seq, laser_seq, rtg_seq, path_seq)

        path = None
        path_topk = [0, 0, 0, 0]
        obs_list = get_obs(states)
        while not safe_check(path, obs_list):
            path_index = net.pred_path(location_torch, laser_torch, path_torch, rtg_torch, path_topk).cpu().numpy()
            path = index_to_path(path_index, location_seq[-1])
            path_topk[0] += 1
            if path_topk[0] > 1:
                if path_topk[0] > 20: 
                    path_topk = [0, 0, 0, 0]
                    last_w = action[0][1]
                    action = numpy.array([[0., 0.9*last_w/math.fabs(last_w) if last_w != 0.0 else 0.9]])
                    for i in range(8):
                        if i == 7: action = numpy.array([[0., 0.]])
                        states, reward, done, info = env.step(action, path=numpy.array(path).reshape((1, 4, 3)))
                        if info['all_down'][0]: break
                    obs_list = get_obs(states)
                location_seq, laser_seq, rtg_seq, path_seq = seq_reset(states)
                location_torch, laser_torch, rtg_torch, path_torch = seq_to_torch(location_seq, laser_seq, rtg_seq, path_seq)

        action = pursuit.action(path, states)
        states, reward, done, info = env.step(action, path=numpy.array(path).reshape((1, 4, 3)))
        if info['all_down'][0]: break

        if len(rtg_seq) >= max_len:
            location_seq.pop(0)
            laser_seq.pop(0)
            rtg_seq.pop(0)
            path_seq.pop(0)
        location_seq.append(vector_to_location(states[1]))
        laser_seq.append(states[0])
        rtg_seq.append(path_rtg_estimate(states))
        path_seq.insert(-1, path_index.reshape((1, 4)))