from envs import make_env, read_yaml
from dwa import DWAPolicy4Nav
from astar import get_global_path
from nn import NaviGPT, GPT2Config
from copy import deepcopy
import sys, os, torch, numpy, math

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

def seq_reset(states):
    location_seq, laser_seq = [vector_to_location(states[1])], [states[0]]
    rtg_seq = [path_rtg_estimate(states)]
    path_seq = [numpy.zeros(shape=(1, 4), dtype=float)]
    return location_seq, laser_seq, rtg_seq, path_seq

def seq_to_torch(location_seq, laser_seq, rtg_seq, path_seq):
    location_torch = torch.from_numpy(numpy.concatenate(location_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    laser_torch = torch.from_numpy(numpy.concatenate(laser_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    rtg_torch = torch.from_numpy(numpy.concatenate(rtg_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    path_torch = torch.from_numpy(numpy.concatenate(path_seq, axis=0)).to(dtype=torch.float32, device="cuda")
    return location_torch, laser_torch, rtg_torch, path_torch

def index_to_path(index, location):
    index = list(index)
    l = deepcopy(location)
    path = [deepcopy(location)]
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

output = './output/'
logname = 'path.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = 'envs/cfg/test-path.yaml'
cfg = read_yaml(cfg)
env = make_env(cfg)
state, _, _, info = env.reset()

dwa = DWAPolicy4Nav(cfg)

GPT_cfg = GPT2Config(n_embd=cfg['token_dim'], n_layer=cfg['nlayer'], n_head=cfg['nhead'], n_inner=cfg['ninner'], resid_pdrop=cfg['dropout'])
net = NaviGPT(GPT_cfg).cuda().float().eval()
net.load_state_dict(torch.load('../../Trainer/NaviFormer/log/138M-PPO-Plus+SCAND/last_model.pt'))

while True:
    dwa_path = dwa.gen_path(state[0]*cfg['laser_max'], state[1], info['robot_pose'])

    astar_path = get_global_path(info['robot_pose'], info['goal_pose'], info['obs_list'], cfg['robot_radius'], 20, 5, (11, 11), 0.2)

    location_seq, laser_seq, rtg_seq, path_seq = seq_reset(state)
    location_torch, laser_torch, rtg_torch, path_torch = seq_to_torch(location_seq, laser_seq, rtg_seq, path_seq)
    path_index = net.pred_path(location_torch, laser_torch, path_torch, rtg_torch).cpu().numpy()
    net_path = index_to_path(path_index, info['robot_pose'])

    state, _, _, info = env.step({'DWA':dwa_path, 'Astar':astar_path, 'NF':net_path})