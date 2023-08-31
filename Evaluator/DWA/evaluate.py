from envs import make_env, read_yaml
from dwa import DWAPolicy4Nav
import sys, os

output = './output/'
logname = '10obs-5ped.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = 'envs/cfg/test.yaml'
cfg = read_yaml(cfg)
env = make_env(cfg)
policy = DWAPolicy4Nav(cfg)
state = env.reset()
while True:
    state, _, _, _ = env.step(policy.gen_action(state[0]*cfg['laser_max'], state[1]))