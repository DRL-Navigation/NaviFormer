from envs import make_env, read_yaml
from dwa import DWAPolicy4Nav
from multiprocessing import Process
import sys, os

output = './output/'
logname = 'exp-gen.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

class MultiEnv(Process):
    def __init__(self, cfg):
        super(MultiEnv, self).__init__()
        self.cfg = cfg
        self.env = make_env(self.cfg)
        self.policy = DWAPolicy4Nav(self.cfg)

    def run(self):
        state = self.env.reset()
        while True:
            state, _, _, _ = self.env.step(self.policy.gen_action(state[0]*self.cfg['laser_max'], state[1]))

cfg_root = 'envs/cfg/exp-generator/'
cfg_list = [
    'medium.yaml',
]

env_list = []
for cfg in cfg_list:
    cfg = read_yaml(cfg_root+cfg)
    for _ in range(cfg['env_num']):
        env = MultiEnv(cfg)
        env.start()
        env_list.append(env)

print("env_num:", len(env_list), flush=True)

for env in env_list:
    env.join()

