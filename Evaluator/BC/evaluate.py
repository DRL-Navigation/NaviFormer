from envs import make_env, read_yaml
from nn import BC, GPT2Config
import sys, os, torch, numpy

def from_discrete_action(cfg, discrete_action):
    return numpy.array(cfg['discrete_actions'][discrete_action.item()]).reshape((1, -1))

output = './output/'
logname = '10obs-5ped.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = 'envs/cfg/test.yaml'
cfg = read_yaml(cfg)
env = make_env(cfg)
GPT_cfg = GPT2Config(n_embd=512)
net = BC(GPT_cfg).cuda().float().eval()
net.load_state_dict(torch.load('../../Trainer/BC/log/model/last_model.pt'))
state = env.reset()
while True:
    state_torch = [torch.from_numpy(state[0]).to(dtype=torch.float32, device="cuda"), torch.from_numpy(state[1]).to(dtype=torch.float32, device="cuda")]
    action_index = net.pred_action(state_torch).cpu().numpy()
    state, _, _, _ = env.step(from_discrete_action(cfg, action_index))