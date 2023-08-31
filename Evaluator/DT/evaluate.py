from envs import make_env, read_yaml
from nn import DT, GPT2Config

import torch, numpy, sys, os

def to_discrete_action(actions):
    output = []
    for action in actions:
        discrete = numpy.zeros(shape=(1, 11*19) ,dtype=float)
        v_num = int(action[0][0] / 0.1)
        w_num = int(action[0][1] / 0.1) + 9
        index = v_num*19+w_num
        discrete[0][index] = 1.0
        output.append(discrete)
    return output

def from_discrete_action(cfg, discrete_action):
    discrete = torch.from_numpy(discrete_action.reshape((-1,)))
    return numpy.array(cfg['discrete_actions'][torch.argmax(discrete, dim=0).item()]).reshape((1, -1))

output = './output/'
logname = '10obs-5ped.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = read_yaml('envs/cfg/test.yaml')
max_len = cfg['max_len']
env = make_env(cfg)
GPT_cfg = GPT2Config(n_embd=cfg['token_dim'], n_layer=cfg['nlayer'], n_head=cfg['nhead'], n_inner=cfg['ninner'], resid_pdrop=cfg['dropout'])
net = DT(GPT_cfg).cuda().float().eval()
net.load_state_dict(torch.load('../../Trainer/DT/log/model/last_model.pt'))
states = env.reset()
while True:
    states_seq = []
    for i in range(len(states)):
        states_seq.append([states[i]])
    rtg_seq = [numpy.array([1300.]).reshape((1, 1))]
    action_seq = [numpy.array([0., 0.]).reshape((1, 2))]
    while True:
        states_torch = [torch.from_numpy(numpy.concatenate(states_seq[i], axis=0)).to(dtype=torch.float32, device="cuda") for i in range(len(states_seq))]
        rtg_torch = torch.from_numpy(numpy.concatenate(rtg_seq, axis=0)).to(dtype=torch.float32, device="cuda")
        action_torch = torch.from_numpy(numpy.concatenate(to_discrete_action(action_seq), axis=0)).to(dtype=torch.float32, device="cuda")
        discrete_action = net.pred_action(states_torch, rtg_torch, action_torch).cpu().numpy()[-1].reshape((1, -1))
        action = from_discrete_action(cfg, discrete_action)
        # print(action[0][0], ',', action[0][1], flush=True)
        states, reward, done, info = env.step(action)
        if info['all_down'][0]: break
        if len(rtg_seq) >= max_len:
            for i in range(len(states)):
                states_seq[i].pop(0)
            rtg_seq.pop(0)
            action_seq.pop(0)
        for i in range(len(states)):
            states_seq[i].append(states[i])
        rtg_seq.append(rtg_seq[-1]-reward.reshape((1, 1)))
        action_seq.insert(-1, action)
