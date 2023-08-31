from envs import read_yaml
from nn import NaviGPT, GPT2Config

import torch
from torchinfo import summary

cfg = read_yaml('envs/cfg/test.yaml')
GPT_cfg = GPT2Config(n_embd=cfg['token_dim'], n_layer=cfg['nlayer'], n_head=cfg['nhead'], n_inner=cfg['ninner'], resid_pdrop=cfg['dropout'])
net = NaviGPT(GPT_cfg).cuda().float()
# net.load_state_dict(torch.load('../../Trainer/Action-GPT/log/model/last_model.pt'))

batch_size = 64
seq_len = 20
states = [torch.ones((batch_size, seq_len, 1, 960), dtype=torch.float32, device='cuda'),
          torch.ones((batch_size, seq_len, 5), dtype=torch.float32, device='cuda'),
          torch.ones((batch_size, seq_len, 3, 48, 48), dtype=torch.float32, device='cuda'),
        ]
rtg = torch.ones((batch_size, seq_len, 1), dtype=torch.float32, device='cuda')
action = torch.ones((batch_size, seq_len, 133), dtype=torch.float32, device='cuda')
mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device='cuda')

summary(model=net, input_data=(states, rtg, action, mask))
