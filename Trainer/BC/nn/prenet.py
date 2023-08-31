import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from typing import List, Tuple

def mlp(input_mlp: List[Tuple[int, int, str]]) -> nn.Sequential:
    if not input_mlp:
        return nn.Sequential()
    mlp_list = []
    for input_dim, out_put_dim, af in input_mlp:
        mlp_list.append(nn.Linear(input_dim, out_put_dim, bias=True))
        if af == "relu":
            mlp_list.append(nn.ReLU())
        elif af == 'elu':
            mlp_list.append(nn.ELU())
        elif af == 'softmax':
            mlp_list.append(nn.Softmax(dim=-1))
        elif af == 'sigmoid':
            mlp_list.append(nn.Sigmoid())
        elif af == 'tanh':
            mlp_list.append(nn.Tanh())
    return nn.Sequential(*mlp_list)

class StateEmbed(nn.Module):
    def __init__(self, dim=512):
        super(StateEmbed, self).__init__()
        self.conv1d1 =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        self.conv1d2 =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        self.fc_1d = mlp([ (7168, dim, "relu")])
        self.fc1 = mlp([ (dim + 5, dim, "relu")])
        self.fc2 = nn.Linear(dim, dim)

    def _encode_laser(self,x ):
        x = self.conv1d1(x)
        x = self.conv1d2(x)
        x = self.fc_1d(x.view(x.shape[0], -1))
        return x

    def forward(self, state):
        encoded_image_laser = self._encode_laser(state[0])
        x = torch.cat((encoded_image_laser, state[1]), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, dim=512):
        super(PolicyNetwork, self).__init__()
        self.state_encode = StateEmbed(dim)
        self.fc = mlp([(dim, 11*19, "softmax")])

    def forward(self, state):
        state = self.state_encode(state)
        softmax = self.fc(state)
        action = torch.argmax(softmax, dim=-1)
        return action, softmax
