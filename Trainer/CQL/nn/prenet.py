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
    
class DiscreteActionEmbed(nn.Module):
    def __init__(self, dim=512):
        super(DiscreteActionEmbed, self).__init__()
        self.embed = nn.Embedding(11*19, dim)

    def forward(self, action):
        return self.embed(action.to(dtype=torch.int64))
    
class Scalar(nn.Module):
    def __init__(self, init_value):
        super(Scalar, self).__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant
    
class PolicyNetwork(nn.Module):
    def __init__(self, dim=512):
        super(PolicyNetwork, self).__init__()
        self.state_encode = StateEmbed(dim)
        self.fc = mlp([(dim, 11*19, "softmax")])

    def forward(self, state, deterministic=False):
        state = self.state_encode(state)
        softmax = self.fc(state)
        action_distribution = Categorical(softmax)
        if deterministic:
            action = torch.argmax(softmax, dim=-1)
        else:
            action = action_distribution.sample()
        return action, action_distribution.log_prob(action)
    
    def repeat_sample(self, state, repeat):
        def extend_and_repeat(tensor, dim, repeat):
            ones_shape = [1 for _ in range(tensor.ndim + 1)]
            ones_shape[dim] = repeat
            return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)
        
        state = self.state_encode(state)
        batch = state.shape[0]
        state = extend_and_repeat(state, 1, repeat).reshape(batch*repeat, -1)
        action_distribution = Categorical(self.fc(state))
        action = action_distribution.sample()
        log_pi = action_distribution.log_prob(action)
        return action.reshape(batch, repeat), log_pi.reshape(batch, repeat, 1)
    
class QFuncNetwork(nn.Module):
    def __init__(self, state_dim=512, action_dim=512):
        super(QFuncNetwork, self).__init__()
        self.state_encode = StateEmbed(state_dim)
        self.action_encode = DiscreteActionEmbed(action_dim)
        self.fc0 = mlp([(state_dim+action_dim, state_dim, "None")])
        self.fc1 = mlp([(state_dim, 1, "None")])

    def forward(self, state, action):
        state = self.state_encode(state)
        action = self.action_encode(action)
        x = torch.cat((state, action), dim=1)
        return self.fc1(self.fc0(x))
    
    def multi_action_count(self, state, actions):
        def extend_and_repeat(tensor, dim, repeat):
            ones_shape = [1 for _ in range(tensor.ndim + 1)]
            ones_shape[dim] = repeat
            return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)
        
        state = self.state_encode(state)
        actions = self.action_encode(actions)
        batch, repeat = actions.shape[0], actions.shape[1]
        state = extend_and_repeat(state, 1, repeat).reshape(batch*repeat, -1)
        action = actions.reshape(batch*repeat, -1)
        x = torch.cat((state, action), dim=1)
        return self.fc1(self.fc0(x)).reshape(batch, repeat, 1)
