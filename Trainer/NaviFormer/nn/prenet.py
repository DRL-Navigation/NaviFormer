import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.mlp_location = mlp([(3, dim, "None")])
        self.conv1d1 =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        self.conv1d2 =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        self.mlp_laser = mlp([(7168, dim*2, "elu"), (dim*2, dim, "None")])
    
    def _encode_laser(self, laser):
        laser = 1.0 / (laser*10.0-0.17)
        x = self.conv1d1(laser)
        x = self.conv1d2(x)
        return self.mlp_laser(x.reshape(x.shape[0], -1))   

    def forward(self, location, laser):
        #Size: (Batch, Length, Dim)
        batch_size, seq_length = laser.shape[0], laser.shape[1]
        location = location.reshape((batch_size*seq_length,)+location.shape[2:])
        laser = laser.reshape((batch_size*seq_length,)+laser.shape[2:])
        location = self.mlp_location(location).reshape(batch_size, seq_length, -1)
        laser = self._encode_laser(laser).reshape(batch_size, seq_length, -1)
        return [location, laser]
    
class PathEmbed(nn.Module):
    def __init__(self, dim=512):
        super(PathEmbed, self).__init__()
        self.embed = nn.Embedding(11*19, dim)

    def forward(self, path):
        batch_size, seq_length = path.shape[0], path.shape[1]
        path = path.reshape((batch_size*seq_length,)+path.shape[2:]).to(dtype=torch.int64)
        path_list = path.split((1,)*path.shape[1], dim=1)
        paths = [self.embed(path_split).reshape(batch_size, seq_length, -1) for path_split in path_list]
        return paths

class RTGEmbed(nn.Module):
    def __init__(self, dim=512, std=25.0, mean=0.0):
        super(RTGEmbed, self).__init__()
        self.fc = mlp([(1, dim, "None")])
        self.std = std
        self.mean = mean

    def forward(self, rtg):
        if self.training:
            rtg += torch.randn_like(rtg)*self.std + self.mean
        return self.fc(rtg)

class PathPredict(nn.Module):
    def __init__(self, dim=512):
        super(PathPredict, self).__init__()
        self.fc = mlp([(dim, 11*19, "softmax")])

    def forward(self, token):
        softmax = self.fc(token)
        path = torch.argmax(softmax, dim=-1)
        return path, softmax
