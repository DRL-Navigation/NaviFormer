from nn.prenet import *
from copy import deepcopy
import math

class BC(nn. Module):
    def __init__(self, config):
        super(BC, self).__init__()
        self.policy = PolicyNetwork(config.n_embd)

    def forward(self, laser, vector):
        return self.policy([laser, vector])

    def learn(self, laser, vector, action):
        _, softmax = self.forward(laser, vector)
        return F.nll_loss(torch.log(softmax), torch.argmax(action, dim=-1))

    def pred_action(self, state):
        action, _ = self.policy(state)
        return action
