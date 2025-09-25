import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / (fan_in ** 0.5)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc1=256, fc2=256):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    # centralized: concat(all_states, all_actions)
    def __init__(self, full_state_size, full_action_size, seed=0, fcs1=256, fc2=256):
        super().__init__()
        torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1)
        self.fc2 = nn.Linear(fcs1 + full_action_size, fc2)
        self.fc3 = nn.Linear(fc2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_states, full_actions):
        xs = F.relu(self.fcs1(full_states))
        x = torch.cat((xs, full_actions), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
