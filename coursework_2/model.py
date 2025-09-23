import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / (fan_in ** 0.5)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=256):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fcs1_units=256, fc2_units=256):
        super().__init__()
        torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.ln1  = nn.LayerNorm(fcs1_units)
        self.fc2  = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3  = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.ln1(self.fcs1(state)))
        x  = torch.cat((xs, action), dim=1)
        x  = F.relu(self.fc2(x))
        return self.fc3(x)



# # coursework_2/model.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / (fan_in ** 0.5)
#     return (-lim, lim)

# class Actor(nn.Module):
#     """
#     Deterministic policy μ(s|θ^μ): state -> action in [-1,1]^action_size
#     Default: 33 -> 256 -> 256 -> 4 with Tanh on output.
#     """
#     def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=256):
#         super().__init__()
#         torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.ln1 = nn.LayerNorm(fc1_units)  # helps stability on Reacher
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state):
#         x = F.relu(self.ln1(self.fc1(state)))
#         x = F.relu(self.fc2(x))
#         return torch.tanh(self.fc3(x))

# class Critic(nn.Module):
#     """
#     Action-value Q(s,a|θ^Q): (state, action) -> scalar Q
#     Default: (33+4) -> 256 -> 256 -> 1.
#     """
#     def __init__(self, state_size, action_size, seed=0, fcs1_units=256, fc2_units=256):
#         super().__init__()
#         torch.manual_seed(seed)
#         self.fcs1 = nn.Linear(state_size, fcs1_units)
#         self.fc2  = nn.Linear(fcs1_units + action_size, fc2_units)
#         self.fc3  = nn.Linear(fc2_units, 1)
#         self.ln1  = nn.LayerNorm(fcs1_units)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state, action):
#         xs = F.relu(self.ln1(self.fcs1(state)))
#         x  = torch.cat((xs, action), dim=1)
#         x  = F.relu(self.fc2(x))
#         return self.fc3(x)
