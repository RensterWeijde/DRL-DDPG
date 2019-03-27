import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, leakage=0.01, seed=123):
        '''Actor that takes the actions in the environment.'''

        super(Actor, self).__init__()
        self.leakage = leakage
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

        self.bn = nn.BatchNorm1d(state_size)
        self.reset_parameters()

    def reset_parameters(self):
        '''Kai Ming weights initialization from Imagenet model.'''
        
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leakage, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leakage, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        '''Forward policy with leaky ReLU.'''
        state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leakage)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leakage)
        x =  torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, fc3=128, leakage=0.01, seed=123):
        '''Critic model to estimate state values, expectations.'''
        
        super(Critic, self).__init__()
        self.leakage = leakage
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        '''Same Kai Ming initialization as in Actor.'''
        
        torch.nn.init.kaiming_normal_(self.fcs1.weight.data, a=self.leakage, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leakage, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        '''Forward pass.'''
        
        state = self.bn(state)
        x = F.leaky_relu(self.fcs1(state), negative_slope=self.leakage)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leakage)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leakage)
        x =  self.fc4(x)
        return x