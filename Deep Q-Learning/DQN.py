import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

WIDTH = 16
HEIGHT = 24

class DQN(torch.nn.Module):
    def __init__(self, features, memory_size):
        super().__init__()
        params = dict()
        params['first_layer_size'] = 200    # neurons in the first layer
        params['second_layer_size'] = 20   # neurons in the second layer
        params['third_layer_size'] = 50    # neurons in the third layer
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']

        self.f1 = nn.Linear(features, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 4)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.memory = collections.deque(maxlen=memory_size)
        # Params
        self.gamma = 0.9

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    def train_short_memory(self, state_old, action, state_new, reward):
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        state_new_tensor = torch.tensor(state_new.reshape(1, 3 * WIDTH * HEIGHT), dtype=torch.float32)
        state_old_tensor = torch.tensor(state_old.reshape(1, 3 * WIDTH * HEIGHT), dtype=torch.float32, requires_grad=True)

        target = reward + self.gamma * torch.argmax(self.forward(state_new_tensor)).item()
        output = self.forward(state_old_tensor)
        target_f = output.clone()
        target_f[0][action] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

    def remember(self, state_old, action, state_new, reward):
        self.memory.append((state_old, action, state_new, reward))

    def copy(self):
        return copy.deepcopy(self)