import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from simulateur_2D_w_gravity import simulateur2D
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import deque

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer1 = nn.Linear(128, 64)  # Nouvelle couche cachée
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer3 = nn.Linear(32, 64)
        self.hidden_layer4 = nn.Linear(64, 128)# Nouvelle couche cachée
        self.output_layer = nn.Linear(128, action_space)

    # forward pass
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))  # Utilisation de la ReLU comme fonction d'activation
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        x = F.relu(self.hidden_layer4(x))  # Utilisation de la ReLU comme fonction d'activation
        actions = self.output_layer(x)
        action_probs = F.softmax(actions, dim=1)
        return action_probs


# Using a neural network to learn state value
class StateValueNetwork(nn.Module):

    # Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer1 = nn.Linear(128, 64)  # Nouvelle couche cachée
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer3 = nn.Linear(32, 64)
        self.hidden_layer4 = nn.Linear(64, 128)# Nouvelle couche cachée
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))  # Utilisation de la ReLU comme fonction d'activation
        x = F.relu(self.hidden_layer2(x))  # Utilisation de la ReLU comme fonction d'activation
        x = F.relu(self.hidden_layer3(x))
        x = F.relu(self.hidden_layer4(x))
        state_value = self.output_layer(x)
        return state_value


def select_action(network, state):

    # convert state to float tensor, add 1 dimension, allocate tensor on device
    state = state.float().unsqueeze(0).to(DEVICE)

    # use network to predict action probabilities
    action_probs = network(state)
    state = state.detach()

    # sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()
    if action.item() == 0:
        act = 'keep'
    elif action.item() == 1:
        act = 'elev_up'
    else:
        act = 'elev_down'
    # return action
    return act, m.log_prob(action)
