import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):

    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, action_space)

    # forward pass
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        actions = self.output_layer(x)
        action_probs = F.softmax(actions, dim=1)
        return action_probs


class StateValueNetwork(nn.Module):

    # Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        state_value = self.output_layer(x)
        return state_value


def select_action(network, state):
    state = state.float().unsqueeze(0).to(DEVICE)
    action_probs = network(state)
    m = Categorical(action_probs)   # Tirer une action selon les probabibilité obtenue
    action = m.sample()
    return action.item(), m.log_prob(action)
