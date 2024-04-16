import torch
from simulateur_2D_w_gravity import simulateur2D
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from training_loop import training_loop
from Agent import PolicyNetwork, StateValueNetwork, load_models, select_action
from find_best_model import find_best_model, return_best_models


MODEL_PATH = "first_attempt_FS"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

#number of episodes to run
NUM_EPISODES = 750

#max steps per episode
MAX_STEPS = 1000

n_obs = 3
n_actions = 3

#device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
j = 0
for i in range(150):
    env = Environment(t_alt=4500, n_step=MAX_STEPS)
    policy_network = PolicyNetwork(n_obs, n_actions).to(DEVICE)
    stateval_network = StateValueNetwork(n_obs).to(DEVICE)
    policy_optimizer = optim.SGD(policy_network.parameters(), lr=0.001)
    stateval_optimizer = optim.SGD(stateval_network.parameters(), lr=0.001)
    if j != 0:
        policy_model_path = os.path.join(MODEL_PATH, f'best_flight_policy_model_{j}.pt')
        policy_network.load_state_dict(torch.load(policy_model_path))
        value_model_path = os.path.join(MODEL_PATH, f'best_flight_value_model_{j}.pt')
        stateval_network.load_state_dict(torch.load(value_model_path))
    training_loop(env, policy_network, policy_optimizer, stateval_network, stateval_optimizer, i,  MODEL_PATH)
    j = find_best_model(i, MODEL_PATH)
    print(j)

return_best_models(MODEL_PATH)
