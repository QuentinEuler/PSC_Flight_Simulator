import torch
import os
import torch.optim as optim
from simulateur_2D import Simulateur2D
from training_loop_flight import training_loop
from Agent import PolicyNetwork, StateValueNetwork
from ActorCritic_loops_flight import find_best_model

MODEL_PATH = "../models_random_reset_3"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

NUM_EPISODES = 750

MAX_STEPS = 100

n_obs = 2
n_actions = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
j = 0

for i in range(150):
    env = Simulateur2D(talt=620, n_step=MAX_STEPS)
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
    print(j) # On affiche le meilleur modèle à chaque itération
