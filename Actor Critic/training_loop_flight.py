import torch
from simulateur_2D import Simulateur2D
import os
import torch.optim as optim
from ActorCritic_loops_flight import training_loop
from Agent import PolicyNetwork, StateValueNetwork

MODEL_PATH = "../models_random_reset_3"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Nombre d'épisode
NUM_EPISODES = 100000

# Nombre maximum d'étapes par épisode
MAX_STEPS = 100

n_obs = 2
n_actions = 3

# Définir le périphérique sur lequel exécuter le modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Construire l'environnement et les agents
env = Simulateur2D(talt=620, n_step=MAX_STEPS)
policy_network = PolicyNetwork(n_obs, n_actions).to(DEVICE)
stateval_network = StateValueNetwork(n_obs).to(DEVICE)

# Charger un agents déjà entraîné
policy_network.load_state_dict(torch.load('../models_random_reset_3/best_flight_policy_model_8_70000.pt'))
stateval_network.load_state_dict(torch.load('../models_random_reset_3/best_flight_value_model_8_70000.pt'))

policy_optimizer = optim.SGD(policy_network.parameters(), lr=0.0001)
stateval_optimizer = optim.SGD(stateval_network.parameters(), lr=0.0001)

training_loop(env, policy_network, policy_optimizer, stateval_network, stateval_optimizer, 9, MODEL_PATH,
              NUM_EPISODES=NUM_EPISODES, MAX_STEPS=MAX_STEPS)
