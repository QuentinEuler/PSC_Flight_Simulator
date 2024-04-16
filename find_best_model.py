import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from Environment import Environment
from torch.distributions import Categorical
import os
from Agent import PolicyNetwork, StateValueNetwork, load_models, select_action

# Définir le périphérique sur lequel exécuter le modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_best_model(i, MODEL_PATH, n_obs=3, n_actions=3):
    scores = []
    for j in range(0, i+1):
        # Créer une nouvelle instance du modèle
        policy_network = PolicyNetwork(n_obs, n_actions).to(DEVICE)

        # Charger les paramètres du modèle à partir du fichier
        policy_model_path = os.path.join(MODEL_PATH, f'best_flight_policy_model_{j}.pt')
        policy_network.load_state_dict(torch.load(policy_model_path))
        policy_network.eval()  # Mettre le modèle en mode évaluation
        # Créer une instance de l'environnement
        env = Environment(t_alt=4500, n_step=1000)
        score = 0
        for k in range(0, 50):
        # Exécuter l'environnement avec le modèle entraîné
            state = env.reset()
            done = False
            while not done:
                # Obtenir l'action à partir du modèle entraîné
                with torch.no_grad():
                    action, _ = select_action(policy_network, state)

                # Exécuter l'action dans l'environnement
                next_state, reward, done = env.step(action)
                state = next_state
                score += reward
        score /= 25
        scores.append(score)
        print(max(scores))
    return scores.index(max(scores))


def return_best_models(MODEL_PATH):
    l = []
    for i in range(50):
        l.append(find_best_model(149, MODEL_PATH))
    l = sorted(list(set(l)))
    print(l)
