import torch
import matplotlib.pyplot as plt
from simulateur_2D import Simulateur2D
import Agent

# Définir le périphérique sur lequel exécuter le modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

n_obs = 2
n_actions = 3

# Créer une nouvelle instance du modèle
policy_network = Agent.PolicyNetwork(n_obs, n_actions).to(DEVICE)

# Charger les paramètres du modèle à partir du fichier
policy_network.load_state_dict(torch.load('../models_random_reset_3/best_flight_policy_model_8_70000.pt'))
policy_network.eval()  # Mettre le modèle en mode évaluation
# Créer une instance de l'environnement
env = Simulateur2D(talt=2480, n_step=10000)

for i in range(0, 30):
# Exécuter l'environnement avec le modèle entraîné
    state = env.reset()
    done = False
    tab_alt = []
    tab_x = []
    while not done:
        # Obtenir l'action à partir du modèle entraîné
        with torch.no_grad():
            action, _ = Agent.select_action(policy_network, state)

        # Exécuter l'action dans l'environnement
        next_state, reward, done = env.step(action)
        state = next_state

        # Enregistrer les données du vol
        tab_alt.append(env.alt_above_ground)
        tab_x.append(env.x)

    plt.plot(tab_x, tab_alt)
    plt.axhline(y=2480, color='r', linestyle='--')
    plt.xlabel('Latitude')
    plt.ylabel('Altitude')
    plt.title('Vol de validation - Altitude cible 2480')
    plt.show()
