import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from Agent import PolicyNetwork, select_action
from simulateur_2D import Simulateur2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training_loop(env, policy_network, policy_optimizer, stateval_network, stateval_optimizer, i,  MODEL_PATH, NUM_EPISODES=750, MAX_STEPS=100, DISCOUNT_FACTOR=0.99):
    best_score = -float('inf')
    for episode in range(NUM_EPISODES):
        # initialiser les variables en début d'épisode
        state = env.reset()
        score = 0
        I = 1
        tab_alt = []
        tab_x = []

        for step in range(MAX_STEPS):
            # obtenir les actions et les log-probabilités
            action, lp = select_action(policy_network, state)
            # mettre à jour l'environnement
            new_state, reward, done = env.step(action)

            score += reward

            # Valeur de l'état actuel du système
            state_val = stateval_network(state.float().unsqueeze(0).to(DEVICE))

            # Valeur du prochain état du système
            new_state_val = stateval_network(new_state.float().unsqueeze(0).to(DEVICE))

            if done:
                new_state_val = torch.tensor([0]).float().unsqueeze(0)

            # Perte de la fonction de valeur
            val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I

            # Perte de la politique
            advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
            policy_loss *= I

            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            stateval_optimizer.zero_grad()
            val_loss.backward()
            stateval_optimizer.step()

            tab_x.append(env.x)
            tab_alt.append(env.alt_above_ground)
            if done:
                break

            # Changement d'état
            state = new_state
            I *= DISCOUNT_FACTOR

        # Si le modèle réalise sa meilleure erformance de la session d'entraînement, on l'enregistre et affiche le vol
        if score > best_score:
            best_score = score
            plt.figure()
            plt.plot(tab_x, tab_alt)
            plt.xlabel('Abscisse')
            plt.ylabel('Altitude')
            plt.title(f"best flight - score : {best_score} - {episode + 1} - steps : {env.count}")
            plt.savefig(f"best_flight_score_{i}.png")
            plt.close()
            policy_model_name = f"best_flight_policy_model_{i}.pt"
            policy_model_path = os.path.join(MODEL_PATH, policy_model_name)
            torch.save(policy_network.state_dict(), policy_model_path)

            value_model_name = f"best_flight_value_model_{i}.pt"
            value_model_path = os.path.join(MODEL_PATH, value_model_name)
            torch.save(stateval_network.state_dict(), value_model_path)
            print("best flight reached score " + str(best_score))

        print(f"Episode {episode + 1}: Actor Loss: {policy_loss.item()}, Critic Loss: {val_loss.item()}")

        # On enregistre également régulièrement le modèle
        if episode == NUM_EPISODES-1 or episode % 10000 == 0:
            policy_model_name = f"best_flight_policy_model_{i}_{episode}.pt"
            policy_model_path = os.path.join(MODEL_PATH, policy_model_name)
            torch.save(policy_network.state_dict(), policy_model_path)

            value_model_name = f"best_flight_value_model_{i}_{episode}.pt"
            value_model_path = os.path.join(MODEL_PATH, value_model_name)
            torch.save(stateval_network.state_dict(), value_model_path)

        # Afiichage régulier des vols du modèle pour suivre l'apprentissage
        if episode % 50 == 0:
            plt.plot(tab_x, tab_alt)
            plt.xlabel('Abscissse')
            plt.ylabel('Altitude')
            plt.savefig('dernier_vol_affiché_')
            if episode % 1000 == 0:
                plt.close()


def find_best_model(i, MODEL_PATH, n_obs=2, n_actions=3):
    scores = []
    for j in range(0, i+1):
        # Créer une nouvelle instance du modèle
        policy_network = PolicyNetwork(n_obs, n_actions).to(DEVICE)

        # Charger les paramètres du modèle à partir du fichier
        policy_model_path = os.path.join(MODEL_PATH, f'best_flight_policy_model_{j}.pt')
        policy_network.load_state_dict(torch.load(policy_model_path))
        policy_network.eval()  # Mettre le modèle en mode évaluation

        # Créer une instance de l'environnement
        env = Simulateur2D(talt=620, n_step=100)
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
        score /= 50
        scores.append(score)
    return scores.index(max(scores)) # Retourne l'indice du meilleur modèle, ce qui permet de l'identifier
