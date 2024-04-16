import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import deque
from Environment import Environment
from Agent import PolicyNetwork, StateValueNetwork, select_action


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training_loop(env, policy_network, policy_optimizer, stateval_network, stateval_optimizer, i,  MODEL_PATH, NUM_EPISODES=750, MAX_STEPS=1000, DISCOUNT_FACTOR=0.99):
    best_score = -float('inf')
    # run episodes
    for episode in range(NUM_EPISODES):
        # init variables
        state = env.reset()
        I = 1

        # run episode, update online
        for step in range(MAX_STEPS):
            # get action and log probability
            action, lp = select_action(policy_network, state)
            # step with action
            new_state, reward, done = env.step(action)

            # update episode score
            env.score += reward

            # get state value of current state
            state_tensor = state.float().unsqueeze(0).to(DEVICE)
            state_val = stateval_network(state_tensor)

            # get state value of next state
            new_state_tensor = new_state.float().unsqueeze(0).to(DEVICE)
            new_state_val = stateval_network(new_state_tensor)

            # if terminal state, next state val is 0
            if done:
                new_state_val = torch.tensor([0]).float().unsqueeze(0)

            # calculate value function loss with MSE
            val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I

            # calculate policy loss
            advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
            policy_loss *= I

            # Backpropagate policy
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            # Backpropagate value
            stateval_optimizer.zero_grad()
            val_loss.backward()
            stateval_optimizer.step()
            if done:
                break

            # move into new state, discount I
            state = new_state
            I *= DISCOUNT_FACTOR

        if env.score > best_score:
            best_score = env.score
            policy_model_name = f"best_flight_policy_model_{i}.pt"
            policy_model_path = os.path.join(MODEL_PATH, policy_model_name)
            torch.save(policy_network.state_dict(), policy_model_path)

            value_model_name = f"best_flight_value_model_{i}.pt"
            value_model_path = os.path.join(MODEL_PATH, value_model_name)
            torch.save(stateval_network.state_dict(), value_model_path)
            print("best flight reached score " + str(best_score))

        print(f"Episode {episode + 1}: Actor Loss: {policy_loss.item()}, Critic Loss: {val_loss.item()}")
