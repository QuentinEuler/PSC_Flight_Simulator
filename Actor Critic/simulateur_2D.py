import numpy as np
import torch


class Simulateur2D:
    def __init__(self, talt=620, n_step=100):
        self.alt_above_ground = 0
        self.x = 0
        self.normalized_alt_above_ground = 0
        self.pitch = 0
        self.normalized_pitch = 0
        self.done = False
        self.talt = talt
        self.reward = 0
        self.i = 0
        self.count = 0
        self.num_step = n_step

    def calculate_reward(self):
        if abs(self.normalized_alt_above_ground - 0.5) <= 0.01:
            self.reward = 5 - abs(self.normalized_alt_above_ground - 0.5)*250
            return self.reward
        if abs(self.normalized_alt_above_ground - 0.5) >= 0.05:
            self.reward = -10
            return self.reward
        else:
            self.reward = -200*abs(self.normalized_alt_above_ground - 0.5)
            return self.reward

    def give_state(self):
        return torch.tensor([self.normalized_alt_above_ground, self.normalized_pitch], dtype=torch.float)

    def reset(self):
        if self.i % 3 == 0:
            self.alt_above_ground = 600
        if self.i % 3 == 1:
            self.alt_above_ground = 620
        if self.i % 3 == 2:
            self.alt_above_ground = 680
        self.normalized_alt_above_ground = self.alt_above_ground / (2*self.talt)
        self.x = 0
        self.pitch = 0
        self.normalized_pitch = 0
        self.reward = 0
        self.count = 0
        self.done = False
        self.i += 1
        return self.give_state()

    def step(self, action):

        if action == 2:
            action = -1

        self.pitch += action

        if self.pitch > 15:
            self.pitch = 15

        elif self.pitch < -15:
            self.pitch = -15
        self.alt_above_ground += (30 * np.sin(self.pitch * np.pi / 180))
        self.x += (30 * np.cos(self.pitch * np.pi / 180))
        self.normalized_alt_above_ground = self.alt_above_ground / (2*self.talt)
        self.normalized_pitch = self.pitch/15

        self.count += 1
        if self.count == self.num_step:
            self.done = True
        if self.alt_above_ground <= 0 or self.alt_above_ground >= 6000:
            self.done = True

        self.calculate_reward()
        return self.give_state(), self.reward, self.done
