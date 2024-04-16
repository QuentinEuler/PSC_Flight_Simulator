import numpy as np
import matplotlib.pyplot as plt
from SimConnect import *
import torch
import torch.nn as nn
import scipy
import torch.optim as optim
import time


class Environment:
    action_elev = ['elev_up', 'elev_down', 'keep']

    def __init__(self, n_step, t_alt=4500):
        self.sm = SimConnect()
        self.aq = AircraftRequests(self.sm, _time=2000)
        self.ae = AircraftEvents(self.sm)
        self.n_step = n_step
        self.count = 0
        self.t_alt = t_alt
        self.reward = 0
        self.done = False
        self.score = 0
        self.max_alt_difference = 250

        self.alt_above_ground = 0
        self.normalized_alt_above_ground = 0
        self.pitch_degree = 0
        self.normalized_pitch_degree = 0
        self.elevator = 0

    def reset(self):

        # REMETTRE À ZÉRO LES PARAMÈTRES DE VOLS

        self.ae.find("ELEVATOR_SET")(0)
        self.ae.find("THROTTLE_FULL")

        # POSITIONNER L'AVION

        self.aq.set("PLANE_PITCH_DEGREES", 0)
        self.aq.set("PLANE_BANK_DEGREES", 0)
        self.aq.set("AIRSPEED", 100)
        self.aq.set("PLANE_ALT_ABOVE_GROUND", 4000)

        # METTRE SUR LA PISTE

        self.aq.set("PLANE_LATITUDE", 0)
        self.aq.set("PLANE_LONGITUDE", 0)
        self.aq.set("PLANE_HEADING_DEGREES_MAGNETIC", 0)

        # METTRE À JOUR LES DONNÉES
        self.count = 0
        self.reward = 0
        self.done = False
        self.score = 0
        self.update()

    def update(self):
        self.alt_above_ground = self.aq.get("PLANE_ALT_ABOVE_GROUND")
        self.normalized_alt_above_ground = self.alt_above_ground / (2 * self.t_alt)
        self.pitch_degree = self.aq.get("PLANE_PITCH_DEGREES")
        self.normalized_pitch_degree = (self.pitch_degree + np.pi / 2) / np.pi
        self.elevator = (self.aq.get("ELEVATOR_POSITION") + 1) / 2

    def elev_down(self):
        self.ae.find("ELEV_DOWN")()
        return 0

    def elev_up(self):
        self.ae.find("ELEV_UP")()
        return 0

    def keep(self):
        return 0

    def reward(self):
        # Calcul de la différence entre l'altitude actuelle et l'altitude cible
        if self.alt_above_ground <= 100 or self.alt_above_ground >= 660:
            self.reward = -10
            return self.reward
        altitude_difference = self.t_alt - self.alt_above_ground
        # Récompense pour la proximité de l'altitude cible
        altitude_reward = max(-1, 1 - abs(altitude_difference) / self.max_alt_difference)
        # Calcul de la différence d'angle entre l'angle actuel et l'angle cible
        normalized_altitude_difference = max(min(altitude_difference / self.max_alt_difference, 1), -1)
        pitch_target = normalized_altitude_difference * 15
        pitch_difference = abs(self.pitch_degree - pitch_target)
        # Normalisation de la différence d'angle entre -45 et 45 degrés
        pitch_reward = 1 - abs(pitch_difference / 15)
        # Calcul de la récompense totale
        altitude_weight = 0.6  # Poids de la récompense liée à l'altitude
        pitch_weight = 0.4  # Poids de la récompense liée au pitch
        self.reward = altitude_weight * altitude_reward + pitch_weight * pitch_reward
        return self.reward()

    def give_state(self):
        return torch.tensor([self.normalized_alt_above_ground, self.normalized_pitch_degree, self.elevator],
                            dtype=torch.float)

    def step(self, elev):

        # EFFECTUER LES ACTIONS
        actions_functions = {
            'elev_up': self.elev_up(),
            'elev_down': self.elev_down(),
            'keep': self.keep()
        }

        actions_functions[self.action_elev[elev]]

        # METTRE À JOUR L'ENVIRONNEMENT

        self.update()

        # METTRE À JOUR LA REWARD
        self.reward()
        self.count += 1
        if self.count == self.n_step:
            self.done = True
        if self.alt_above_ground <= 0 or self.alt_above_ground >= 6000:
            self.done = True
        return self.give_state(), self.reward, self.done
