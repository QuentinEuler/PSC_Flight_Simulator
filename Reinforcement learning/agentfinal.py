
import simconnect_interface as sim
import numpy as np
import torch
from random import randint
import random
import math
from copy import deepcopy

################# Différentes fonctions de Reward #############################
altCible = 800
ALTITUDE = 0
PITCH = 1

def polynomial_reward(s): #Reward polynomial
    
    norm = 0.1 #Paramètre de normalisation
    
    try :
        return 1/(1+(norm**2)*(altCible-s[ALTITUDE])**2)
    except OverflowError:
        return 0

def polynomialCos_reward(s): #Reward polynomial avec prise en compte du pitch
    
    norm = 0.1 #Paramètre de normalisation
    
    try :
        return (1/(1+(norm**2)*(altCible-s[ALTITUDE])**2))*(abs(math.cos(s[PITCH])))
    except OverflowError:
        return 0
    
def zeroUn_reward(s): #Reward zero ou un
    
    #Paramètres d'écart
    ecart1 = 50
    ecart2 = 200
    
    if abs(s[ALTITUDE]-altCible)< ecart1 :
        return 1
    elif abs(s[ALTITUDE]-altCible)> ecart2 : 
        return -1
    else:
        return 0
    
def linear_reward(s): #Reward lineaire
    
    return 1-abs(s[ALTITUDE]-altCible)/altCible
    

################# Espace des actions ##########################################

m = 4 #determine le nombre d'action -> #A = 2m+1
A = [(i-m)/m for i in range(2*m+1)] #Espace des actions


################# Définitions du réseau de neurones ###########################

class AI(torch.nn.Module) :
    def __init__(self) :
        super(AI, self).__init__()
            
        #Définitions des couches du réseau de neurones
        
        self.linear1 = torch.nn.Linear(3,30)
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(30,60)
        self.activation2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(60,1)



    
    #La fonction forward convertit le réseau de neurones en fonction 
    
    def forward(self, x) : 
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)

        #x = self.activation6(x)
        return x

################# Définition de l'agent  ######################################


class Agent :
    def __init__(self, a, lr_, gamma_, N_, reward_func) :
        
        #Réseau de neurone de l'agent
        self.ai = a #Réseau de neurones
        self.optimizer = torch.optim.SGD(a.parameters(), lr = lr_) #lr = learning rate
        
        #Paramètres de l'apprentissage
        self.N = N_ #Nombre d'étapes de mémoires prises en compte à chaque descente de gradient
        self.gamma = gamma_ #Paramètre gamma du Q-learning
        self.reward = reward_func #Fonction de reward
        self.proba = 0 #Probabilité de faire une action aléatoire
        
        #Mémoire de l'agent
        self.R = [] #Garde en mémoire les quadruplets etat,action,etat suivant, reward
        self.sommeL = 0 #Loss total sur un vol
        self.sommeR = 0 #Reward total sur un vol
        
        
        #Infos sur l'avion
        self.etat_precedent = (0,0) 
       

    def start(self,proba,altitude) : #Remet à 0 la mémoire de l'agent pour un nouveau vol 
        sim.set("alt", altitude)
        self.proba = proba
        self.sommeR = 0
        self.sommeL = 0

        return 0

    def state(self): #Renvoie l'état actuel
        return (sim.call("alt"), sim.call("pitch"))

############### Définition des étapes de l'apprentissage ######################

    def compute(self) : #Met à jour l'état de l'elevator 
        s = self.state()
        a = self.mu(s) #a devient l'action choisis par le réseau
        sim.set("elevators", a)
        self.etat_precedent = (s,a)
        return a

    def learn(self,sm): #Fais la descente de gradient

        s,a = self.etat_precedent
        r = self.reward2(s,a,sm)
        self.R.append((s,a,r,self.state()))
        self.sommeR += self.reward(s)
        if (len(self.R) > self.N):
            Y = torch.tensor([float(0)])
            
            #On choisit les N étapes dans la mémoire qui seront pris en compte
            for i in range(self.N):
                if i == (self.N - 1):
                    k = self.N - 1
                else:
                    k = randint(0,self.N-1)
                s,a,r,s2 = self.R[k]
                Y +=(r - self.Q(s,a))**2
            loss = 1/self.N * Y
            self.sommeL += loss.item()
            
            #On caclule le gradient du loss
            loss.backward()
            
            #On met à jour le réseau de neurones
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = False)


    def Q(self,s,a): #Renvoie la valeur du réseau de neurones Q(s,a)
        args = torch.tensor(np.double([*s,a])).type(torch.FloatTensor)
        return self.ai(args)
    
    
    
    def mu(self,s): #Choisis l'action qui maximise Q
        maxA = -1
        maxQ = self.Q(s,maxA).item()
        for a in A:
            if self.Q(s,a).item()>=maxQ:
                maxA = a
                maxQ = self.Q(s,a).item()
        if random.random()<self.proba :
            return random.choice(A)
        return maxA


    def reward2(self,s,a,sm): #Reward qui prend plus en compte les états suivant en sommant les rewards
        sm2 = deepcopy(sm)
        sm2.elevators = a
        r = 0 
        n = 15
        for i in range(n):
            sm2.compute_state(1)
            s = sm2.state()
            r += 1/n * self.reward(s)
        return r
