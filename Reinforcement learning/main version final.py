import simulator as smlt
import simconnect_interface as sim
import agentfinal as ag
import matplotlib.pyplot as plt
import math
import random
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




class apprentissage():

    def __init__(self) :
        
        #Paramètres d'apprentissage :
        lr = 0.001 #Learning rate
        gamma = 0.8 #Paramètres gamma du Q-Learning
        N = 3 #Taille mini-batchs
        reward = ag.polynomialCos_reward #Fonction de reward
        
        #Agent de la classe agent que nous allons entraîner
        self.agent = ag.Agent(ag.AI(), lr, gamma, N, reward) 
        
        #Simulateur
        self.sm = smlt.SimConnect(["lift","thrust","weight","drag"])
        
        #Mémoire pour l'apprentissage
        self.R = [] #Garde en mémoire le reward

        
        self.precision = 5 #Paramètre de précision du simulateur
        
        
        
        
        
    def boucle_apprentissage(self,n) :
        
        for h in range(n) :
            
            #On initialise le simulateur
            self.sm = smlt.SimConnect(["lift","thrust","weight","drag"])
            sim.sim.__init__(self.sm)
            
            
            
            #On choisit la probabilité de faire une action aléatoire sur le vol
            if h%5 == 0 : #On ne fausse pas les vols dont les résultats servent à suivre l'apprentissage
                proba = 0
            else:
                proba = 0.2*(1-h/n)
            
            #On choisit l'altitude de départ
            altDebut = 800
            
            
            #On initialise l'agent
            self.agent.start(proba,altDebut)
            self.sm.pos = [0,0,altDebut]
            
            
            self.update_simulator(self.precision,self.sm)
            
            
            for i in range(1,100) :
                
                
                #On fait une etape de vol :
                    
                #On fait une action aléatoire si besoin
                if random.random()< proba :
                    self.etat_aleatoire()
                    
   
                self.agent.compute() 
                    
                    
                self.update_simulator(self.precision, self.sm) #On met à jour l'avion
                self.agent.learn(self.sm) #On apprend
                
                #On regarde si l'avion ne sort pas de la zone d'apprentissage
                if sim.get("alt")<0 or sim.get("alt")>1800:
                    break
            #On met à jour la mémoire
            self.R.append(self.agent.sommeR / i)
            
            #Cette étape permet de suivre l'evolution 
            if h%5==0:
                print("vol numero "+ str(h) + "    Durée du vol = "+ str(i) + " Altitude de départ: " + str(altDebut))
                print("Reward moyen = "+ str(self.agent.sommeR / i))
                print("Loss moyen = "+ str(self.agent.sommeL / i))
            if h%20==0 and h>0:
                self.show()
            
    

    
    def show(self) :
        for j in range(1,3):
            plt.subplot(2,2,j)
            ords = []
            absc = []
            elev = []

            #On initialise le vol
            self.sm = smlt.SimConnect(["lift","thrust","weight","drag"])
            sim.sim.__init__(self.sm)
            altDebut = 600
            proba = 0
            self.agent.start(proba,altDebut)
            self.sm.pos = [0,0,altDebut]
            
            #On fait le vol
            for i in range(1,800) :
                self.update_simulator(self.precision, self.sm)
                self.agent.compute()

                if sim.get("alt")<0 or sim.get("alt")>1800:
                    break
    

                ords.append(sim.call("alt"))
                absc.append(sim.call("lat"))
                elev.append(sim.call("elevator"))

                    
            plt.plot(absc,ords)
            plt.xlabel("Latitude")
            plt.ylabel("Altitude")
            
        plt.subplot(2,2,3)
        plt.plot(self.R)
        plt.xlabel("Nombre de vols")
        plt.ylabel("Reward moyen")
        
        plt.subplot(2,2,4)
        plt.plot(elev)
        plt.xlabel("Etapes de vol")
        plt.ylabel("Position elevator")
        plt.tight_layout()
        plt.show()
        

    
    def update_simulator(self,n,sm): 
        for i in range(n):
            sm.compute_state(1/n)
            
    def etat_aleatoire(self):
        alti = float(np.random.randn(1)*30)
        theta = float(np.random.randn(1)*90)
        self.sm.pos += np.array([0,0,alti])
        self.sm.pitch = theta
        
ap = apprentissage()
ap.boucle_apprentissage(1000000)    
