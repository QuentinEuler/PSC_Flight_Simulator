import simconnect_interface as sim
import numpy as np
import torch
from random import randint

A = [-1,0,1]

class AI(torch.nn.Module) :
    def __init__(self) :
        super(AI, self).__init__()

        self.linear1 = torch.nn.Linear(4,128)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128,128)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(128,128)
        self.activation3 = torch.nn.Sigmoid()
        self.linear4 = torch.nn.Linear(128,128)
        self.activation4 = torch.nn.Sigmoid()
        self.linear5 = torch.nn.Linear(128,128)
        self.activation5 = torch.nn.Sigmoid()
        self.linear6 = torch.nn.Linear(128,1)
        self.activation6 = torch.nn.Sigmoid()


    def forward(self, x) :
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        x = self.activation5(x)
        x = self.linear6(x)
        x = self.activation6(x)
        return x

class Agent :
    def __init__(self, a) :
        self.ai = a
        self.optimizer = torch.optim.SGD(self.ai.parameters(), lr = 0.0003)
        self.sa = (0,0)
        self.R = []
        self.N = 10
        self.gamma = 0.7



    def start(self) :
        return 0

    def state(self):
        return (sim.call("alt"), sim.call("pitch"), sim.call("abs_speed"))

    # this function interact with the plane
    # use functions defined in simconnect_interface to interact with the plane
    def compute(self) :
        s = self.state()
        a = self.mu(s)
        sim.set("elevators", a)
        self.sa = (s,a)
        return a

    def learn(self):
        self.R.append((*self.sa,self.reward(),self.state()))
        if (len(self.R) > self.N):
            Y = torch.tensor([float(0)])
            for i in range(self.N):
                k = randint(0,self.N-1)
                s,a,r,s2 = self.R[k]
                Y +=(r + self.gamma*self.Q(s2,self.mu(s2)).item()- self.Q(s,a))
            loss = 1/self.N * Y
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = False)


    def Q(self,s,a):
        args = torch.tensor(np.double([*s,a])).type(torch.FloatTensor)
        args[1]*=100
        args[3]*=100

        return self.ai(args)
    def mu(self,s):
        maxA = -1
        maxQ = self.Q(s,maxA).item()
        for a in A:
            if self.Q(s,a).item()>maxQ:
                maxA = a
                maxQ = self.Q(s,a).item()
        return a

    def reward(self):
        return 1/(1+(self.state()[0] - 400)**2)




# example of a compute function
#    def compute(self) :
#        if sim.call("alt") < 10 :
#            sim.set("control_column", 1)
#            sim.set("pitch", 0)
#        elif sim.call("alt") < 1000 :
#            sim.set("pitch", np.pi/5)
#        else :
#            sim.set("pitch", -0.2)
#            sim.set("control_column", 0.0988)