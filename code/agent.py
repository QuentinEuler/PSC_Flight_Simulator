import simconnect_interface as sim
import numpy as np
import torch

class AI(torch.nn.Module) :
    def __init__(self) :
        super(AI, self).__init__()

        self.linear1 = torch.nn.Linear(4,128)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128,128)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(128,128)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(128,128)
        self.activation4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(128,128)
        self.activation5 = torch.nn.ReLU()
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
        self.optimizer = torch.optim.SGD(self.ai.parameters(), lr = 0.03)
        self.time=0

        self.last_chosen_action = None

    def start(self) :
        self.time = 0
        self.GAMMA = 0.3

    # this function interact with the plane
    # use functions defined in simconnect_interface to interact with the plane
    def compute(self) :
        self.time += 1

        Y=[0,0,0]
        for a in [-1,0,1] :
            X = torch.tensor([sim.call("alt"), sim.call("pitch"), sim.call("abs_speed"), a])
            Y[a+1] = self.ai(X)

        max_a = 0
        a=0
        for i in [-1,0,1] :
            if Y[i+1].item() > max_a :
                max_a = Y[i+1].item()
                a = i

        if self.time>1 :
            R = 0
            if sim.call("alt") > 50 and sim.call("alt") < 150 :
                R = 1
            S = Y[a+1].item()
            loss = (self.last_chosen_action - (R + self.GAMMA * S))**2

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = False)

        self.last_chosen_action = Y[a+1]

        sim.set("elevators", a)

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
