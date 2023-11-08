import simconnect_interface as sim
import numpy as np
import torch

class AI(torch.nn.Module) :
    def __init__(self) :
        super(AI, self).__init__()

        self.linear1 = torch.nn.Linear(3,128)
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(128,128)
        self.activation2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(128,2)
        self.activation3 = torch.nn.Softmax()

    def forward(self, x) :
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        return x

class Agent :
    def __init__(self, a) :
        self.ai = a
        self.optimizer = torch.optim.SGD(self.ai.parameters(), lr = 0.3)
        self.time=0

    def start(self) :
        self.time = 0
        self.terminal_reward = 0

    # this function interact with the plane
    # use functions defined in simconnect_interface to interact with the plane
    def compute(self, learn=True, terminal=False) :
        self.time += 1

        X = torch.tensor([sim.call("alt")/100, sim.call("pitch"), sim.call("abs_speed")/10000])
        Y = self.ai(X)
        
        reward = 0
        if sim.call("alt") > 300 and sim.call("alt") < 2000 :
            self.terminal_reward += 1
        else :
            reward = -1

        if (terminal) :
            print(self.terminal_reward)
            reward = self.terminal_reward

        loss = ((Y-reward)**2).sum()
        if learn :
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = False)

        if Y[0]<1/7 :
            sim.set("elevators", -1)
        elif Y[0] < 6/7 :
            sim.set("elevators", 0)
        else :
            sim.set("elevators", 1)

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
