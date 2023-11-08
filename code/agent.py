import simconnect_interface as sim
import numpy as np
import torch

class AI(torch.nn.Module) :
    def __init__(self) :
        super(AI, self).__init__()

        self.linear1 = torch.nn.Linear(3,128)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128,128)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(128,128)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(128,128)
        self.activation4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(128,128)
        self.activation5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(128,2)
        self.activation6 = torch.nn.Softmax()

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
        self.time=0

    def start(self) :
        self.time = 0
        self.terminal_reward = 0

    # this function interact with the plane
    # use functions defined in simconnect_interface to interact with the plane
    def compute(self, learn=True, terminal=False) :
        self.time += 1

        X = torch.tensor([sim.call("alt"), sim.call("pitch"), sim.call("abs_speed")])
        Y = self.ai(X)
        
        reward = 0
        if sim.call("alt") > 70 and sim.call("alt") < 130 :
            self.terminal_reward += 1
        else :
            reward = 0

        if (terminal) :
            print(self.terminal_reward)
            reward = self.terminal_reward

        #loss = (Y[0]-reward)**2
        #print(sim.call("abs_speed")*torch.sin(sim.call("pitch") + Y[0]*0.06))
        #print(sim.call("abs_speed"))
        loss = ((sim.call("alt") + 10*sim.call("abs_speed")*torch.sin(sim.call("pitch") + (Y[0]*2-1)*0.06) - 100)/10000)**2 + (sim.call("pitch") + (Y[0]*2-1)*0.06)**2
        #print(Y)
        #print(loss)
        #print("")
        if learn :
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = False)

        a = Y[0]
        l = np.random.rand()
        if l>1 :
            a=np.random.rand()
        """
        if a<0.2 :
            sim.set("elevators", -1)
        elif a < 0.3 :
            sim.set("elevators", 0)
        else :
            sim.set("elevators", 1)
        """
        sim.set("elevators", a.item()*2 -1)


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
