import simconnect_interface as sim
import numpy as np
import torch

class AI(torch.nn.Module) :
    def __init__(self) :
        super(AI, self).__init__()

        self.linear1 = torch.nn.Linear(4,32)
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(128,128)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32,32)
        self.activation3 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(32,1)

    def forward(self, x) :
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear6(x)
        return x

class Agent :
    def __init__(self, a) :
        self.ai = a
        self.optimizer = torch.optim.SGD(self.ai.parameters(), lr = 0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.9)
        self.time=0
        self.total_reward=0

        self.last_chosen_action = None

    def start(self) :
        self.time = 0
        self.GAMMA = 0.1

        self.total_reward=0

        self.ai_ = AI()
        self.ai_.load_state_dict(self.ai.state_dict())
        #self.scheduler.step()

    # this function interact with the plane
    # use functions defined in simconnect_interface to interact with the plane
    def compute(self,dt=1) :
        Q=[0,0,0]
        for a in [-1,0,1] :
            X = torch.tensor([np.floor(max(min(sim.call("alt"),10000),-2000)/30).item(), 10*sim.call("pitch"), self.time, 100*a])
            Q[a+1] = self.ai_(X)

        max_a = -10000
        a=0
        for i in [-1,0,1] :
            if Q[i+1].item() > max_a :
                max_a = Q[i+1].item()
                a = i

        if self.time>1 :
            #R = np.exp(-((sim.call("alt")-100)/40)**2)
            R=0
            if True :#self.time<150 :
                if sim.call("alt") >100 and sim.call("alt") <2000 :
                    R=1
                elif sim.call("alt") < -1000 or sim.call("alt")>5000 :
                    R=-1
                elif sim.call("alt") < 100 or sim.call("alt")>2000 :
                    R=-0.3
            else :
                if sim.call("alt") >800 and sim.call("alt") <1200 :
                    R=1
                elif sim.call("alt") < -1000 or sim.call("alt")>5000 :
                    R=-0.1
                elif sim.call("alt") < 100 or sim.call("alt")>12000 :
                    R=-0.03

            S = Q[a+1].item()
            loss = (self.ai(self.last_chosen_action) - (R + self.GAMMA * S))**2
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = False)

            self.total_reward+=R

        self.last_chosen_action = torch.tensor([np.floor(max(min(sim.call("alt"),1000),-200)/10).item(), 10*sim.call("pitch"), self.time, 100*a])

        sim.set("pitch", a*np.pi/8)
        
        self.time+=dt

# example of a compute function
#    def compute(self) :
#        if sim.call("alt") < 10 :
#            sim.set("control_column", 1)
#            sim.set("elevators", 0)
#        elif sim.call("alt") < 1000 :
#            sim.set("elevators", 0)
#        else :
#            sim.set("elevator", -1)
#            sim.set("control_column", 0.0988)
