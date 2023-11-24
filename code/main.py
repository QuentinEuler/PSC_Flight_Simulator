import simulator as smlt
import simconnect_interface as sim
import agent as ag
import matplotlib.pyplot as plt

import torch

def main() :
    agent = ag.Agent(ag.AI())

    for h in range(1000) :
        sm = smlt.SimConnect(["thrust","drag"])
        sim.sim.__init__(sm)

        agent.start()
        for i in range(40) :
            sm.compute_state(1)
            agent.compute()

        print(agent.total_reward)

    show(agent)

    return 0

def show(agent) :
    ords = []
    absc = []
    color = []

    sm = smlt.SimConnect(["thrust","drag"])
    sim.sim.__init__(sm)
    agent.start()
    for i in range(40) :
        sm.compute_state(1)
        agent.compute()

        if i%3 == 0 :
            ords.append(sim.call("alt"))
            absc.append(sim.call("lat"))
            color.append(i/1001)
    plt.scatter(absc,ords,c=color)
    plt.plot(absc,ords)
    plt.show()

main()
