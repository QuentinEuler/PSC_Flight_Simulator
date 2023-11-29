import simulator as smlt
import simconnect_interface as sim
import agent as ag
import matplotlib.pyplot as plt

import torch

def main() :
    agent = ag.Agent(ag.AI())

    for h in range(10000) :
        sm = smlt.SimConnect(["thrust","drag"])
        sim.sim.__init__(sm)

        agent.start()
        for i in range(300) :
            sm.compute_state(1)
            if i%10==0 :
                agent.compute(dt=3)
        print(agent.total_reward)
        #agent.scheduler.step()
        if h%40==0 :
            show(agent)

        #print(agent.total_reward)

    show(agent)

    return 0

def show(agent) :
    ords = []
    absc = []
    color = []

    sm = smlt.SimConnect(["thrust","drag"])
    sim.sim.__init__(sm)
    agent.start()
    for i in range(300) :
        sm.compute_state(1)
        if i%3==0 :
            agent.compute(dt=3)

        if i%3 == 0 :
            ords.append(sim.call("alt"))
            absc.append(sim.call("lat"))
            color.append(i/1001)
    plt.scatter(absc,ords,c=color)
    plt.plot(absc,ords)
    plt.show()

main()
