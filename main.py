import simulator as smlt
import simconnect_interface as sim
import agent as ag
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main() :
    agent = ag.Agent(ag.AI())

    for h in range(30) :
        sm = smlt.SimConnect(["lift","thrust","weight","drag"])
        sim.sim.__init__(sm)

        agent.start()
        sm.compute_state(1)
        for i in range(300) :
            agent.compute()
            sm.compute_state(1)
            agent.learn()
        print(h)
        print("")

    show(agent)

    return 0

def show(agent) :
    ords = []
    absc = []
    color = []

    sm = smlt.SimConnect(["lift","thrust","weight","drag"])
    sim.sim.__init__(sm)
    agent.start()
    for i in range(100) :
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