import simulator as smlt
import simconnect_interface as sim
import agent as ag
import matplotlib.pyplot as plt

def main() :
    agent = ag.Agent(ag.AI())

    ords = []
    absc = []

    for h in range (3) :
        sm = smlt.SimConnect(["thrust","weight","lift","drag"])
        sim.sim.__init__(sm)
        for i in range(30) :
            sm.compute_state(1)
            agent.compute()
        print(h)


    sm = smlt.SimConnect(["thrust","weight","lift","drag"])
    sim.sim.__init__(sm)
    for i in range(100) :
        sm.compute_state(1)
        agent.compute()

        if i%3 == 0 :
            ords.append(sim.call("alt"))
            absc.append(sim.call("lat"))
    plt.scatter(absc,ords)
    plt.show()

    return 0

main()
