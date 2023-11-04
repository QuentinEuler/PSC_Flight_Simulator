import simulator as smlt
import simconnect_interface as sim
import agent as ag
import matplotlib.pyplot as plt

def main() :
    agent = ag.Agent(ag.AI())

    ords = []
    absc = []
    color = []

    for h in range(1000) :
        sm = smlt.SimConnect(["lift","thrust","weight","drag"])
        sim.sim.__init__(sm)

        agent.start()
        for i in range(1000) :
            sm.compute_state(1)
            agent.compute(terminal=(i==999))
        print(h)
        print("")


    sm = smlt.SimConnect(["lift","thrust","weight","drag"])
    sim.sim.__init__(sm)
    agent.start()
    for i in range(800) :
        sm.compute_state(1)
        agent.compute(False)

        if i%3 == 0 :
            ords.append(sim.call("alt"))
            absc.append(sim.call("lat"))
            color.append(i/1001)
    plt.scatter(absc,ords,c=color)
    plt.plot(absc,ords)
    plt.show()

    return 0

main()
