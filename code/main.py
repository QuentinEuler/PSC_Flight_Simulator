import test
import simulateur.simulator as smlt
import simconnect_interface as sim

def main() :
    sm = smlt.SimConnect(["thrust","weight","lift","drag"])
    sim.sim.__init__(sm)

    print(sim.call("alt"))

    return 0

main()
