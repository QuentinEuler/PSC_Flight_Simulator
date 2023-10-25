import numpy as np
import matplotlib.pyplot as plt

class Force() :
    def __init__(self, sm, name = "empty_force") :
        self.plane = sm
        self.name = name

    def compute_force(self) :
        return np.array([0,0,0])

class Weight(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "weight")

        self.local_acceleration = 9.81

    def compute_force(self) :
        if self.plane.pos[2] > 0 :
            return np.array([0,0,-self.plane.MASS*self.local_acceleration])
        else :
            return 0

class Thrust(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "thrust")

    def compute_force(self) :
        thrust = self.plane.control_column*self.plane.MAX_THRUST
        return thrust*self.plane.direction()

class Drag(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "drag")

    def compute_force(self) :
        return -self.plane.abs_speed()*self.plane.speed*(1) # replace 1 with a more accurate formula taking the density of external air into account

class Lift(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "lift")

    def compute_force(self) :
        return self.plane.abs_speed()**2*(1)*np.array([-np.sin(self.plane.pitch)*np.cos(self.plane.pitch),-np.sin(self.plane.pitch)*np.sin(self.plane.bank),np.cos(self.plane.pitch)]) # replace 1 with a more accurate formula taking the density of external air into account



class SimConnect() :
    def __init__(self, frcs) :
        self.pos = np.array([0,0,0]) # x, y, z in meters
        self.speed = np.array([0,0,0])

        self.pitch = 0 # rads
        self.bank = 0 # rads

        self.control_column = 0 # between 0 and 1 : position of the aircraft control column

        self.MASS = 300 # in kg
        self.MAX_THRUST = 30000 # in N

        self.FORCES = []
        for s in frcs :
            f=Force(self)
            if s == "weight" :
                f = Weight(self)
            elif s == "thrust" :
                f = Thrust(self)
            elif s == "drag" :
                f = Drag(self)
            elif s == "lift" :
                f = Lift(self)
            self.FORCES.append(f)

    def direction(self) :
        return np.array([np.cos(self.pitch)*np.cos(self.bank),np.cos(self.pitch)*np.sin(self.bank),np.sin(self.pitch)])

    def abs_speed(self) :
        return np.sqrt(self.speed[0]**2 + self.speed[1]**2 + self.speed[2]**2)

    def compute_state(self, dt=1) :
        resulting_force = 0
        for F in self.FORCES :
            resulting_force += F.compute_force()

        accel = 1/self.MASS * resulting_force

        self.speed = self.speed + accel*dt
        
        self.pos = self.pos + self.speed*dt

class AircraftRequests() :
    def __init__(self, sm, _time) :
        self.plane = sm

    def get(self,arg) :
        if arg == "PLANE_LATITUDE" :
            return self.plane.pos[0]
        if arg == "PLANE_LONGITUDE" :
            return self.plane.pos[1]
        if arg == "PLANE_ALTITUDE" :
            return self.plane.pos[2]
        if arg == "PLANE_ALT_ABOVE_GROUND" :
            return self.plane.pos[2]
        #if arg == "PLANE_BANK_DEGREES" :
        if arg == "PLANE_PITCH_DEGREES" :
            return self.plane.pitch
        #if arg == "AIRSPEED_TRUE" :
        if arg == "VELOCITY_BODY_X" :
            return self.plane.speed[0]
        if arg == "VELOCITY_BODY_Y" :
            return self.plane.speed[1]
        if arg == "VELOCITY_BODY_Z" :
            return self.plane.speed[2]
    #if arg == "AILERON_POSITION" :
    #if arg == "ELEVATOR_POSITION" :
    #if arg == "BRAKE_INDICATOR" :
    #if arg == "PLANE_HEADING_DEGREES_TRUE" :

def test() :
    sm = SimConnect(["thrust","weight","lift","drag"])
    sm.control_column = 0.1

    ords = []
    absc = []
    
    for i in range(2000) :
        sm.compute_state(0.1)
        if i==1000 :
            sm.pitch=-0.2
            sm.control_column = 0.0988
        if i%30 == 0 :
            ords.append(sm.pos[2])
            absc.append(sm.pos[0])
            #print("time : %i seconds"%i)
            #print(sm.pos)
            #print(sm.speed)
            #print("\n\n")
    return (absc,ords)

def aff(absc,ords) :
    plt.scatter(absc,ords)
    plt.show()

aff(*test())
