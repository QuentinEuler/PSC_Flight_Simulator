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
        return -self.plane.abs_speed()*self.plane.speed*(2) # replace 1 with a more accurate formula taking the density of external air into account

class Lift(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "lift")

    def compute_force(self) :
        return self.plane.abs_speed()**2*(1/2)*np.array([-np.sin(self.plane.pitch),0,np.cos(self.plane.pitch)]) # replace 1 with a more accurate formula taking the density of external air into account



class SimConnect() :
    def __init__(self, frcs) :
        self.pos = np.array([0,0,0]) # x, y, z in meters
        self.speed = np.array([0,0,0])

        self.pitch = 0 # rads
        self.bank = 0 # rads
        self.cap = 0 # rads

        self.control_column = 0.6 # between 0 and 1 : position of aircraft control column
        self.elevators = 0 # between -1 and 1 : rotation rate = elevators * MAX_ROTATION_RATE

        self.MASS = 300 # in kg
        self.MAX_THRUST = 30000 # in N
        self.MAX_ROTATION_RATE = 0.06 # in rads/s

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
        return np.array([np.cos(self.pitch)*np.cos(self.cap),np.cos(self.pitch)*np.sin(self.cap),np.sin(self.pitch)])

    def abs_speed(self) :
        return np.linalg.norm(self.speed)

    def compute_state(self, dt=1) :
        resulting_force = 0
        for F in self.FORCES :
            resulting_force += F.compute_force()

        accel = 1/self.MASS * resulting_force

        self.speed = self.speed + accel*dt

        self.pos = self.pos + self.speed*dt

        self.pitch = (self.pitch + self.MAX_ROTATION_RATE * self.elevators + np.pi) % (2*np.pi) - np.pi

class AircraftRequests() :
    def __init__(self, sm, _time) :
        self.plane = sm

    def get(self, arg) :
        if arg == "PLANE_LATITUDE" :
            return self.plane.pos[0].item()
        if arg == "PLANE_LONGITUDE" :
            return self.plane.pos[1].item()
        if arg == "PLANE_ALTITUDE" :
            return self.plane.pos[2].item()
        if arg == "PLANE_ALT_ABOVE_GROUND" :
            return self.plane.pos[2].item()
        #if arg == "PLANE_BANK_DEGREES" :
        if arg == "PLANE_PITCH_DEGREES" :
            return self.plane.pitch
        if arg == "AIRSPEED_TRUE" :
            return self.plane.abs_speed().item()
        if arg == "VELOCITY_BODY_X" :
            return self.plane.speed[0].item()
        if arg == "VELOCITY_BODY_Y" :
            return self.plane.speed[1].item()
        if arg == "VELOCITY_BODY_Z" :
            return self.plane.speed[2].item()
    #if arg == "AILERON_POSITION" :
    #if arg == "ELEVATOR_POSITION" :
    #if arg == "BRAKE_INDICATOR" :
        if arg == "PLANE_HEADING_DEGREES_TRUE" :
            return self.cap

        if arg == "CONTROL_COLUMN" :
            return self.control_column

class AircraftEvents() :
    def __init__(self, sm) :
        self.plane = sm

    def set(self, arg, val) :
        if arg == "ELEVATOR_POSITION" :
            self.plane.elevators = val
        if arg == "CONTROL_COLUMN" :
            self.plane.control_column = val

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

#aff(*test())
