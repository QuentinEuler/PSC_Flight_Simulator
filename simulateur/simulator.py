import numpy

def Force() :
    def __init__(self, sm, name = "empty_force") :
        self.plane = sm
        self.name = name

    def compute_force(self) :
        return numpy.array([0,0,0])

def Weight(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "weight")

        self.local_acceleration = 9.81

    def compute_force(self) :
        return numpy.array([0,0,-self.plane.MASS*self.local_acceleration])

def Thrust(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "thrust")

    def compute_force(self) :
        thrust = self.plane.control_column*self.plane.MAX_THRUST
        return thrust*self.plane.direction()

def Drag(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "drag")

    def compute_force(self) :
        return -self.plane.abs_speed()*self.plane.speed()*(1) # replace 1 with a more accurate formula taking the density of external air into account

def Lift(Force) :
    def __init__(self, sm) :
        super().__init__(sm, "lift")

    def compute_force(self) :
        return numpy.array([0,0,self.plane.abs_speed()**2*(1)]) # replace 1 with a more accurate formula taking the density of external air into account



def SimConnect() :
    def __init__(self) :
        self.pos = numpy.array([0,0,0]) # x, y, z in meters
        self.speed = numpy.array([0,0,0])

        self.pitch = 0 # rads
        self.bank = 0 # rads

        self.control_column = 0 # between 0 and 1 : position of the aircraft control column

        self.MASS = 300 # in kg
        self.MAX_THRUST = 30000 # in N

    def direction(self) :
        return numpy.array([numpy.cos(self.plane.pitch)*numpy.cos(self.plane.bank),numpy.cos(self.plane.pitch)*numpy.sin(self.plane.bank),numpy.sin(self.plane.pitch)])

    def abs_speed(self) :
        return numpy.sqrt(self.speed[0]**2 + self.speed[1]**2 + self.speed[2]**2)

    def compute_state(self, dt=1) :
        resulting_force = 0
        for F in self.FORCES :
            F.compute_force()
            resulting_force += F

        accel = 1/self.MASS * resulting_force

        self.speed = self.speed + accel*dt
        
        self.pos = self.pos + self.speed*dt

def AircraftRequests() :
    def __init__(self, sm, _time) :
        self.plane = sm

    def get(arg) :
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
 
