#Simplification de la syntaxe SimConnect :


import simulator as smlt
import math
import time

class sim :
    def __init__(s) :
        sm = s
        sim.aq = smlt.AircraftRequests(sm, _time=2000)
        sim.ae = smlt.AircraftEvents(sm)

def get(arg):
    val = call(arg)
    return(val)

def call(arg):
    val = None
    if arg == "lat":
        val = sim.aq.get("PLANE_LATITUDE")
    if arg == "long":
        val = sim.aq.get("PLANE_LONGITUDE")
    if arg == "alt":
        val = sim.aq.get("PLANE_ALTITUDE")
    if arg == "r_alt":
        val = sim.aq.get("PLANE_ALT_ABOVE_GROUND")
    if arg == "bank":
        val = sim.aq.get("PLANE_BANK_DEGREES")
    if arg == "pitch":
        val = sim.aq.get("PLANE_PITCH_DEGREES")
    if arg == "abs_speed":
        val = sim.aq.get("AIRSPEED_TRUE")
    if arg == "x_speed":
        val = sim.aq.get("VELOCITY_BODY_X")
    if arg == "y_speed":
        val = sim.aq.get("VELOCITY_BODY_Y")
    if arg == "z_speed":
        val = sim.aq.get("VELOCITY_BODY_Z")
    if arg == "aileron":
        val = sim.aq.get("AILERON_POSITION")
    if arg == "elevator":
        val = sim.aq.get("ELEVATOR_POSITION")
    if arg == "throttle":
        val = sim.aq.get()
    if arg == "brake":
        val = sim.aq.get("BRAKE_INDICATOR")
    if arg == "cap":
        val = sim.aq.get("PLANE_HEADING_DEGREES_TRUE")

    if arg == "control_column" :
        val = sim.aq.get("CONTROL_COLUMN")
    return val

def set(arg, val) :
    if arg == "elevators" :
        sim.ae.set("ELEVATOR_POSITION", val)
    if arg == "control_column" :
        sim.ae.set("CONTROL_COLUMN", val)

#Pour tester un appel :
#get("cap")