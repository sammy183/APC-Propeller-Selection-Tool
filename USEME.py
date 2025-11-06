# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:58:27 2025

Prereq python packages: matplotlib, numpy, scipy

How this works:
    Input aircraft velocity (Vinf) in m/s at your desired flight condition
    Input estimated RPM from your propulsion setup (this will take some iteration)
    Input density if specified 
    Denote the maximum mechanical power (Pmax, mech ~= Pmax,elec * eta_battery_to_propshaft)
        Can approximate as Pmax,elec * 0.85 to start
    Provide the wing area and drag coefficient, this will show the propeller that provide the required thrust at cruise
    
Then you will get an efficiency map with limits! 

Alternatively:
    Input aircraft Sw, CD, Plimit and let the optimization go!

@author: NASSAS
"""
from APCPropellerTool import EfficiencyMap, OptimalRPMEfficiencyMap, OptimalEfficiencyMap, ThrustRPMEfficiencyMap, UnconstrainedOptimalEfficiencyMap

#%% Basic efficiency map
# Input Vinf, RPM --> get propeller efficiency over all pitches and diameters in the APC database
Vinf = 30 #m/s
RPM = 7600    
Sw = 0.6 # m^2
CD = 0.03
Plimit = 900 # mechanical W; note that electrical W required to meet this will be higher

EfficiencyMap(Vinf, RPM, Sw = Sw, CD = CD, Plimit = Plimit, prop_types = 'electric')
# runtime ~ 0.5s

#%% Optimal efficiency map
# Automatically determines the optimal Vinf, RPM to match constraints
Sw = 0.6
CD = 0.03
Plimit = 500

# OptimalEfficiencyMap(Sw, CD, Plimit = Plimit, prop_types = 'electric')
# runtime ~ 1 minute
#%% Thrust-constrained optimal RPM efficiency map
# for a given design speed (Vinf) and aircraft parameters, optimize RPM for maximum propeller efficiency
Vinf = 65
Sw = 0.6
CD = 0.022
Plimit = 3000
maxdiam = 19 # maximum propeller diameter in inches

# ThrustRPMEfficiencyMap(Vinf, Sw=Sw, CD=CD, Plimit=Plimit, diamlimit = maxdiam, prop_types = 'all')
# runtime ~2 minutes