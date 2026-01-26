# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:58:27 2025

Prereq python packages: matplotlib, numpy, scipy, tqdm, numba, gekko

How this works:
    Input aircraft velocity (Vinf) in m/s at your desired flight condition
    Input estimated RPM from your propulsion setup (this will take some iteration)
    Input density if specified 
    Denote the maximum mechanical power (Pmax, mech ~= Pmax,elec * eta_battery_to_propshaft)
        Can approximate as Pmax,elec * 0.85 to start
    Provide the wing area and drag coefficient, this will show the propeller that provide the required thrust at cruise
    
    prop_types = 'electric' is all APCE propellers 
      ...      = 'all' is the entire database
    
Then you will get an efficiency map with limits! 

Alternatively:
    Input aircraft Sw, CD, Plimit and let the optimization go!

@author: Sammy N. Nassau
"""
from APCPropellerTool import EfficiencyMap, OptimalEfficiencyMap, ThrustRPMEfficiencyMap

#%% Basic efficiency map
# Input Vinf, RPM --> get propeller efficiency over all pitches and diameters in the APC database
Vinf = 30       # freestream velocity, m/s
RPM = 7600    
Sw = 0.89        # wing area, m^2
CD = 0.03       # drag coefficient
Plimit = 900    # mechanical W; note that electrical W required to meet this will be higher

EfficiencyMap(Vinf, RPM, Sw = Sw, CD = CD, Plimit = Plimit, 
              prop_types = 'electric')

# runtime ~ 0.5s

#%% Optimal efficiency map
# Automatically determines the optimal Vinf, RPM to match power and thrust limits.
Sw = 0.89
CD = 0.03
Plimit = 1000   # shaft power limit in W
diamlimit = 23  # maximum propeller diameter in inches

# OptimalEfficiencyMap(Sw, CD, Plimit = Plimit,  
#                      diamlimit = 23, prop_types = 'electric')

# runtime ~ 1 minute

#%% Thrust-constrained optimal RPM efficiency map
# for a given design speed (Vinf) and aircraft parameters, optimize RPM for maximum propeller efficiency
Vinf = 30
Sw = 0.89
CD = 0.03
Plimit = 1000 
maxdiam = 23    # maximum propeller diameter in inches

# ThrustRPMEfficiencyMap(Vinf, Sw = Sw, CD=CD, Plimit=Plimit, 
#                        diamlimit = maxdiam, prop_types = 'electric')

# runtime ~2 minutes