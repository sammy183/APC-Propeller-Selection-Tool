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

@author: NASSAS
"""
from APCPropellerTool import EfficiencyMap, OptimalRPMEfficiencyMap, OptimalEfficiencyMap, ThrustRPMEfficiencyMap, UnconstrainedOptimalEfficiencyMap

#%% Basic Efficiency map
Vinf = 30 #m/s
RPM = 6300 
Sw = 0.6
CD = 0.03
Plimit = 500

EfficiencyMap(Vinf, RPM, Sw = Sw, CD = CD, Plimit = Plimit)

#%% Optimal Efficiency Map
# takes longer to run but automatically determines the optimal Vinf, RPM to match constraints
Sw = 0.6
CD = 0.03
Plimit = 500

OptimalEfficiencyMap(Sw, CD, Plimit = Plimit, diamlimit = 23, prop_types = 'all')
