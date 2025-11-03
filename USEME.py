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

from APCPropellerTool import EfficiencyMap, OptimalRPMEfficiencyMap, OptimalEfficiencyMap, ThrustRPMEfficiencyMap

#%% Example usage!
Vinf = 15.0 #m/s
RPM = 7600
Sw = 0.87
CD = 0.27
Plimit = 1100

# EfficiencyMap(Vinf, RPM, Sw = Sw, CD = CD, Plimit = Plimit)

#%% Optimal RPM Efficiency map (determines the best RPM for the given freestream velocity automatically)
# will take ~20-30s to run for all APC electric props and ~2 min 10s for all APC props
# NOTE: constraints will not always result in a feasible result since RPM is selected soley for max efficiency

# This is the optimal RPM DISCOUNTING constraints
# OptimalRPMEfficiencyMap(Vinf, Sw = Sw, CD = CD, 
#                        Plimit = Plimit, 
#                        prop_types = 'electric')
    
    
#%% Optimal Efficiency Map 
# Automatically determines unconstrained Vinf and RPM for maximum efficiency of each propeller
OptimalEfficiencyMap(prop_types = 'electric')

