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

from APCPropellerTool import EfficiencyMap

#%% Example usage!
Vinf = 30 #m/s
RPM = 7800 
Sw = 0.6
CD = 0.03
Plimit = 2700

EfficiencyMap(Vinf, RPM, Sw = Sw, CD = CD, Plimit = Plimit)