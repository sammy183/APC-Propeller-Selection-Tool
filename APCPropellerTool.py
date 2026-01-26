# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 18:30:58 2025

Function For Propeller Selection Via the APC Database

How it works:
    Input aircraft velocity (Vinf) in m/s at your desired flight condition
    Input estimated RPM from your propulsion setup (this will take some iteration)
    Input density if specified 
    Denote the maximum mechanical power (Pmax, mech ~= Pmax,elec * eta_battery_to_propshaft)
        Can approximate as Pmax,elec * 0.85 to start
    Provide the wing area and drag coefficient, this will show the propeller that provide the required thrust at cruise
    
Then you will get an efficiency map with limits! 
@author: NASSAS
"""

import os
# from gekko import GEKKO
import numpy as np
from numpy.polynomial import Polynomial
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import patheffects
import re
from propulsions import *
from tqdm import tqdm

inm = 0.0254
ftm = 0.3048

#%% MAIN FUNCTION
def EfficiencyMap(Vinf, RPM, 
                  rho = 1.225, Plimit = 1e6, 
                  Tlimit = 1e6, Sw = 0.0, CD = 0.0, 
                  prop_types = 'electric', path = 'PropDatabase', 
                  grade = 15):
    '''
    Inputs
    --------------------------------------------------------------------------------------
    Required:
        Vinf: freestream velocity in m/s
        RPM: propeller RPM
    Optional:
        rho: air density in kg/m^3 (default to 1.225)
        Plimit: maximum mechanical power limit (W)
        
        Tlimit: thrust in (N) to define the minimum required thrust directly
        Sw: wing area in (m^2) to define the minimum required thrust via T = D
        CD: drag coefficient to define the minimum required thrust via T = D
        (NOTE: if Tlimit is defined it will allways be the constraint over D from Sw, CD)
        
        prop_types: determines whether to use all APC props in the database or just APCE props. When using all props, several overlap
        path: filepath to the APC database from APCPropellerTool.py
        grade: increase to add more contourlines
        
    Output
    --------------------------------------------------------------------------------------
    Plots Propeller Efficiency vs Pitch and Diameter 
    
    '''
    props = os.listdir(path)
    props = props[:-1] #removes proplist.txt
    
    prop_APC = []
    if prop_types == 'electric':
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            # prop_APCE.append(propnameref)
            if 'WE' in propnameref:
                continue
            elif 'E.' in propnameref:
                prop_APC.append(propnameref)
    else:
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            prop_APC.append(propnameref)

    pitches = []
    diameters = []
    CTs = []
    CPs = []
    etas = []
    Ts = [] #N
    Ps = [] #W
    maxRPMs = []
    maxVs = []
    maxVspec = [] # maximum V at specified RPM


    for propname in prop_APC:
        with open(f'{path}/PER3_{propname}', 'r') as f:
            data_content = f.read()
            headline = data_content.splitlines()[0]
            headline = headline.split('x')
            for i, line in enumerate(headline):
                headline[i] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
        diam = float(headline[0])
        pitch = float(headline[1])
            
        # CT and CP for specified prop at specified RPM, J to see trend
        calcname = propname.split('.dat')[0]
        D = diam*inm
        PROP_DATA, numba_prop_data = parse_coef_propeller_data(calcname)
        rpm_values = np.array(PROP_DATA['rpm_list'])
        
        Jmax = 0.0
        for rpm in rpm_values:
                Jmax = max(PROP_DATA[rpm]['J'][-1], Jmax)
                
        # rpm_values, CT_polys, CP_polys, J_DOMAINS = initialize_RPM_polynomials(PROP_DATA)
        maxRPMs.append(rpm_values[-1])
        
        maxVs.append(Jmax*(rpm_values[-1]/60)*D)
        if RPM > rpm_values[-1]:
            maxVspec.append(0.0)
        else:
            maxVspec.append(Jmax*((RPM/60)*D)) # for this we want max J corresponding to the given RPM value!
        J = Vinf/((RPM/60)*D)
        CT = CTBase(RPM, J, rpm_values, numba_prop_data)
        CP = CPBase(RPM, J, rpm_values, numba_prop_data)
        Ts.append(rho*((RPM/60)**2)*(D**4)*CT)
        Ps.append(rho*((RPM/60)**3)*(D**5)*CP)
        CTs.append(CT)
        CPs.append(CP) 
        if CP != 0.0:
            etas.append((CT*J/CP))
        else:
            etas.append(0.0)
        diameters.append(float(diam))
        pitches.append(float(pitch))
    
    etas = np.array(etas)
    etanozero = etas[etas > 0.0]
    if etanozero.size == 0:
        raise KeyError('No feasible propellers for selected RPM, Vinf!\nIncrease RPM or decrease Vinf')
    mask = etas > etanozero.min()
    etas = etas[mask]
    Ts = np.array(Ts)[mask]
    Ps = np.array(Ps)[mask]
    diameters = np.array(diameters)[mask]
    pitches = np.array(pitches)[mask]
    goodidx = np.array([False]*etas.size)
    
    fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
    img = ax.tricontourf(diameters, pitches, etas*100, levels = grade)#np.linspace(0, 1200, 15))

    clabelsize = 6
    spacing = 50
    if Tlimit < 1e6:
        lines = ax.tricontour(diameters, pitches, Ts, levels = [Tlimit], colors = '#cc0000')
        ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
        lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        goodidx[Ts < Tlimit] = True
    
    if Sw > 0.0 and CD > 0.0 and Tlimit == 1e6:
        D = 0.5*rho*(Vinf**2)*Sw*CD
        lines = ax.tricontour(diameters, pitches, Ts, levels = [D], colors = '#cc0000')
        ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
        lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        goodidx[Ts < D] = True
        
    if Plimit < 1e6:
        lines2 = ax.tricontour(diameters, pitches, Ps/1000, levels = [Plimit/1000], colors = 'orangered')
        ax.clabel(lines2, levels = lines2.levels, fmt = '%.1f kW', fontsize = clabelsize, inline_spacing = spacing)
        lines2.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
        goodidx[Ps > Plimit] = True
        
    # # iffy stall line
    if Vinf < 15:
        stalldiams = np.linspace(np.array(diameters).min(), np.array(pitches).max()*3/2-0.1, 100)
        stallpitches = stalldiams*0.667 # ecalc approx stall where P/D > 0.667
        ax.plot(stalldiams, stallpitches, color = '#cc0000', label = 'Stall', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
        ax.legend()
        
    # location of maximum efficiency
    maxidx = np.argmax(etas)
    if diameters[maxidx].is_integer():
        diammax = int(diameters[maxidx])
    else:
        diammax = diameters[maxidx]
    if pitches[maxidx].is_integer():
        pitchmax = int(pitches[maxidx])
    else:
        pitchmax = pitches[maxidx]
        
    if prop_types == 'electric':
        useE = 'E'
    else:
        useE = ''
    print(f'Maximum Propeller Efficiency (unconstrained) is {etas.max()*100:.1f}% with the {diammax}x{pitchmax}{useE}')
        
    # location of maximum constrained efficiency (via masks)
    eta_adjust = etas.copy()
    eta_adjust[goodidx] = 0.0
    if eta_adjust.max() == 0.0:
        print('ERROR: Constraints infeasible for propellers at RPM of maximum efficiency!')
    else:
        con_maxidx = np.argmax(eta_adjust)
        if diameters[con_maxidx].is_integer():
            con_diammax = int(diameters[con_maxidx])
        else:
            con_diammax = diameters[con_maxidx]
        if pitches[con_maxidx].is_integer():
            con_pitchmax = int(pitches[con_maxidx])
        else:
            con_pitchmax = pitches[con_maxidx]
        print(f'Maximum Propeller Efficiency (constrained) is {eta_adjust.max()*100:.1f}% with the {con_diammax}x{con_pitchmax}{useE}')
        plt.scatter(con_diammax, con_pitchmax, marker = 'x', color = 'blue', label = f'Max Constrained; {con_diammax}x{con_pitchmax}{useE}; {eta_adjust.max()*100:.1f}% $\\eta_p$')
        
    plt.scatter(diammax, pitchmax, marker = '^', color = 'black', label = f'Max Unconstrained; {diammax}x{pitchmax}{useE}; {etas.max()*100:.1f}% $\\eta_p$')

    plt.legend(fontsize = 7, loc = 'lower right')
    plt.colorbar(img, label = r'$\eta_p$ (%)')
    # img = ax.scatter(diameters, pitches)
    plt.xlabel('Diameter (in)')
    plt.ylabel('Pitch (in)')
    plt.title(f'APC Propeller Efficiencies at {RPM} RPM and {Vinf:.2f} m/s')
    plt.minorticks_on()
    plt.grid()
    plt.show()

def OptimalRPMEfficiencyMap(Vinf, 
                  rho = 1.225, Plimit = 1e6, 
                  Tlimit = 1e6, Sw = 0.0, CD = 0.0, 
                  prop_types = 'electric', path = 'PropDatabase', 
                  grade = 15):
    '''
    Inputs
    --------------------------------------------------------------------------------------
    Required:
        Vinf: freestream velocity in m/s
    Optional:
        rho: air density in kg/m^3 (default to 1.225)
        Plimit: maximum mechanical power limit (W)
        
        Tlimit: thrust in (N) to define the minimum required thrust directly
        Sw: wing area in (m^2) to define the minimum required thrust via T = D
        CD: drag coefficient to define the minimum required thrust via T = D
        (NOTE: if Tlimit is defined it will allways be the constraint over D from Sw, CD)
        
        prop_types: determines whether to use all APC props in the database or just APCE props. When using all props, several overlap
        path: filepath to the APC database from APCPropellerTool.py
        grade: increase to add more contourlines
        
    Output
    --------------------------------------------------------------------------------------
    Plots Optimal Propeller Efficiency vs Pitch and Diameter (with RPM contours)
    
    '''
    props = os.listdir(path)
    props = props[:-1] #removes proplist.txt
    
    prop_APC = []
    if prop_types == 'electric':
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            # prop_APCE.append(propnameref)
            if 'WE' in propnameref:
                continue
            elif 'E.' in propnameref:
                prop_APC.append(propnameref)
    else:
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            prop_APC.append(propnameref)

    pitches = []
    diameters = []
    CTs = []
    CPs = []
    etas = []
    Ts = [] #N
    Ps = [] #W
    RPMs = []
    for propname in tqdm(prop_APC):
        with open(f'{path}/PER3_{propname}', 'r') as f:
            data_content = f.read()
            headline = data_content.splitlines()[0]
            headline = headline.split('x')
            for i, line in enumerate(headline):
                headline[i] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
        diam = float(headline[0])
        pitch = float(headline[1])
            
        # CT and CP for specified prop at specified RPM, J to see trend
        calcname = propname.split('.dat')[0]
        D = diam*inm
        PROP_DATA, numba_prop_data = parse_coef_propeller_data(calcname)
        rpm_values = np.array(PROP_DATA['rpm_list'])
                        
        # rpm_values, CT_polys, CP_polys, J_DOMAINS = initialize_RPM_polynomials(PROP_DATA)
        eta, RPM = OptRPMeta(Vinf, rpm_values, numba_prop_data, D, ns = 150)
        RPMs.append(RPM)
        J = Vinf/((RPM/60)*D)
        CT = CTBase(RPM, J, rpm_values, numba_prop_data)
        CP = CPBase(RPM, J, rpm_values, numba_prop_data)
        Ts.append(rho*((RPM/60)**2)*(D**4)*CT)
        Ps.append(rho*((RPM/60)**3)*(D**5)*CP)
        CTs.append(CT)
        CPs.append(CP) 
        if CP != 0.0:
            etas.append((CT*J/CP))
        else:
            etas.append(0.0)
        diameters.append(float(diam))
        pitches.append(float(pitch))
    
    etas = np.array(etas)
    etanozero = etas[etas > 0.0]
    mask = etas > etanozero.min()
    etas = etas[mask]
    Ts = np.array(Ts)[mask]
    Ps = np.array(Ps)[mask]
    RPMs = np.array(RPMs)[mask]
    diameters = np.array(diameters)[mask]
    pitches = np.array(pitches)[mask]
    goodidx = np.array([False]*etas.size)

    fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
    img = ax.tricontourf(diameters, pitches, etas*100, levels = grade)#np.linspace(0, 1200, 15))

    clabelsize = 6
    spacing = 50
    # if Tlimit < 1e6:
    #     lines = ax.tricontour(diameters, pitches, Ts, levels = [Tlimit], colors = '#cc0000')
    #     ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
    #     lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
    #     goodidx[Ts < Tlimit] = True
    
    # if Sw > 0.0 and CD > 0.0 and Tlimit == 1e6:
    #     D = 0.5*rho*(Vinf**2)*Sw*CD
    #     lines = ax.tricontour(diameters, pitches, Ts, levels = [D], colors = '#cc0000')
    #     ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
    #     lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
    #     goodidx[Ts < D] = True
        
    # if Plimit < 1e6:
    #     lines2 = ax.tricontour(diameters, pitches, Ps/1000, levels = [Plimit/1000], colors = 'orangered')
    #     ax.clabel(lines2, levels = lines2.levels, fmt = '%.0f kW', fontsize = clabelsize, inline_spacing = spacing)
    #     lines2.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
    #     goodidx[Ps > Plimit] = True
        
    # # iffy stall line
    # if Vinf < 15:
    #     stalldiams = np.linspace(np.array(diameters).min(), np.array(pitches).max()*3/2-0.1, 100)
    #     stallpitches = stalldiams*0.667 # ecalc approx stall where P/D > 0.667
    #     ax.plot(stalldiams, stallpitches, color = '#cc0000', label = 'Stall', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
    #     ax.legend()
        
    lines3 = ax.tricontour(diameters, pitches, RPMs/1000, levels = 9, colors = 'k')
    ax.clabel(lines3, levels = lines3.levels, fmt = '%.0f kRPM', fontsize = clabelsize, inline_spacing = spacing)

    # lines4 = ax.tricontour(diameters, pitches, Ts, levels = 9)
    # ax.clabel(lines4, levels = lines4.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)

    # lines5 = ax.tricontour(diameters, pitches, Ps, levels = 9, colors = 'k')
    # ax.clabel(lines5, levels = lines5.levels, fmt = '%.1f W', fontsize = clabelsize, inline_spacing = spacing)

    # location of maximum efficiency
    maxidx = np.argmax(etas)
    if diameters[maxidx].is_integer():
        diammax = int(diameters[maxidx])
    else:
        diammax = diameters[maxidx]
    if pitches[maxidx].is_integer():
        pitchmax = int(pitches[maxidx])
    else:
        pitchmax = pitches[maxidx]
        
    if prop_types == 'electric':
        useE = 'E'
    else:
        useE = ''
    print(f'Maximum Propeller Efficiency (unconstrained) is {etas.max()*100:.1f}% with the {diammax}x{pitchmax}{useE} at {RPMs[maxidx]:.0f} RPM')
        
    # location of maximum constrained efficiency (via masks)
    # eta_adjust = etas.copy()
    # eta_adjust[goodidx] = 0.0
    # if eta_adjust.max() == 0.0:
    #     print('ERROR: Constraints infeasible for propellers at RPM of maximum efficiency!')
    # else:
    #     con_maxidx = np.argmax(eta_adjust)
    #     if diameters[con_maxidx].is_integer():
    #         con_diammax = int(diameters[con_maxidx])
    #     else:
    #         con_diammax = diameters[con_maxidx]
    #     if pitches[con_maxidx].is_integer():
    #         con_pitchmax = int(pitches[con_maxidx])
    #     else:
    #         con_pitchmax = pitches[con_maxidx]
    #     print(f'Maximum Propeller Efficiency (constrained) is {eta_adjust.max()*100:.1f}% with the {con_diammax}x{con_pitchmax}{useE} at {RPMs[con_maxidx]:.0f} RPM')
    #     plt.scatter(con_diammax, con_pitchmax, marker = 'x', color = 'blue', label = f'Max Constrained; {con_diammax}x{con_pitchmax}{useE}; {eta_adjust.max()*100:.1f}% $\\eta_p$; {RPMs[con_maxidx]:.0f} RPM')

    plt.scatter(diammax, pitchmax, marker = '^', color = 'black', label = f'Max Unconstrained; {diammax}x{pitchmax}{useE}; {etas.max()*100:.1f}% $\\eta_p$; {RPMs[maxidx]:.0f} RPM')

    plt.legend(fontsize = 7, loc = 'lower right')
    plt.colorbar(img, label = r'$\eta_p$ (%)')
    # img = ax.scatter(diameters, pitches)
    plt.xlabel('Diameter (in)')
    plt.ylabel('Pitch (in)')
    plt.title(f'APC Propeller Efficiencies at {Vinf:.2f} m/s and Optimal RPMs')
    plt.minorticks_on()
    plt.grid()
    plt.show()
    
def ThrustRPMEfficiencyMap(Vinf, 
                  rho = 1.225, Plimit = 1e6, diamlimit = 1e6,
                  Tlimit = 1e6, Sw = 0.0, CD = 0.0, 
                  prop_types = 'electric', path = 'PropDatabase', 
                  grade = 15):
    '''
    Inputs
    --------------------------------------------------------------------------------------
    Required:
        Vinf: freestream velocity in m/s
    Optional:
        rho: air density in kg/m^3 (default to 1.225)
        Plimit: maximum mechanical power limit (W)
        
        Tlimit: thrust in (N) to define the minimum required thrust directly
        Sw: wing area in (m^2) to define the minimum required thrust via T = D
        CD: drag coefficient to define the minimum required thrust via T = D
        (NOTE: if Tlimit is defined it will allways be the constraint over D from Sw, CD)
        
        prop_types: determines whether to use all APC props in the database or just APCE props. When using all props, several overlap
        path: filepath to the APC database from APCPropellerTool.py
        grade: increase to add more contourlines
        
    Output
    --------------------------------------------------------------------------------------
    Plots Optimal Propeller Efficiency vs Pitch and Diameter (with RPM contours)
    
    selects RPM such that T = D (or Tlimit) at the given Velocity
    '''
    props = os.listdir(path)
    props = props[:-1] #removes proplist.txt
    
    # props = ['PER3_16x10E.dat', 'PER3_16x12E.dat', 
    #          'PER3_18x12E.dat', 'PER3_22x12E.dat',
    #          'PER3_12x12E.dat', 'PER3_14x14E.dat',
    #          'PER3_18x8E.dat']
    prop_APC = []
    if prop_types == 'electric':
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            # prop_APCE.append(propnameref)
            if 'WE' in propnameref:
                continue
            elif 'E.' in propnameref:
                prop_APC.append(propnameref)
    else:
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            prop_APC.append(propnameref)

    if Tlimit < 1e6:
        Treq = Tlimit 
    elif Sw > 0.0 and CD > 0.0:
        Treq = 0.5*rho*(Vinf**2)*Sw*CD
        
    pitches = []
    diameters = []
    etas = []
    Ts = [] #N
    Ps = [] #W
    RPMs = []
    for propname in tqdm(prop_APC):
        with open(f'{path}/PER3_{propname}', 'r') as f:
            data_content = f.read()
            headline = data_content.splitlines()[0]
            headline = headline.split('x')
            for i, line in enumerate(headline):
                headline[i] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
        diam = float(headline[0])
        pitch = float(headline[1])
            
        # CT and CP for specified prop at specified RPM, J to see trend
        calcname = propname.split('.dat')[0]
        D = diam*inm
        PROP_DATA, numba_prop_data = parse_coef_propeller_data(calcname)
        rpm_values = np.array(PROP_DATA['rpm_list'])
        
        # RPM and eta corresponding to T = D
        try:
            eta, RPM = ThrustRPMeta(Vinf, Treq, rho, rpm_values, numba_prop_data, D, ns = 150)
        except:
            continue
        RPMs.append(RPM)
        etas.append(eta)
        J = Vinf/((RPM/60)*D)
        CT = CTBase(RPM, J, rpm_values, numba_prop_data)
        CP = CPBase(RPM, J, rpm_values, numba_prop_data)
        Ts.append(rho*((RPM/60)**2)*(D**4)*CT)
        Ps.append(rho*((RPM/60)**3)*(D**5)*CP)
        diameters.append(float(diam))
        pitches.append(float(pitch))
    
    etas = np.array(etas)
    etanozero = etas[etas > 0.0]
    mask = etas > etanozero.min()
    etas = etas[mask]
    Ts = np.array(Ts)[mask]
    Ps = np.array(Ps)[mask]
    RPMs = np.array(RPMs)[mask]
    diameters = np.array(diameters)[mask]
    pitches = np.array(pitches)[mask]
    goodidx = np.array([False]*etas.size)

    fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
    img = ax.tricontourf(diameters, pitches, etas*100, levels = grade)

    clabelsize = 6
    spacing = 50
    # if Tlimit < 1e6:
    #     lines = ax.tricontour(diameters, pitches, Ts, levels = [Tlimit], colors = '#cc0000')
    #     ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
    #     lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
    #     goodidx[Ts < Tlimit] = True
    
    # if Sw > 0.0 and CD > 0.0 and Tlimit == 1e6:
    #     D = 0.5*rho*(Vinf**2)*Sw*CD
    #     lines = ax.tricontour(diameters, pitches, Ts, levels = [D], colors = '#cc0000')
    #     ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
    #     lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
    #     goodidx[Ts < D] = True
        
    if Plimit < 1e6:
        lines2 = ax.tricontour(diameters, pitches, Ps/1000, levels = [Plimit/1000], colors = 'orangered')
        ax.clabel(lines2, levels = lines2.levels, fmt = '%.1f kW', fontsize = clabelsize, inline_spacing = spacing)
        lines2.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
        goodidx[Ps > Plimit] = True
        
    if diamlimit < 1e6:
        lines2 = ax.plot([diamlimit, diamlimit], ax.get_ylim(), color = '#cc0000', 
                         path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        goodidx[diameters > diamlimit] = True
        
    # # iffy stall line
    # if Vinf < 15:
    #     stalldiams = np.linspace(np.array(diameters).min(), np.array(pitches).max()*3/2-0.1, 100)
    #     stallpitches = stalldiams*0.667 # ecalc approx stall where P/D > 0.667
    #     ax.plot(stalldiams, stallpitches, color = '#cc0000', label = 'Stall', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
    #     ax.legend()
        
    lines3 = ax.tricontour(diameters, pitches, RPMs/1000, levels = 5, colors = 'k')
    ax.clabel(lines3, levels = lines3.levels, fmt = '%.1f kRPM', fontsize = clabelsize, inline_spacing = spacing)

    # lines4 = ax.tricontour(diameters, pitches, Ts, levels = 9)
    # ax.clabel(lines4, levels = lines4.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)

    # lines5 = ax.tricontour(diameters, pitches, Ps/1000, levels = 5, colors = 'orangered')
    # ax.clabel(lines5, levels = lines5.levels, fmt = '%.1f kW', fontsize = clabelsize, inline_spacing = spacing)

    # location of maximum efficiency
    maxidx = np.argmax(etas)
    if diameters[maxidx].is_integer():
        diammax = int(diameters[maxidx])
    else:
        diammax = diameters[maxidx]
    if pitches[maxidx].is_integer():
        pitchmax = int(pitches[maxidx])
    else:
        pitchmax = pitches[maxidx]
        
    if prop_types == 'electric':
        useE = 'E'
    else:
        useE = ''
    print(f'Maximum Propeller Efficiency (unconstrained) is {etas.max()*100:.1f}% with the {diammax}x{pitchmax}{useE} at {RPMs[maxidx]:.0f} RPM')
        
    # location of maximum constrained efficiency (via masks)
    eta_adjust = etas.copy()
    eta_adjust[goodidx] = 0.0
    if eta_adjust.max() == 0.0:
        print('ERROR: Constraints infeasible for propellers at RPM of maximum efficiency!')
    else:
        con_maxidx = np.argmax(eta_adjust)
        if diameters[con_maxidx].is_integer():
            con_diammax = int(diameters[con_maxidx])
        else:
            con_diammax = diameters[con_maxidx]
        if pitches[con_maxidx].is_integer():
            con_pitchmax = int(pitches[con_maxidx])
        else:
            con_pitchmax = pitches[con_maxidx]
        print(f'Maximum Propeller Efficiency (constrained) is {eta_adjust.max()*100:.1f}% with the {con_diammax}x{con_pitchmax}{useE} at {RPMs[con_maxidx]:.0f} RPM')
        plt.scatter(con_diammax, con_pitchmax, marker = 'x', color = 'blue', label = f'Max Constrained\n{con_diammax}x{con_pitchmax}{useE}; {eta_adjust.max()*100:.1f}% $\\eta_p$; {RPMs[con_maxidx]:.0f} RPM, {Ps[con_maxidx]:.0f} W')

    plt.scatter(diammax, pitchmax, marker = '^', color = 'black', label = f'Max Unconstrained\n{diammax}x{pitchmax}{useE}; {etas.max()*100:.1f}% $\\eta_p$; {RPMs[maxidx]:.0f} RPM; {Ps[maxidx]:.0f} W')

    plt.legend(fontsize = 7, loc = 'lower right')
    plt.colorbar(img, label = r'$\eta_p$ (%)')
    # img = ax.scatter(diameters, pitches)
    plt.xlabel('Diameter (in)')
    plt.ylabel('Pitch (in)')
    plt.title(f'APC Propeller Efficiencies and RPMs at {Vinf:.2f} m/s with T = {Treq:.1f} N')
    plt.minorticks_on()
    plt.grid()
    plt.show()

    
def UnconstrainedOptimalEfficiencyMap(rho = 1.225, Plimit = 1e6, 
                  Tlimit = 1e6, Sw = 0.0, CD = 0.0, 
                  prop_types = 'electric', path = 'PropDatabase', 
                  grade = 15):
    '''
    Inputs
    --------------------------------------------------------------------------------------
    Required:
        Vinf: freestream velocity in m/s
    Optional:
        rho: air density in kg/m^3 (default to 1.225)
        Plimit: maximum mechanical power limit (W)
        
        Tlimit: thrust in (N) to define the minimum required thrust directly
        Sw: wing area in (m^2) to define the minimum required thrust via T = D
        CD: drag coefficient to define the minimum required thrust via T = D
        (NOTE: if Tlimit is defined it will allways be the constraint over D from Sw, CD)
        
        prop_types: determines whether to use all APC props in the database or just APCE props. When using all props, several overlap
        path: filepath to the APC database from APCPropellerTool.py
        grade: increase to add more contourlines
        
    Output
    --------------------------------------------------------------------------------------
    Plots Optimal Propeller Efficiency vs Pitch and Diameter (with RPM contours)
    
    '''
    props = os.listdir(path)
    props = props[:-1] #removes proplist.txt
    
    prop_APC = []
    if prop_types == 'electric':
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            # prop_APCE.append(propnameref)
            if 'WE' in propnameref:
                continue
            elif 'E.' in propnameref:
                prop_APC.append(propnameref)
    else:
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            prop_APC.append(propnameref)

    pitches = []
    diameters = []
    CTs = []
    CPs = []
    etas = []
    Ts = [] #N
    Ps = [] #W
    RPMs = []
    Vinfs = []
    for propname in tqdm(prop_APC):
        with open(f'{path}/PER3_{propname}', 'r') as f:
            data_content = f.read()
            headline = data_content.splitlines()[0]
            headline = headline.split('x')
            for i, line in enumerate(headline):
                headline[i] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
        diam = float(headline[0])
        pitch = float(headline[1])
            
        # CT and CP for specified prop at specified RPM, J to see trend
        calcname = propname.split('.dat')[0]
        D = diam*inm
        PROP_DATA, numba_prop_data = parse_coef_propeller_data(calcname)
        rpm_values = np.array(PROP_DATA['rpm_list'])
        
        # rpm_values, CT_polys, CP_polys, J_DOMAINS = initialize_RPM_polynomials(PROP_DATA)
        optRPM, optVinf, optEta = OptEta(rpm_values, numba_prop_data, D, ns = 150)
        Vinfs.append(optVinf)
        RPMs.append(optRPM)
        etas.append(optEta)
        J = optVinf/((optRPM/60)*D)
        CT = CTBase(optRPM, J, rpm_values, numba_prop_data)
        CP = CPBase(optRPM, J, rpm_values, numba_prop_data)
        Ts.append(rho*((optRPM/60)**2)*(D**4)*CT)
        Ps.append(rho*((optRPM/60)**3)*(D**5)*CP)
        CTs.append(CT)
        CPs.append(CP) 
        diameters.append(float(diam))
        pitches.append(float(pitch))
    
    etas = np.array(etas)
    etanozero = etas[etas > 0.0]
    mask = etas > etanozero.min()
    etas = etas[mask]
    Ts = np.array(Ts)[mask]
    Ps = np.array(Ps)[mask]
    RPMs = np.array(RPMs)[mask]
    Vinfs = np.array(Vinfs)[mask]
    diameters = np.array(diameters)[mask]
    pitches = np.array(pitches)[mask]
    goodidx = np.array([False]*etas.size)

    fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
    img = ax.tricontourf(diameters, pitches, etas*100, levels = grade)#np.linspace(0, 1200, 15))
    # img = ax.scatter(diameters, pitches, c=etas*100, s = 3)#, levels = grade)#np.linspace(0, 1200, 15))

    clabelsize = 6
    spacing = 50
    # if Tlimit < 1e6:
    #     lines = ax.tricontour(diameters, pitches, Ts, levels = 5, colors = '#cc0000')
    #     ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
    #     lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
    #     goodidx[Ts < Tlimit] = True
    
    # if Sw > 0.0 and CD > 0.0 and Tlimit == 1e6:
    #     D = 0.5*rho*(Vinfs**2)*Sw*CD
    #     lines = ax.tricontour(diameters, pitches, Ts-D, levels = 5, colors = '#666666')
    #     ax.clabel(lines, levels = lines.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)
    #     # lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
    #     goodidx[Ts < D] = True
        
    # if Plimit < 1e6:
    #     lines2 = ax.tricontour(diameters, pitches, Ps/1000, levels = 5, colors = 'orangered')
    #     ax.clabel(lines2, levels = lines2.levels, fmt = '%.0f kW', fontsize = clabelsize, inline_spacing = spacing)
    #     # lines2.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
    #     goodidx[Ps > Plimit] = True
        
    # # iffy stall line
    # if Vinf < 15:
    #     stalldiams = np.linspace(np.array(diameters).min(), np.array(pitches).max()*3/2-0.1, 100)
    #     stallpitches = stalldiams*0.667 # ecalc approx stall where P/D > 0.667
    #     ax.plot(stalldiams, stallpitches, color = '#cc0000', label = 'Stall', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
    #     ax.legend()
        
    lines3 = ax.tricontour(diameters, pitches, RPMs/1000, levels = 5, colors = 'k')
    ax.clabel(lines3, levels = lines3.levels, fmt = '%.0f kRPM', fontsize = clabelsize, inline_spacing = spacing)

    lines4 = ax.tricontour(diameters, pitches, Vinfs, levels = 5, colors = '#cc0000')
    ax.clabel(lines4, levels = lines4.levels, fmt = '%.0f m/s', fontsize = clabelsize, inline_spacing = spacing)

    # lines4 = ax.tricontour(diameters, pitches, Ts, levels = 9)
    # ax.clabel(lines4, levels = lines4.levels, fmt = '%.1f N', fontsize = clabelsize, inline_spacing = spacing)

    # lines5 = ax.tricontour(diameters, pitches, Ps, levels = 9, colors = 'k')
    # ax.clabel(lines5, levels = lines5.levels, fmt = '%.1f W', fontsize = clabelsize, inline_spacing = spacing)

    # location of maximum efficiency
    maxidx = np.argmax(etas)
    if diameters[maxidx].is_integer():
        diammax = int(diameters[maxidx])
    else:
        diammax = diameters[maxidx]
    if pitches[maxidx].is_integer():
        pitchmax = int(pitches[maxidx])
    else:
        pitchmax = pitches[maxidx]
        
    if prop_types == 'electric':
        useE = 'E'
    else:
        useE = ''
    print(f'Maximum Propeller Efficiency (unconstrained) is {etas.max()*100:.1f}% with the {diammax}x{pitchmax}{useE} at {RPMs[maxidx]:.0f} RPM and {Vinfs[maxidx]:.1f} m/s')
        
    # # location of maximum constrained efficiency (via masks)
    # eta_adjust = etas.copy()
    # eta_adjust[goodidx] = 0.0
    # if eta_adjust.max() == 0.0:
    #     print('ERROR: Constraints infeasible for propellers at RPM of maximum efficiency!')
    # else:
    #     con_maxidx = np.argmax(eta_adjust)
    #     if diameters[con_maxidx].is_integer():
    #         con_diammax = int(diameters[con_maxidx])
    #     else:
    #         con_diammax = diameters[con_maxidx]
    #     if pitches[con_maxidx].is_integer():
    #         con_pitchmax = int(pitches[con_maxidx])
    #     else:
    #         con_pitchmax = pitches[con_maxidx]
    #     print(f'Maximum Propeller Efficiency (constrained) is {eta_adjust.max()*100:.1f}% with the {con_diammax}x{con_pitchmax}{useE} at {RPMs[con_maxidx]:.0f} RPM and {Vinfs[con_maxidx]:.1f} m/s')
    #     plt.scatter(con_diammax, con_pitchmax, marker = 'x', color = 'blue', label = f'Max Constrained\n{con_diammax}x{con_pitchmax}{useE}\n{eta_adjust.max()*100:.1f}% $\\eta_p$\n{RPMs[con_maxidx]:.0f} RPM\n{Vinfs[con_maxidx]:.1f} m/s')

    plt.scatter(diammax, pitchmax, marker = '^', color = 'black', label = f'Maximum\n{diammax}x{pitchmax}{useE}\n{etas.max()*100:.1f}% $\\eta_p$\n{RPMs[maxidx]:.0f} RPM\n{Vinfs[maxidx]:.1f} m/s')

    plt.legend(fontsize = 7)
    plt.colorbar(img, label = r'$\eta_p$ (%)')
    # img = ax.scatter(diameters, pitches)
    plt.xlabel('Diameter (in)')
    plt.ylabel('Pitch (in)')
    plt.title('APC Propeller Efficiencies at Optimal Vinf and Optimal RPM')
    plt.minorticks_on()
    plt.grid()
    plt.show()
    
   
#%% Optimal automation function: for given Sw, CD, Pmax, diameter max, find optimal etas for all props
def OptimalEfficiencyMap(Sw, CD, rho = 1.225, Plimit = 1e6, 
                  Tlimit = 1e6, diamlimit = 1e6, 
                  prop_types = 'electric', path = 'PropDatabase', 
                  grade = 15):
    '''
    Inputs
    --------------------------------------------------------------------------------------
    Required:
        Vinf: freestream velocity in m/s
    Optional:
        rho: air density in kg/m^3 (default to 1.225)
        Plimit: maximum mechanical power limit (W)
        
        Tlimit: thrust in (N) to define the minimum required thrust directly
        Sw: wing area in (m^2) to define the minimum required thrust via T = D
        CD: drag coefficient to define the minimum required thrust via T = D
        (NOTE: if Tlimit is defined it will allways be the constraint over D from Sw, CD)
        
        prop_types: determines whether to use all APC props in the database or just APCE props. When using all props, several overlap
        path: filepath to the APC database from APCPropellerTool.py
        grade: increase to add more contourlines
        
    Output
    --------------------------------------------------------------------------------------
    Plots Optimal Propeller Efficiency vs Pitch and Diameter (with RPM contours)
    
    '''
    props = os.listdir(path)
    props = props[:-1] #removes proplist.txt
    
    prop_APC = []
    if prop_types == 'electric':
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            # prop_APCE.append(propnameref)
            if 'WE' in propnameref:
                continue
            elif 'E.' in propnameref:
                prop_APC.append(propnameref)
    else:
        for i, prop in enumerate(props):
            propnameref = prop.split('_')[1]
            prop_APC.append(propnameref)

    pitches = []
    diameters = []
    CTs = []
    CPs = []
    etas = []
    Ts = [] #N
    Ps = [] #W
    RPMs = []
    Vinfs = []
    for propname in tqdm(prop_APC):
        with open(f'{path}/PER3_{propname}', 'r') as f:
            data_content = f.read()
            headline = data_content.splitlines()[0]
            headline = headline.split('x')
            for i, line in enumerate(headline):
                headline[i] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
        diam = float(headline[0])
        pitch = float(headline[1])
            
        # CT and CP for specified prop at specified RPM, J to see trend
        calcname = propname.split('.dat')[0]
        D = diam*inm
        PROP_DATA, numba_prop_data = parse_coef_propeller_data(calcname)
        rpm_values = np.array(PROP_DATA['rpm_list'])
        
        # rpm_values, CT_polys, CP_polys, J_DOMAINS = initialize_RPM_polynomials(PROP_DATA)
        optRPM, optVinf, optEta = ConsOptEta(Sw, CD, Plimit, rho, diam*inm, rpm_values, numba_prop_data)
        Vinfs.append(optVinf)
        RPMs.append(optRPM)
        etas.append(optEta)
        J = optVinf/((optRPM/60)*D)
        CT = CTBase(optRPM, J, rpm_values, numba_prop_data)
        CP = CPBase(optRPM, J, rpm_values, numba_prop_data)
        Ts.append(rho*((optRPM/60)**2)*(D**4)*CT)
        Ps.append(rho*((optRPM/60)**3)*(D**5)*CP)
        CTs.append(CT)
        CPs.append(CP) 
        diameters.append(float(diam))
        pitches.append(float(pitch))
    
    etas = np.array(etas)
    etanozero = etas[etas > 0.0]
    mask = etas > etanozero.min()
    etas = etas[mask]
    Ts = np.array(Ts)[mask]
    Ps = np.array(Ps)[mask]
    RPMs = np.array(RPMs)[mask]
    Vinfs = np.array(Vinfs)[mask]
    diameters = np.array(diameters)[mask]
    pitches = np.array(pitches)[mask]
    goodidx = np.array([False]*etas.size)

    fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
    img = ax.tricontourf(diameters, pitches, etas*100, levels = grade)#np.linspace(0, 1200, 15))
    clabelsize = 6
    spacing = 50
    if Plimit < 1e6:
        lines2 = ax.tricontour(diameters, pitches, Ps, levels = [Plimit], colors = 'orangered')
        ax.clabel(lines2, levels = lines2.levels, fmt = '%.0f W', fontsize = clabelsize, inline_spacing = spacing)
        lines2.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
        goodidx[Ps > Plimit] = True
        
    if diamlimit < 1e6:
        lines2 = ax.plot([diamlimit, diamlimit], ax.get_ylim(), color = '#cc0000', 
                         path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        goodidx[diameters > diamlimit] = True

        
    lines3 = ax.tricontour(diameters, pitches, RPMs/1000, levels = 3, colors = 'k')
    ax.clabel(lines3, levels = lines3.levels, fmt = '%.0f kRPM', fontsize = clabelsize, inline_spacing = spacing)

    lines4 = ax.tricontour(diameters, pitches, Vinfs, levels =  [np.floor(np.max(Vinfs))*0.833, np.floor(np.max(Vinfs))*0.966, np.floor(np.max(Vinfs))], colors = '#cc0000')
    ax.clabel(lines4, levels = lines4.levels, fmt = '%.0f m/s', fontsize = clabelsize, inline_spacing = spacing)
    # lines4 = ax.tricontour(diameters, pitches, Ts, levels =  np.linspace(0.8*np.max(Ts), np.max(Ts), 3), colors = '#666666')
    # ax.clabel(lines4, levels = lines4.levels, fmt = '%.0f N', fontsize = clabelsize, inline_spacing = spacing)
    # lines4 = ax.tricontour(diameters, pitches, Ps/1000, levels = [np.floor(np.max(Ps/1000))*0.833, np.floor(np.max(Ps/1000))*0.966, np.floor(np.max(Ps/1000))], colors = 'orangered')
    # ax.clabel(lines4, levels = lines4.levels, fmt = '%.3f kW', fontsize = clabelsize, inline_spacing = spacing)

    # location of maximum efficiency
    maxidx = np.argmax(etas)
    if diameters[maxidx].is_integer():
        diammax = int(diameters[maxidx])
    else:
        diammax = diameters[maxidx]
    if pitches[maxidx].is_integer():
        pitchmax = int(pitches[maxidx])
    else:
        pitchmax = pitches[maxidx]
        
    if prop_types == 'electric':
        useE = 'E'
    else:
        useE = ''
    print(f'Maximum Propeller Efficiency is {etas.max()*100:.1f}% with the {diammax}x{pitchmax}{useE} at {RPMs[maxidx]:.0f} RPM, {Vinfs[maxidx]:.1f} m/s, {Ps[maxidx]:.0f} W')
        
    # # # location of maximum constrained efficiency (via masks)
    eta_adjust = etas.copy()
    eta_adjust[goodidx] = 0.0
    if eta_adjust.max() == 0.0:
        print('ERROR: Constraints infeasible for propellers at RPM of maximum efficiency!')
    else:
        con_maxidx = np.argmax(eta_adjust)
        if diameters[con_maxidx].is_integer():
            con_diammax = int(diameters[con_maxidx])
        else:
            con_diammax = diameters[con_maxidx]
        if pitches[con_maxidx].is_integer():
            con_pitchmax = int(pitches[con_maxidx])
        else:
            con_pitchmax = pitches[con_maxidx]
        print(f'Maximum Propeller Efficiency (constrained) is {eta_adjust.max()*100:.1f}% with the {con_diammax}x{con_pitchmax}{useE} at {RPMs[con_maxidx]:.0f} RPM and {Vinfs[con_maxidx]:.1f} m/s')
        plt.scatter(con_diammax, con_pitchmax, marker = 'x', color = 'blue', label = f'Constrained\n{con_diammax}x{con_pitchmax}{useE}\n{eta_adjust.max()*100:.1f}% $\\eta_p$\n{RPMs[con_maxidx]:.0f} RPM\n{Vinfs[con_maxidx]:.1f} m/s')

    plt.scatter(diammax, pitchmax, marker = '^', color = 'black', label = f'Maximum\n{diammax}x{pitchmax}{useE}\n{etas.max()*100:.1f}% $\\eta_p$\n{RPMs[maxidx]:.0f} RPM\n{Vinfs[maxidx]:.1f} m/s\n{Ps[maxidx]:.0f} W')

    plt.legend(fontsize = 7)
    plt.colorbar(img, label = r'$\eta_p$ (%)')
    plt.xlabel('Diameter (in)')
    plt.ylabel('Pitch (in)')
    plt.title('Optimal APC Propeller Efficiencies')
    plt.minorticks_on()
    plt.grid()
    plt.show()
    
#%%######################################################################
####################### USE FUNCTION HERE! ##############################
#########################################################################
# EfficiencyMap(30, 7800, grade = 15, Sw = 0.6, CD = 0.03, Plimit = 2700)