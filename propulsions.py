# -*- coding: utf-8 -*-
"""
Propeller coefficients (CT, CP) are primarily acquired by interpolation between the APC technical datasheets 
(see: https://www.apcprop.com/technical-information/performance-data/?v=7516fd43adaa)

These datasheets were generated using Blade Element Theory in conjunction with NASA TAIR and some databases for airfoil data. 
They are NOT fully accurate, especially for propeller stall, which mostly occurs when propellers low diam/pitch ratios travel at lower freestream velocities. 

    
@author: NASSAS
"""

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib import patheffects
from gekko import GEKKO
from numba import njit

lbfN = 4.44822
ftm = 0.3048
MPH_TO_MPS = 0.44704  # Conversion factor: 1 mph to m/s

global propQnames
propQnames = ['Total Thrust (lbf)', 'Total Torque (Nm)', 'RPM', 'Drive Efficiency', 'Propeller Efficiency', 'Motor Efficiency', 'ESC Efficiency', 'Mech. Power Out of 1 Motor (W)', 
                   'Elec. Power Into 1 Motor (W)', 'Elec. Power Into 1 ESC (W)', 'Current in 1 Motor (A)', 'Current in 1 ESC (A)', 'Current in Battery (A)',
                   'Voltage in 1 Motor (V)', 'Voltage in 1 ESC (V)', 'Battery Voltage (V)', 'Voltage Per Cell (V)', 'State of Charge (%)']

#%% ################### Data Parsing ###################
def parse_coef_propeller_data(prop_name):
    """
    prop_name in the form: 16x10E, 18x12E, 12x12, etc (no PER3_ and no .dat to make it easier for new users)
    Parses the provided PER3_16x10E.dat content to extract RPM, V (m/s), Thrust (N), Torque (N-m).
    Stores in PROP_DATA as {rpm: {'V': np.array, 'Thrust': np.array, 'Torque': np.array}}
    """    
    PROP_DATA = {}

    with open(f'PropDatabase/PER3_{prop_name}.dat', 'r') as f:
        data_content = f.read()

    current_rpm = None
    in_table = False
    table_lines = []
    
    for line in data_content.splitlines():
        line = line.strip()
        if line.startswith("PROP RPM ="):
            # Extract RPM
            current_rpm = int(line.split("=")[-1].strip())
            in_table = False
            table_lines = []
        elif line.startswith("V") and "J" in line and current_rpm is not None:
            # Start of table headers
            in_table = True
        elif in_table and line and not line.startswith("(") and len(line.split()) >= 10:
            # Parse data rows (ensure it's a data line with enough columns)
            parts = line.split()
            try:
                J = float(parts[1])  # advance ratio J
                CT = float(parts[3])  # thrust coef
                CP = float(parts[4])  # power coef (can convert to CQ)
                # v_mps = v_mph * MPH_TO_MPS  # Convert to m/s
                table_lines.append((J, CT, CP))
            except (ValueError, IndexError):
                continue  # Skip malformed lines
        elif in_table and (line == "" or "PROP RPM" in line):
            # End of table for this RPM, store if data exists
            if current_rpm and table_lines:
                J_list, CT_list, CP_list = zip(*sorted(table_lines))  # Sort by V for interp1d
                PROP_DATA[current_rpm] = {
                    'J': np.array(J_list),
                    'CT': np.array(CT_list),
                    'CP': np.array(CP_list)
                }
            in_table = False
    
    # Sort RPM keys for efficient lookup
    PROP_DATA['rpm_list'] = sorted(PROP_DATA.keys())
    
    # array based datastructure where each index corresponds to rpm_values[i] (or i+1*1000 RPM)
    # and in each index there is [[V values], [Thrust values], [Torque values]] at the indices, 0, 1, 2
    numba_prop_data = []
    for RPM in PROP_DATA['rpm_list']:
        # KEY new adjustment that adds empty values to make sure the data is evenly spaced!
        while PROP_DATA[RPM]['J'].size != 30: # every column should be 30!
            PROP_DATA[RPM]['J'] = np.append(PROP_DATA[RPM]['J'], PROP_DATA[RPM]['J'][-1])
            PROP_DATA[RPM]['CT'] = np.append(PROP_DATA[RPM]['CT'], PROP_DATA[RPM]['CT'][-1])
            PROP_DATA[RPM]['CP'] = np.append(PROP_DATA[RPM]['CP'], PROP_DATA[RPM]['CP'][-1])

        datasection = np.array([PROP_DATA[RPM]['J'], 
                                PROP_DATA[RPM]['CT'], 
                                PROP_DATA[RPM]['CP']])
        numba_prop_data.append(datasection)
        
    numba_prop_data = np.stack(numba_prop_data)
    return(PROP_DATA, numba_prop_data)

#%% Functions for CP and CT interpolation based on set RPM and J
def CPBase(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio

    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    CPs = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        # NEW CODE WAS NEEDED TO BRING IT TO 0 WHEN J WAS OUTSIDE BOUNDS!!
        if J > data[0].max():
            CPs.append(0.0)
            continue
        CPs.append(np.interp(J, data[0], data[2]))
        
    CPs = np.array(CPs)
    
    if len(closest_rpms) == 1:
        return CPs[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * CPs[0] + weight * CPs[1]

def CTBase(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio
    
    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
    
    CTs = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 - 1)]
        
        # NEW CODE WAS NEEDED TO BRING IT TO 0 WHEN J WAS OUTSIDE BOUNDS!!
        if J > data[0].max():
            CTs.append(0.0)
            continue
        
        CTs.append(np.interp(J, data[0], data[1]))
    CTs = np.array(CTs)
    
    if len(closest_rpms) == 1:
        return CTs[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * CTs[0] + weight * CTs[1]
    
#%% Numba Interpolation Functions for J, CT, CP data
@njit(fastmath = True)
def CPNumba(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio

    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    CPs = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        # NEW CODE WAS NEEDED TO BRING IT TO 0 WHEN J WAS OUTSIDE BOUNDS!!
        if J > data[0].max():
            CPs.append(0.0)
            continue
        CPs.append(np.interp(J, data[0], data[2]))
        
    CPs = np.array(CPs)
    
    if len(closest_rpms) == 1:
        return CPs[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * CPs[0] + weight * CPs[1]

@njit(fastmath = True)
def CTNumba(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio
    
    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    CTs = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        if J > data[0].max():
            CTs.append(0.0)
            continue
        CTs.append(np.interp(J, data[0], data[1]))
        
    CTs = np.array(CTs)
    
    if len(closest_rpms) == 1:
        return CTs[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * CTs[0] + weight * CTs[1]
    
@njit(fastmath = True)
def etaNumba(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio
    
    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    etas = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        if J > data[0].max():
            etas.append(0.0)
            continue
        etas.append(np.interp(J, data[0], data[3]))
        
    etas = np.array(etas)
    
    if len(closest_rpms) == 1:
        return etas[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * etas[0] + weight * etas[1]




#%% Functions for mild optimization methods
def OptRPMeta(Vinf, rpm_values, numba_prop_data, D, ns = 150):
    '''
    Vinf (m/s)
    D (m)
    
    ns = 150 provides an accuracy of around 4 digits for eta, RPM
    
    determines the RPM of max efficiency for the selected prop at specified RPM
    '''
    allRPMs = np.repeat(rpm_values, 30)
    allJs = numba_prop_data[:, 0, :].flatten()
    allCTs = numba_prop_data[:, 1, :].flatten()
    allCPs = numba_prop_data[:, 2, :].flatten()
    alletas = (allCTs*allJs)/allCPs

    RPM_g = np.linspace(allRPMs.min(), allRPMs.max(), ns)
    J_g = np.linspace(allJs.min(), allJs.max(), ns)#numbapropdata[:, 0, :]
    Jg, RPMg = np.meshgrid(J_g, RPM_g)
    eta_grid = griddata((allJs, allRPMs), alletas, (Jg, RPMg), method='linear')
    J_line = Vinf/((RPM_g/60)*D)
    eta_line = griddata((Jg.flatten(), RPMg.flatten()), eta_grid.flatten(), (J_line, RPM_g), method='linear')
    valid = ~np.isnan(eta_line)
    if eta_line[valid].size == 0:
        return(0.0, 1000)
    max_idx = np.argmax(eta_line[valid])
    max_eta = eta_line[valid][max_idx]
    max_rpm = RPM_g[valid][max_idx]
    return(max_eta, max_rpm)

#%% For RPM such that T meets the requirement for cruise
def ThrustRPMeta(Vinf, Treq, rho, rpm_values, numba_prop_data, D, ns = 150):
    '''
    Vinf (m/s)
    D (m)
    
    ns = 150 provides an accuracy of around 4 digits for eta, RPM
    
    determines the RPM to meet T = D
    
    # todo: replace with numbafied functions that work much faster
    '''
    allRPMs = np.repeat(rpm_values, 30)
    allJs = numba_prop_data[:, 0, :].flatten()
    allCTs = numba_prop_data[:, 1, :].flatten()
    allCPs = numba_prop_data[:, 2, :].flatten()
    alletas = (allCTs*allJs)/allCPs

    T = rho*((allRPMs/60)**2)*(D**4)*allCTs # thrust in N
    RPM_g = np.linspace(allRPMs.min(), allRPMs.max(), ns)
    J_g = np.linspace(allJs.min(), allJs.max(), ns)#numbapropdata[:, 0, :]
    Jg, RPMg = np.meshgrid(J_g, RPM_g)
    T_grid = griddata((allJs, allRPMs), T, (Jg, RPMg), method='linear')
    J_line = Vinf/((RPM_g/60)*D)
    T_line = griddata((Jg.flatten(), RPMg.flatten()), T_grid.flatten(), (J_line, RPM_g), method='linear')
    valid = ~np.isnan(T_line)
    if T_line[valid].size == 0:
        return(0.0, 1000)
    
    eta_grid = griddata((allJs, allRPMs), alletas, (Jg, RPMg), method='linear')
    eta_line = griddata((Jg.flatten(), RPMg.flatten()), eta_grid.flatten(), (J_line, RPM_g), method='linear')
    if eta_line[valid].size == 0:
        return(0.0, 1000)
    
    TDidx = np.argmin(np.abs(T_line[valid] - Treq))
    if np.min(np.abs(T_line[valid] - Treq)) >= 1: # check that it can actually meet the thrust req
        raise ValueError('Thrust not matched')
        
    eta_TD = eta_line[valid][TDidx]
    RPM_TD = RPM_g[valid][TDidx]
    
    
    return(eta_TD, RPM_TD)

#%% Full eta optimization
# goal: for a given propeller, find the RPM and velocity for maximum efficiency
def OptEta(rpm_values, numba_prop_data, D, ns = 150):
    '''
    D (m)
    
    ns = 150 provides an accuracy of around 4 digits for eta, RPM
    '''
    allRPMs = np.repeat(rpm_values, 30)
    allJs = numba_prop_data[:, 0, :].flatten()
    allCTs = numba_prop_data[:, 1, :].flatten()
    allCPs = numba_prop_data[:, 2, :].flatten()
    alletas = (allCTs*allJs)/allCPs
    maxloc = np.argmax(alletas)
    
    optEta = alletas.max()
    optRPM = allRPMs[maxloc]
    optVinf = allJs[maxloc]*((optRPM/60)*D)
    
    return(optRPM, optVinf, optEta)


#%% Eta optimization under constraints
@njit(fastmath = True)
def innerdatafunc(RPMs, Js, rpm_values, numba_prop_data, n):
    CT = np.zeros((n, n))
    CP = np.zeros((n, n))
    # eta = np.zeros((n, n))
    for i, RPM in enumerate(RPMs):
        for j, J in enumerate(Js):
            CT[i, j] = CTNumba(RPM, J, rpm_values, numba_prop_data)
            CP[i, j] = CPNumba(RPM, J, rpm_values, numba_prop_data)
            # eta[i, j] = etaNumba(RPM, J, rpm_values, numba_prop_data)
    return(CT, CP)

def ConsOptEta(Sw, CD, Pmax, rho, d, rpm_values, numba_prop_data):
    # data from interpolation functions
    allRPMs = np.repeat(rpm_values, 30)
    allJs = numba_prop_data[:, 0, :].flatten()
    allCTs = numba_prop_data[:, 1, :].flatten()
    allCPs = numba_prop_data[:, 2, :].flatten()
    # alletas = (allCTs*allJs)/allCPs
    
    n = 50
    RPMs = np.linspace(1000, rpm_values[-1], n)
    Js = np.linspace(0, allJs.max(), n)
    CTs, CPs = innerdatafunc(RPMs, Js, rpm_values, numba_prop_data, n) # squaring the data (find a better way later)
    etas = np.zeros((n,n)) 
    mask = CPs != 0.0
    etas[mask] = (CTs*Js)[mask]/CPs[mask]
    
    m = GEKKO(remote = False)
    J = m.Var((allJs.min() + allJs.max())/2, allJs.min(), allJs.max())
    RPM = m.Var((allRPMs.min() + allRPMs.max())/2, allRPMs.min(), allRPMs.max())
    CT = m.Var(0.01) 
    CP = m.Var(0.05)
    eta = m.Var(0.6)
    
    degs = 5
    m.bspline(RPM, J, CT, RPMs, Js, CTs, data = True, kx = degs, ky = degs)
    m.bspline(RPM, J, CP, RPMs, Js, CPs, data = True, kx = degs, ky = degs)
    m.bspline(RPM, J, eta, RPMs, Js, etas, data = True, kx = degs, ky = degs)

    m.Equation([CT == (0.5*Sw*CD*(1/(d**2))*(J**2)),
                CP <= Pmax/(rho*((RPM/60)**3)*(d**5)), 
                1000 < RPM, 
                RPM < rpm_values[-1], 
                0 < J, 
                J < Js[-1]-0.01])
    m.Maximize(eta)
    m.solve(disp = False, solver = 3)   
    
    max_rpm = RPM.value[0]
    max_J = J.value[0]
    max_vinf = max_J*(max_rpm/60)*d 
    max_eta = eta.value[0]

    # CTreq = 0.5*Sw*CD*(1/d**2)*(allJs**2)
    # CTdiff = allCTs - CTreq

    # # CTdiff_interp = CloughTocher2DInterpolator((allJs, allRPMs), CTdiff)
    # eta_interp = CloughTocher2DInterpolator((allJs, allRPMs), alletas)

    # n = 250
    # Js = np.linspace(allJs.min(), allJs.max(), n)
    # RPMs = np.linspace(allRPMs.min(), allRPMs.max(), n)
    # Jg, RPMg = np.meshgrid(Js, RPMs)

    # # CTdiff_g = CTdiff_interp(Jg, RPMg)

    # fig, ax = plt.subplots()
    # cs = ax.tricontour(allJs, allRPMs, CTdiff, levels=[0.0])
    # plt.close(fig)

    # # Pick the longest path (where CT = CTreq so T = D)
    # paths = cs.get_paths()
    # path = max(paths, key=lambda p: p.vertices.shape[0])
    # J_line, RPM_line = path.vertices.T 
    
    # #interp
    # eta_line = eta_interp(J_line, RPM_line)
    # Vinf_line = J_line*(RPM_line/60.0)*d

    # mask = ~np.isnan(eta_line)
    # eta_line = eta_line[mask]
    # Vinf_line = Vinf_line[mask]
    # RPM_line = RPM_line[mask]
    # sort_idx = np.argsort(Vinf_line)
    # Vinf_line = Vinf_line[sort_idx]
    # eta_line = eta_line[sort_idx]
    # RPM_line = RPM_line[sort_idx]
    # imax = np.argmax(eta_line)
    # max_eta = eta_line[imax]
    # max_vinf = Vinf_line[imax]
    # max_rpm = RPM_line[imax]
    
    return(max_rpm, max_vinf, max_eta)

    
#%% EXTRAS FROM WORKING OUT THE MAGIC
# import matplotlib.pyplot as plt
# PROP_DATA, numbapropdata = parse_coef_propeller_data('16x10E')
# rpm_values = np.array(PROP_DATA['rpm_list'])

# inm = 0.0254

# # holy shit this is so much more efficient. And now numba prop data IS actually a full array!! 
# # Meaning you should be able to numbafy the whole thing!!! enormous implications
# allRPMs = np.repeat(rpm_values, 30)
# allJs = numbapropdata[:, 0, :].flatten()
# allCTs = numbapropdata[:, 1, :].flatten()
# allCPs = numbapropdata[:, 2, :].flatten()
# alletas = (allCTs*allJs)/allCPs


# # Vinf = 30.0 #m/s
# # D = 0.5*(Vinf**2)*0.8*0.03
# rho = 1.19
# d = 16*inm
# Sw = 0.6
# CD = 0.05 
# Pmax = 1100 #W
# CPupperbound = Pmax/(rho*((allRPMs/60)**3)*(d**5))
# CPlimits = allCPs-CPupperbound

# # eta plot
# fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
# # fill = ax.scatter(allJs, allRPMs, c=allCPs, marker = 's', s = 60)
# fill = ax.tricontourf(allJs, allRPMs, alletas, levels = 18) #levels = np.linspace(1e-10, allCTs.max(), 15))
# fig.colorbar(fill, label = 'eta')
# CTreq = 0.5*Sw*CD*(1/(d**2))*(allJs**2)
# line = ax.tricontour(allJs, allRPMs, allCTs-CTreq, colors = 'k', levels = [0.0])
# line.clabel(fmt = 'T = D')

# # J = Vinf/((allRPMs/60)*(d))
# allVinfs = allJs*(allRPMs/60)*d 
# lines = ax.tricontour(allJs, allRPMs, allVinfs, levels = 5, cmap = 'inferno')
# lines.clabel(fmt = '%.1f m/s')

# line = ax.tricontour(allJs, allRPMs, CPlimits, colors = 'k', levels = [0.0])
# line.clabel(fmt = f'{Pmax} W')
# line.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])

# # lines = ax.tricontour(allJs, allRPMs, 0.5*(allVinfs**2)*Sw*CD, levels = 5, cmap = 'bwr')
# # lines.clabel(fmt = '%.1f N')

# # plt.plot(J[J < allJs.max()], allRPMs[J < allJs.max()]) # velocity line
# plt.xlabel('J')
# plt.ylabel('RPM')
# plt.yticks(rpm_values)
# plt.title(f'Raw Coefficient Data for 16x10E')
# plt.show()

# # CT plot
# fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
# # fill = ax.scatter(allJs, allRPMs, c=allCPs, marker = 's', s = 60)
# fill = ax.tricontourf(allJs, allRPMs, allCTs-CTreq, levels = 30) #levels = np.linspace(1e-10, allCTs.max(), 15))
# fig.colorbar(fill, label = 'CT prop - CT req for T = D')
# # lines = ax.tricontourf(allJs, allRPMs, , levels = np.linspace(CTreq.min(), CTreq.max(), 50))
# # CTreq = D/(rho*((allRPMs/60)**2)*((16*inm)**4))
# # CTreq = np.flip(CTreq)
# line = ax.tricontour(allJs, allRPMs, allCTs-CTreq, colors = 'k', levels = [0.0])
# line.clabel()

# line = ax.tricontour(allJs, allRPMs, CPlimits, colors = 'k', levels = [0.0])
# line.clabel(fmt = f'{Pmax} W')
# line.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])

# plt.xlabel('J')
# plt.ylabel('RPM')
# plt.yticks(rpm_values)
# plt.title(f'Raw Coefficient Data for 16x10E')
# plt.show()

# # CP plot with Pmax limit
# Pmax = 1100 #W
# CPupperbound = Pmax/(rho*((allRPMs/60)**3)*(d**5))

# fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
# CPlimits = allCPs-CPupperbound
# fill = ax.tricontourf(allJs, allRPMs, CPlimits, levels = 15) #levels = np.linspace(1e-10, allCTs.max(), 15))
# fig.colorbar(fill, label = 'CP')
# line = ax.tricontour(allJs, allRPMs, allCTs-CTreq, colors = 'k', levels = [0.0])
# line.clabel()

# line = ax.tricontour(allJs, allRPMs, CPlimits, colors = 'k', levels = [0.0])
# line.clabel(fmt = f'{Pmax} W')
# line.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
# plt.xlabel('J')
# plt.ylabel('RPM')
# plt.yticks(rpm_values)
# plt.title(f'Raw Coefficient Data for 16x10E')
# plt.show()

# Kinda despise that AI can return code like this that works. It's definitely not optimized, but it does work...

#%% it just works
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CloughTocher2DInterpolator

# start = time.perf_counter()

# # --- compute CT difference (T = D condition) ---
# CTreq = 0.5 * Sw * CD * (1 / d**2) * (allJs**2)
# CTdiff = allCTs - CTreq

# # where CPlimits > 0 is acceptable
# Pmax = 1100 #W
# CPupperbound = Pmax/(rho*((allRPMs/60)**3)*(d**5))
# CPlimits = allCPs-CPupperbound

# # --- Build smooth cubic interpolators (no Triangulation object needed) ---
# # CTdiff_interp = CloughTocher2DInterpolator((allJs, allRPMs), CTdiff)
# eta_interp = CloughTocher2DInterpolator((allJs, allRPMs), alletas)

# # --- Generate a moderately dense grid for contour extraction ---
# n = 250
# Js = np.linspace(allJs.min(), allJs.max(), n)
# RPMs = np.linspace(allRPMs.min(), allRPMs.max(), n)
# Jg, RPMg = np.meshgrid(Js, RPMs)

# # CTdiff_g = CTdiff_interp(Jg, RPMg)

# # --- Extract T = D line (CTdiff = 0 contour) ---
# fig, ax = plt.subplots()
# cs = ax.tricontour(allJs, allRPMs, CTdiff, levels=[0.0])
# plt.close(fig)

# # Pick the longest path
# paths = cs.get_paths()
# path = max(paths, key=lambda p: p.vertices.shape[0])
# J_line, RPM_line = path.vertices.T

# # --- Interpolate η and compute velocity ---
# eta_line = eta_interp(J_line, RPM_line)
# Vinf_line = J_line*(RPM_line/60.0)*d

# # --- Clean up and sort ---
# mask = ~np.isnan(eta_line)
# eta_line = eta_line[mask]
# Vinf_line = Vinf_line[mask]
# RPM_line = RPM_line[mask]

# sort_idx = np.argsort(Vinf_line)
# Vinf_line = Vinf_line[sort_idx]
# eta_line = eta_line[sort_idx]
# RPM_line = RPM_line[sort_idx]

# # --- Find max efficiency point ---
# imax = np.argmax(eta_line)
# max_eta = eta_line[imax]
# max_vinf = Vinf_line[imax]
# max_rpm = RPM_line[imax]

# print(f"Max η ≈ {max_eta:.4f} at V∞ ≈ {max_vinf:.2f} m/s, RPM ≈ {max_rpm:.1f}")

# # --- Plot ---
# plt.figure(figsize=(6,4))
# plt.plot(Vinf_line, eta_line, 'k-', label='η along T=D')
# plt.scatter(max_vinf, max_eta, color='red', label='Max η')
# plt.xlabel('V∞ [m/s]')
# plt.ylabel('η')
# plt.title('Propeller Efficiency along T = D')
# plt.legend()
# plt.grid(True)
# plt.show()

# end = time.perf_counter()
# print(f'{end-start:.5f} s')

# fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
# # fill = ax.scatter(allJs, allRPMs, c=allCPs, marker = 's', s = 60)
# fill = ax.tricontourf(allJs, allRPMs, allCPs) #levels = np.linspace(1e-10, allCTs.max(), 15))
# fig.colorbar(fill, label = 'CP')
# # line = ax.tricontour(allJs, allRPMs, allCTs, levels = [CTreq], colors = 'k')
# # line.clabel()
# J = Vinf/((allRPMs/60)*(16*inm))
# plt.plot(J[J < allJs.max()], allRPMs[J < allJs.max()])
# plt.xlabel('J')
# plt.ylabel('RPM')
# plt.yticks(rpm_values)
# plt.title(f'Raw Coefficient Data for 16x10E')
# plt.show()


# # n = RPM/60
# # J = V/nD
# # eta = (CT(RPM, J)*J)/CP(RPM, J)
# # want to optimize eta for specified J

# # now that we have alletas and allJs this gets a LOT easier
# # for specified Vinf, we can determine the closest Js for interpolation

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata, bisplrep

# # Suppose you already have:
# # allRPMs, allJs, alletas (same as your code)
# # and you want max eta for a given Vinf:
# D = 16 * 0.0254  # convert inches to meters
# Vinf = 30.0

# ns = 150 # ~4 digits of accuracy!
# RPM_g = np.linspace(allRPMs.min(), allRPMs.max(), ns)
# J_g = np.linspace(allJs.min(), allJs.max(), ns)#numbapropdata[:, 0, :]
# Jg, RPMg = np.meshgrid(J_g, RPM_g)
# eta_grid = griddata((allJs, allRPMs), alletas, (Jg, RPMg), method='linear')
# J_line = Vinf/((RPM_g/60)*D)
# eta_line = griddata((Jg.flatten(), RPMg.flatten()), eta_grid.flatten(), (J_line, RPM_g), method='linear')
# valid = ~np.isnan(eta_line)
# max_idx = np.argmax(eta_line[valid])
# max_eta = eta_line[valid][max_idx]
# max_rpm = RPM_g[valid][max_idx]

# print(f"Maximum η ≈ {max_eta:.6f} at {max_rpm:.0f} RPM")

# # Optional: plot the efficiency curve vs RPM
# plt.figure(figsize=(6,4))
# plt.plot(RPM_g, eta_line, label=f'V∞ = {Vinf} m/s')
# plt.scatter(max_rpm, max_eta, color='red', zorder=5, label='Max η')
# plt.xlabel('RPM')
# plt.ylabel('η')
# plt.title('Interpolated Efficiency vs RPM')

# ok now lets suppose you have a P and T req
# rho = 1.19
# say T = 15 N
# say P = 3000 W
# Plim = 3000
# Tlim = 30

# P = rho*((allRPMs/60)**3)*(D**5)*allCPs
# T = rho*((allRPMs/60)**2)*(D**4)*allCTs
# T_grid = griddata((allJs, allRPMs), T, (Jg, RPMg), method='linear')
# T_line = griddata((Jg.flatten(), RPMg.flatten()), T_grid.flatten(), (J_line, RPM_g), method='linear')
# Tlimloc = np.argmin(np.abs(T_line[~np.isnan(T_line)] - Tlim))
# Tmax = T_line[~np.isnan(T_line)].max()
# plt.plot(RPM_g, T_line/Tmax, label='norm T')
# plt.scatter(RPM_g[~np.isnan(T_line)][Tlimloc], T_line[~np.isnan(T_line)][Tlimloc]/Tmax, label = 'Tlimit')
# P_grid = griddata((allJs, allRPMs), P, (Jg, RPMg), method='linear')
# P_line = griddata((Jg.flatten(), RPMg.flatten()), P_grid.flatten(), (J_line, RPM_g), method='linear')
# Pmax = P_line[~np.isnan(P_line)].max()
# plt.plot(RPM_g, P_line/Pmax, label='norm P')
# Plimloc = np.argmin(np.abs(P_line[~np.isnan(P_line)] - Plim))
# plt.scatter(RPM_g[~np.isnan(P_line)][Plimloc], P_line[~np.isnan(P_line)][Plimloc]/Pmax, label = 'Plimit')

# plt.legend()
# plt.show()

# #%% try2 with bsplines (does not work)
# from scipy.interpolate import make_interp_spline, RBFInterpolator, bisplev, SmoothBivariateSpline, RegularGridInterpolator

# #### the following code works but the fit is too inaccurate near the peak. griddata was just better all along
# # RPMspec = np.linspace(allRPMs.min(), allRPMs.max(), 1000)
# # J_spec = Vinf/((RPMspec/60)*D)
# # mask = J_spec <= allJs.max()
# # RPMspec = RPMspec[mask]
# # J_spec = J_spec[mask]

# # plt.figure(figsize=(6,4))
# # plt.plot(RPMspec, SmoothBivariateSpline(allJs, allRPMs, alletas).ev(J_spec, RPMspec), label=f'V∞ = {Vinf} m/s')
# # # plt.scatter(max_rpm, max_eta, color='red', zorder=5, label='Max η')
# # plt.xlabel('RPM')
# # plt.ylabel('η')
# # plt.ylim([0.45, 0.85])
# # plt.xlim([6000, 16000])

# # plt.title('Interpolated Efficiency vs RPM')
# # plt.legend()
