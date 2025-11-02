# -*- coding: utf-8 -*-
"""
Propeller coefficients (CT, CP) are primarily acquired by interpolation between the APC technical datasheets 
(see: https://www.apcprop.com/technical-information/performance-data/?v=7516fd43adaa)

These datasheets were generated using Blade Element Theory in conjunction with NASA TAIR and some databases for airfoil data. 
They are NOT fully accurate, especially for propeller stall, which mostly occurs when propellers low diam/pitch ratios travel at lower freestream velocities. 

    
@author: NASSAS
"""

import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d, griddata

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
    

#%% Functions for mild optimization methods
def OptEta(Vinf, rpm_values, numba_prop_data, D, ns = 150):
    '''
    Vinf (m/s)
    D (m)
    
    ns = 150 provides an accuracy of around 4 digits for eta, RPM
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



# fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
# # fill = ax.scatter(allJs, allRPMs, c=allCPs, marker = 's', s = 60)
# fill = ax.scatter(allJs, allRPMs, c=alletas) #levels = np.linspace(1e-10, allCTs.max(), 15))
# fig.colorbar(fill, label = 'eta')

# Vinf = 40.0 #m/s
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

# # Step 1: Create a fine grid of (J, RPM)
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
