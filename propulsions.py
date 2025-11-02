# -*- coding: utf-8 -*-
"""
Propulsions Model Restructuring (8/29/2025)
File structure:
    - Numerical methods for numba calculation (bisection, secant, brent, etc)
    - Preprocessing functions for propeller, motor, battery, ESC data
    - PointResult function
    - LinePlot functions (one for each input combo)
    - ContourPlot functions
    - CubicPlot functions
    - Pareto Front functions
    - Mission model compatible functions
    - GEKKO prop model functions for reference 

Primary inputs:
    Battery State Of Charge (SOC) as %
    Freestream Velocity (Vinf) as m/s
    Throttle Setting (dT) as %
    
LinePlot function:
    By fixing two of these inputs, a line plot of a propulsion quantity (propQ) can be plotted with respect to the third.

ContourPlot function:
    By fixing one of these inputs, a contour plot of a propQ is plotted wrt the two unfixed inputs.

CubicPlot function:
    Additionally a cubic plot of propQ variation with all three variables is available, but usually hard to use in reports.

for all of these functions, SOC can be provided as SOC (%), Voc (Volt), or t (s), which assumes constant current (i.e. good for aircraft in cruise)

Available propQs are:
    T (lbf)     (thrust for all motors)
    Q (N*m)     (torque for all motors)
    RPM         (for a single motor/propeler)

nondimensional:
    CT          (propeller (prop) torque constant)
    CP          (propeller power constant)
    eta_p       (propeller efficiency)
    eta_m       (motor efficiency)
    eta_c       (controller (ESC) efficiency)
    eta_drive   (drive efficiency)

all in Watts (W):
    Pout    (mechanical W for a single motor)
    Pin_m   (electric W input to a single motor (equivalent to Pout from a single ESC))
    Pin_c   (electric W input to all ESCs (equivalent to Pout from the battery))
    
all in Ampere (A):
    Im      (motor current for a single motor)
    Ic      (ESC current for a single ESC)
    Ib      (battery current)
    
all in Volts (V):
    Voc     (cell voltage)
    Vb      (battery voltage)
    Vm      (motor voltage)
    Vc      (ESC voltage)
    
    
Propeller coefficients (CT, CP) are primarily acquired by interpolation between the APC technical datasheets 
(see: https://www.apcprop.com/technical-information/performance-data/?v=7516fd43adaa)

These datasheets were generated using Blade Element Theory in conjunction with NASA TAIR and some databases for airfoil data. 
They are NOT fully accurate, especially for propeller stall, which mostly occurs when propellers low diam/pitch ratios travel at lower freestream velocities. 

Use the APCBEMTvsUIUCexpdata.py file to automatically compare APC electric propeller data from both sources!

While a wonderful amount of experimental data has been gathered by Michael Selig of UIUC's group (see: https://m-selig.ae.illinois.edu/props/propDB.html),
this data does not extend into the high performance ranges (say RPMs of 6k-12k for ~15-20 inch diameter propellers) that typically occur with 6-12S liPo batteries.

As of 8/29/2025, this experimental data is not implemented yet, but in the future it will be used in conjunction with the APC data for a mixed fidelity approach.



Model Formulation (primary based on Saemi 2023, secondary if available data based on Gong 2018):
    Simplified RPM (with constant ESC efficiency and constant I0)
    SAEMI 2023: https://www.mdpi.com/2226-4310/11/1/16
    GONG 2018:  https://www.researchgate.net/profile/Andrew-Gong-2/publication/326263042
    Jeong 2020: https://www.researchgate.net/publication/347270768_Improvement_of_Electric_Propulsion_System_Model_for_Performance_Analysis_of_Large-Size_Multicopter_UAVs
    
LiPo discharge curve (Voc(SOC)) from Chen 2006 (https://rincon-mora.gatech.edu/publicat/jrnls/tec05_batt_mdl.pdf):
    Voc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3
    
    Alternative from Jeong 2020
    Voc = 1.7*(SOC**3) - 2.1*(SOC**2) + 1.2*SOC + 3.4

In reality the discharge curve (and total capacity) is strongly influenced by cell temperature and battery health. 
Corrections on Voc for battery health are optionally defined by inputting Voc at maximum charge (determined experimentally) (NOT IMPLEMENTED AS OF 8/29/2025).
    
All of the following constants used in the propulsion model are aquired via the Motors, Batteries, and ESCs CSV sheets, which can all be adjusted by any user.
Motor constants:    KV (RPM/V), I0 (A), Rm (Ohm)
Battery constants:  CB (mAh), ns (number of cells in series), Rb (Ohm)

ESC constants (Saemi):      Rds (Ohm), fPWM (Hz), Tsd (s), Psb (W)   <---- currently defaults from Saemi 2023 are used
ESC constants (Gong):       a_m (the slope of constant a), a_0 (y intercept), b, c_m, c_0  <---- only available for a very limited selection of tested ESCs
ESC constnats (Jeong):       #TODO FILL IN JEONG CONSTANTS
    
KV = speed constant, I0 = no-load current, Rm = motor resistance
CB = battery capacity, ns = number of cells (in series), Rb = battery resistance

Simplifed RPM formulation (known Vsoc, Vinf, dT):
    
    RPM guess
    J = Vinf/((RPM/60)*d)
    CP = CPNumba(RPM, J, rpm_list, coef_numba_prop_data)
    Q = rho*((RPM/60)**2)*(d**5)*(CP/(2*np.pi))
    Im = Q*KV*(np.pi/30) + I0 # for one motor
    Ib = (nmot/eta_c)*Im
    Vb = ns*(Voc) - Ib*Rb
    Vm = dT*Vb
    RPMcalc = KV*(Vm - Im*Rm)
    res = RPMcalc - RPM

Simplified RPM formulation (known runtime (t), constant current):

    RPM guess
    J = Vinf/((RPM/60)*d)
    CP = CPNumba(RPM, J, rpm_list, coef_numba_prop_data)
    Q = rho*((RPM/60)**2)*(d**5)*(CP/(2*np.pi))
    Im = Q*KV*(np.pi/30) + I0 # for one motor
    Ib = (nmot/eta_c)*Im
    SOC = 1.0 - (Ib*t)/(CB*3.6)
    Voc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3
    Vb = ns*(Voc) - Ib*Rb
    Vm = dT*Vb
    RPMcalc = KV*(Vm - Im*Rm)
    res = RPMcalc - RPM
    
minimize res!

This formulation is advantageous for its computational efficiency. With only one variable, plots can be made extremely quickly.
Significant loss of accuracy due to the constant I0 assumption compared to the other models.

    
@author: NASSAS
"""

import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d

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
        datasection = np.array([PROP_DATA[RPM]['J'], 
                                PROP_DATA[RPM]['CT'], 
                                PROP_DATA[RPM]['CP']])
        numba_prop_data.append(datasection)
        
    return(PROP_DATA, numba_prop_data)

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