# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 13:25:49 2026

@author: marcu
"""

import sympy as sp

# Symbols
r, z = sp.symbols('r z', positive=True)
r0, lam = sp.symbols('r0 lam', positive=True)

# Define analytic flux function
Psi = (r**2 - r0**2)**2 + lam*z**2

# Grad-Shafranov operator
Psi_rr = sp.diff(Psi, r, 2)
Psi_r = sp.diff(Psi, r)
Psi_zz = sp.diff(Psi, z, 2)

Delta_star_Psi = Psi_rr - (1/r)*Psi_r + Psi_zz
Delta_star_Psi = sp.simplify(Delta_star_Psi)

Delta_star_Psi


# Expand RHS structure
Delta_star_Psi_expanded = sp.expand(Delta_star_Psi)

# Expand Psi too
Psi_expanded = sp.expand(Psi)

# Collect terms in r^2 and constants
sp.collect(Delta_star_Psi_expanded, [r**2, z**2])


import numpy as np

def psi_ground_truth(r, z, r0=1.0, lam=1.5):
    return (r**2 - r0**2)**2 + lam*z**2

def gs_operator_psi(r, z, r0=1.0, lam=1.5):
    Psi = psi_ground_truth(r, z, r0, lam)
    
    Psi_r = 4*r*(r**2 - r0**2)
    Psi_rr = 12*r**2 - 4*r0**2
    Psi_zz = 2*lam
    
    return Psi_rr - Psi_r/r + Psi_zz