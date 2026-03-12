# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 16:04:40 2025

@author: marcu
"""


import numpy as np

import jax
import jax.numpy as jnp


# print(type(data))
# print(data.shape)
# print(data.dtype)

# print(data)


from scipy.interpolate import griddata

eq = np.load('C:/Users/marcu/Downloads/b06.npy', allow_pickle=True).item() #change according to directory


R_pts = eq["R"].ravel()
Z_pts = eq["Z"].ravel()
psi_pts = eq["psi"].ravel()


R_lin = np.linspace(R_pts.min(), R_pts.max(), 300)
Z_lin = np.linspace(Z_pts.min(), Z_pts.max(), 300)
RR, ZZ = np.meshgrid(R_lin, Z_lin)

# Example scalar field (placeholder)
values = np.ones_like(R_pts)  

psi_RZ = griddata(
    points=(R_pts, Z_pts),
    values=values,
    xi=(RR, ZZ),
    method="cubic"
)


#Sample plot for placeholder field 
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.contour(RR, ZZ, psi_RZ, levels=30)
plt.xlabel("R [m]")
plt.ylabel("Z [m]")
plt.title("Interpolated ψ(R,Z)")
plt.axis("equal")
plt.colorbar()
plt.show()


def generate_data_helena(epsilon, kappa, delta, P, n_radial, n_angular, R_eq, Z_eq, psi_eq):
    """
    Generate training data (coordinates, labels, and source term) based on helena equilibrium data 
    
    Parameters:
        epsilon, kappa: Geometric parameters.
        delta, P: Task-specific parameters.
        n_radial, n_angular: Number of points in the radial and angular directions.
        
    Returns:
        data_train: Cartesian coordinates (R, Z).
        labels: Analytic solution evaluated at data_train.
        g: Source term for the PDE.
        i_bc: Indicator array for boundary condition points.
    """
    
    # Create a meshgrid in polar coordinates
    radial = jnp.linspace(0, epsilon, n_radial)
    angular = jnp.linspace(0, 2 * jnp.pi, n_angular)
    radial_mesh, angular_mesh = jnp.meshgrid(radial, angular)
    
    # Convert polar to Cartesian coordinates
    R_train = 1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta) * jnp.sin(angular_mesh))
    Z_train = radial_mesh * kappa * jnp.sin(angular_mesh)
    data_train = jnp.column_stack([R_train.ravel(), Z_train.ravel()])
    
    # --- Interpolate equilibrium ψ ---
    from scipy.interpolate import griddata
    import numpy as np
    
    R_pinn = np.array(data_train[:, 0])
    Z_pinn = np.array(data_train[:, 1])
    
    psi_interp = griddata((R_eq, Z_eq), psi_eq, (R_pinn, Z_pinn), method="cubic")
    
    valid = ~np.isnan(psi_interp)

    data_train = data_train[valid]
    labels = jnp.array(psi_interp[valid]).reshape(-1, 1)

    g = P * data_train[:, 0:1]**2 + (1 - P)
    i_bc = jnp.where(radial_mesh.ravel()[valid] == epsilon, 1, 0)

    return data_train, labels, g, i_bc, R_pinn, Z_pinn, psi_interp


# selecting geometric values, values given are samples 
epsilon = 0.32          # Elongation parameter
n_radial = 10     # Number of radial grid points, 50
n_angular = 360   # Number of angular grid points, 720
delta = 0.7
P = 0.8
kappa = 1.7

data_train, labels, g, i_bc, R_pinn, Z_pinn, psi_interp = generate_data_helena(epsilon, kappa, delta, P, n_radial, n_angular, R_pts, Z_pts, psi_pts)

R_pts = eq["R"].ravel()
Z_pts = eq["Z"].ravel()
psi_pts = eq["psi"].ravel()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: Original equilibrium grid ---
sc0 = axes[0].scatter(
    R_pts, Z_pts,
    c=psi_pts,
    s=5,
)
axes[0].set_title("Original equilibrium grid (ψ)")
axes[0].set_xlabel("R")
axes[0].set_ylabel("Z")
axes[0].set_aspect("equal")
plt.colorbar(sc0, ax=axes[0])

# --- Plot 2: Interpolated ψ on PINN grid ---
sc1 = axes[1].scatter(
    R_pinn, Z_pinn,
    c=psi_interp,
    s=5,
)
axes[1].set_title("Interpolated ψ on PINN grid")
axes[1].set_xlabel("R")
axes[1].set_ylabel("Z")
axes[1].set_aspect("equal")
plt.colorbar(sc1, ax=axes[1])

plt.tight_layout()
plt.show()