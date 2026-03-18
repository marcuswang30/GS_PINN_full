# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 23:01:38 2026

@author: marcu
"""

"""
Utility functions for GS-PINN.

Includes functions to calculate system matrices, generate training data, 
compute the analytic solution, and plot results.
"""

import jax
import jax.numpy as jnp
import flax.serialization
from jax import jacfwd, vmap
import numpy as np
import matplotlib.pyplot as plt
import optax
import config
from flax import linen as nn
import os
from scipy.interpolate import griddata

import configv2 as config

# def calc_A(epsilon, delta, kappa, P):
#     """
#     Calculate the 7x7 matrix A (coefficients for the Grad–Shafranov equation).
    
#     Returns:
#         A: jnp.ndarray of shape (7, 7)
#     """
#     # Coefficients for the first row
#     c11 = 1
#     c12 = (epsilon + 1)**2
#     c13 = -jnp.log(epsilon + 1) * (epsilon + 1)**2
#     c14 = (epsilon + 1)**4
#     c15 = 3 * jnp.log(epsilon + 1) * (epsilon + 1)**4
#     c16 = (epsilon + 1)**6
#     c17 = -15 * jnp.log(epsilon + 1) * (epsilon + 1)**6

#     # Coefficients for the second row
#     c21 = 1
#     c22 = (epsilon - 1)**2
#     c23 = -jnp.log(1 - epsilon) * (epsilon - 1)**2
#     c24 = (epsilon - 1)**4
#     c25 = 3 * jnp.log(1 - epsilon) * (epsilon - 1)**4
#     c26 = (epsilon - 1)**6
#     c27 = -15 * jnp.log(1 - epsilon) * (epsilon - 1)**6

#     # Coefficients for the third row (example, similar structure follows for rows 4-7)
#     c31 = 1
#     c32 = (delta * epsilon - 1)**2
#     c33 = -(jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2 - epsilon**2 * kappa**2)
#     c34 = ((delta * epsilon - 1)**4 - 4 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**2)
#     c35 = (3 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**4 + 2 * epsilon**4 * kappa**4
#            - 9 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**2 - 12 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2)
#     c36 = ((delta * epsilon - 1)**6 - 12 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**4 + 8 * epsilon**4 * kappa**4 * (delta * epsilon - 1)**2)
#     c37 = -(15 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**6 - 8 * epsilon**6 * kappa**6 - 75 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**4
#            + 140 * epsilon**4 * kappa**4 * (delta * epsilon - 1)**2 - 180 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**4
#            + 120 * epsilon**4 * kappa**4 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2)
    
#     c41 = 0
#     c42 = -2 * (delta * epsilon - 1)
#     c43 = (delta * epsilon + 2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1) - 1)
#     c44 = -(4 * (delta * epsilon - 1)**3 - 8 * epsilon**2 * kappa**2 * (delta * epsilon - 1))
#     c45 = -(12 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**3 + 3 * (delta * epsilon - 1)**3 - 30 * epsilon**2 * kappa**2 * (delta * epsilon - 1) - 24 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1))
#     c46 = -(6 * (delta * epsilon - 1)**5 + 16 * epsilon**4 * kappa**4 * (delta * epsilon - 1) - 48 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**3)
#     c47 = (90 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**5 + 15 * (delta * epsilon - 1)**5 + 400 * epsilon**4 * kappa**4 * (delta * epsilon - 1) - 480 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**3 + 240 * epsilon**4 * kappa**4 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1) - 720 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**3)

#     c51 = 0
#     c52 = -(jnp.arcsin(delta) + 1)**2 * 2*(epsilon + 1) /(epsilon*kappa**2)
#     c53 = 2 -(jnp.arcsin(delta) + 1)**2 * -(epsilon + 2*jnp.log(epsilon + 1)*(epsilon + 1) + 1) /(epsilon*kappa**2)
#     c54 = - 8*(epsilon + 1)**2 -(jnp.arcsin(delta) + 1)**2 * 4*(epsilon + 1)**3 /(epsilon*kappa**2)
#     c55 = -(18*(epsilon + 1)**2 + 24*jnp.log(epsilon + 1)*(epsilon + 1)**2) -(jnp.arcsin(delta) + 1)**2 *(3*(epsilon + 1)**3 + 12*jnp.log(epsilon + 1)*(epsilon + 1)**3)/(epsilon*kappa**2)
#     c56 = -24*(epsilon + 1)**4 -(jnp.arcsin(delta) + 1)**2 * 6*(epsilon + 1)**5 /(epsilon*kappa**2)
#     c57 = (150*(epsilon + 1)**4 + 360*jnp.log(epsilon + 1)*(epsilon + 1)**4) -(jnp.arcsin(delta) + 1)**2 * -(15*(epsilon + 1)**5 + 90*jnp.log(epsilon + 1)*(epsilon + 1)**5)/(epsilon*kappa**2)

#     c61 = 0
#     c62 = -(jnp.arcsin(delta) - 1)**2 * 2*(epsilon - 1)/(epsilon*kappa**2)
#     c63 = 2-(jnp.arcsin(delta) - 1)**2 *-(epsilon + 2*jnp.log(1 - epsilon)*(epsilon - 1) - 1) /(epsilon*kappa**2)
#     c64 = -8*(epsilon - 1)**2-(jnp.arcsin(delta) - 1)**2 * 4*(epsilon - 1)**3/(epsilon*kappa**2)
#     c65 = -(24*jnp.log(1 - epsilon)*(epsilon - 1)**2 + 18*(epsilon - 1)**2)-(jnp.arcsin(delta) - 1)**2 *(12*jnp.log(1 - epsilon)*(epsilon - 1)**3 + 3*(epsilon - 1)**3) /(epsilon*kappa**2)
#     c66 = -24*(epsilon - 1)**4-(jnp.arcsin(delta) - 1)**2 * 6*(epsilon - 1)**5/(epsilon*kappa**2)
#     c67 = (360*jnp.log(1 - epsilon)*(epsilon - 1)**4 + 150*(epsilon - 1)**4)-(jnp.arcsin(delta) - 1)**2 * -(90*jnp.log(1 - epsilon)*(epsilon - 1)**5 + 15*(epsilon - 1)**5)/(epsilon*kappa**2)

#     c71 = 0
#     c72 = 2
#     c73 = -(2*jnp.log(1 - delta*epsilon) + 3) -kappa*-2*epsilon*kappa/(epsilon*(delta**2 - 1))
#     c74 = (12*(delta*epsilon - 1)**2 - 8*epsilon**2*kappa**2) -kappa*8*epsilon*kappa*(delta*epsilon - 1)**2/(epsilon*(delta**2 - 1))
#     c75 = (36*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2 + 21*(delta*epsilon - 1)**2 - 54*epsilon**2*kappa**2 - 24*epsilon**2*kappa**2*jnp.log(1 - delta*epsilon)) -kappa*(18*epsilon*kappa*(delta*epsilon - 1)**2 - 8*epsilon**3*kappa**3 + 24*epsilon*kappa*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2)/(epsilon*(delta**2 - 1))
#     c76 = (30*(delta*epsilon - 1)**4 + 16*epsilon**4*kappa**4 - 144*epsilon**2*kappa**2*(delta*epsilon - 1)**2) -kappa*(24*epsilon*kappa*(delta*epsilon - 1)**4 - 32*epsilon**3*kappa**3*(delta*epsilon - 1)**2)/(epsilon*(delta**2 - 1))
#     c77 =  -(450*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**4 + 165*(delta*epsilon - 1)**4 + 640*epsilon**4*kappa**4 - 2160*epsilon**2*kappa**2*(delta*epsilon - 1)**2 + 240*epsilon**4*kappa**4*jnp.log(1 - delta*epsilon) - 2160*epsilon**2*kappa**2*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2)     -kappa*-(48*epsilon**5*kappa**5 + 150*epsilon*kappa*(delta*epsilon - 1)**4 - 560*epsilon**3*kappa**3*(delta*epsilon - 1)**2 + 360*epsilon*kappa*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**4 - 480*epsilon**3*kappa**3*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2)/(epsilon*(delta**2 - 1))

    
#     A = jnp.array([
#         [c11, c12, c13, c14, c15, c16, c17],
#         [c21, c22, c23, c24, c25, c26, c27],
#         [c31, c32, c33, c34, c35, c36, c37],
#         [c41, c42, c43, c44, c45, c46, c47],
#         [c51, c52, c53, c54, c55, c56, c57],
#         [c61, c62, c63, c64, c65, c66, c67],
#         [c71, c72, c73, c74, c75, c76, c77]
#     ]).reshape((7, 7))
#     return A

# def calc_b(epsilon, delta, kappa, P):
#     """
#     Calculate the 7x1 vector b (right-hand side of the Grad–Shafranov equation).
    
#     Returns:
#         b: jnp.ndarray of shape (7, 1)
#     """
#     b1 = (jnp.log(epsilon + 1) * (P - 1) * (epsilon + 1) ** 2) / 2 - (P * (epsilon + 1) ** 4) / 8
#     b2 = (jnp.log(1 - epsilon) * (P - 1) * (epsilon - 1)**2) / 2 - (P * (epsilon - 1)**4) / 8
#     b3 = (jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2 * (P - 1)) / 2 - (P * (delta * epsilon - 1)**4) / 8
#     b4 = (P * (delta * epsilon - 1)**3) / 2 - ((delta * epsilon - 1) * (P - 1)) / 2 - jnp.log(1 - delta * epsilon) * (delta * epsilon - 1) * (P - 1)
#     b5 = -((jnp.arcsin(delta) + 1)**2 * (((P - 1) * (epsilon + 1)) / 2 - (P * (epsilon + 1)**3) / 2 + jnp.log(epsilon + 1) * (P - 1) * (epsilon + 1))) / (epsilon * kappa**2)
#     b6 = -((jnp.arcsin(delta) - 1)**2 * (((P - 1) * (epsilon - 1))/2 - (P * (epsilon - 1)**3)/2 + jnp.log(1 - epsilon) * (P - 1) * (epsilon - 1)))/(epsilon*kappa**2)
#     b7 = (3*P)/2 + jnp.log(1 - delta*epsilon)*(P - 1) - (3*P*(delta*epsilon - 1)**2)/2 - 3/2

#     b = jnp.array([b1, b2, b3, b4, b5, b6, b7]).reshape((7, 1))
#     return b

# def psi_analytic(R, Z, c1, c2, c3, c4, c5, c6, c7, P):
#     """
#     Compute the analytic solution ψ as the sum of a particular solution (psi_P)
#     and a homogeneous solution (psi_H).
    
#     Parameters:
#         R, Z: Coordinates.
#         c1,...,c7: Coefficients from the linear system.
#         P: Pressure parameter.
    
#     Returns:
#         psi: The computed analytic solution.
#     """
#     psi_P = (P * R**4) / 8 + (1 - P) * (R**2) / 2 * jnp.log(R)
#     psi1 = 1
#     psi2 = R**2
#     psi3 = Z**2 - R**2 * jnp.log(R)
#     psi4 = R**4 - 4 * R**2 * Z**2
#     psi5 = 2 * Z**4 - 9 * Z**2 * R**2 + 3 * R**4 * jnp.log(R) - 12 * R**2 * Z**2 * jnp.log(R)
#     psi6 = R**6 - 12 * R**4 * Z**2 + 8 * R**2 * Z**4
#     psi7 = 8 * Z**6 - 140 * Z**4 * R**2 + 75 * Z**2 * R**4 - 15 * R**6 * jnp.log(R) \
#            + 180 * R**4 * Z**2 * jnp.log(R) - 120 * R**2 * Z**4 * jnp.log(R)
#     psi_H = c1 * psi1 + c2 * psi2 + c3 * psi3 + c4 * psi4 + c5 * psi5 + c6 * psi6 + c7 * psi7
#     return psi_P + psi_H

def ana_sol(x, y): #analytical solution for helena solovev 
    tau = config.delta_vals
    eps = config.epsilon
    E = config.kappa_vals
    T_1 = (x - eps/2 * (1 - x**2))**2
    T_2 = (1 - eps**2/4)*((1 + eps*x)**2)
    T_3 = tau * x * (1 + eps * x / 2)
    T_4 = (y**2)/(E**2)
    psi =  T_1 + (T_2 + T_3)*(T_4)
    return psi


def hel_editor(hel, B_val):
    hel.namelist['phys']['B'] =  B_val
    hel.namelist['shape']['ias'] = 1
    hel.namelist['profile']['aga'] = 0
    hel.namelist['num']['nr'] = 51
    hel.namelist['num']['np'] = 128
    hel.namelist['num']['nrmap'] = 101
    hel.namelist['num']['npmap'] = 128
    hel.namelist['num']['nchi'] = 128
    print(hel.namelist)
    
    hel.run()
    
    print(hel)
    
    eq_name = f"f_-1_p_-1_B_{B_val}.npy"
    id = 30
    for id in range(1, len(hel.s), 4):
        plt.scatter(hel.rgrid[id, :], hel.zgrid[id, :])
    
        # Construct psi on the same grid
        psi_grid = hel.s[::-1, None]**2 * np.ones_like(hel.rgrid)
        
        # Save everything
        np.save(
         eq_name, #change numbering when making changes
        {
            "R": hel.rgrid,
            "Z": hel.zgrid,
            "psi": psi_grid,
            "s": hel.s
        })
         
    import sys     
    sys.path.append('/content/GS_PINN_full/GS_PINN_template')
    # from GS_PINN_template import configv2 as config
    # from GS_PINN_template import utilsv2 as utils
        
    print('config.A_helena:',config.A_helena)
    
    # redefine variables     
    A_helena = hel.abc[0]
    B_helena = hel.abc[1]
    eq = np.load(f'C:/Users/marcu/Downloads/{eq_name}', allow_pickle=True).item()
        
    print('new config.A_helena:',config.A_helena, 'B value', B_val)
    #print('new config.B_vals:',config.B_vals)
    return eq, A_helena, B_helena    

def training_data(eq, B, A):
    R_data = eq["R"]
    Z_data = eq["Z"]
    psi_data = eq["psi"]
    print(R_data.shape, Z_data.shape, psi_data.shape)
    cs = plt.pcolormesh(R_data[:], Z_data[:], psi_data[:])
    plt.colorbar(cs)

    # lcfs_path = cs.collections[0].get_paths()[0]
    # lcfs_vertices = lcfs_path.vertices
    # plt.close()

    R_lcfs = R_data[0]
    Z_lcfs = Z_data[0]
    plt.plot(R_lcfs, Z_lcfs, 'k-')
    plt.show()

    # from scipy.interpolate import splprep, splev

    # tck, _ = splprep([R_lcfs, Z_lcfs], s=1e-5, per=True)
    # u = np.linspace(0, 1, 400)   # number of BC points
    # R_bc, Z_bc = splev(u, tck)

    i_bc = np.column_stack([R_lcfs, Z_lcfs])
    labels_bc = np.ones((len(R_lcfs), 1))



    R_eq = R_data.ravel()
    Z_eq = Z_data.ravel()
    psi_eq = psi_data.ravel()
    #psi_eq.reshape(100, 100)

    #changing coordinate systems 
    x_eq = R_eq
    y_eq = Z_eq
    print(x_eq.min())
    print(x_eq.max())
    print(y_eq.min())
    print(y_eq.max())





    # HELENA already uses normalized psī = ψ / ψ_boundary
    psi_bar_eq = psi_eq[::, None]#[:: -1, None]

    #Interpolate between coordinate systems 
    from scipy.interpolate import griddata

    # Define PINN training grid (in x,y)
    Nx, Ny = config.n_train_x, config.n_train_y
    x_lin = np.linspace(x_eq.min(), x_eq.max(), Nx)
    y_lin = np.linspace(y_eq.min(), y_eq.max(), Ny)
    xg, yg = np.meshgrid(x_lin, y_lin, indexing="xy")

    data_train = np.column_stack([xg.ravel(), yg.ravel()])
    # data_train.shape == (10000, 2)

    # Interpolate HELENA ψ onto PINN grid
    psi_grid = griddata(
        points=np.column_stack([x_eq, y_eq]),
        values=psi_bar_eq.ravel(),
        xi=data_train,
        method="cubic"
    )[:, None]
    #psi_grid.shape == (Ny, Nx)

    # Flatten interpolation 
    #inputs = np.stack([xg.flatten(), yg.flatten()], axis=1)
    labels = psi_grid.reshape(-1,1) #psi_grid.flatten()


    # Remove NaNs (outside LCFS)
    mask = ~np.isnan(labels[:, 0])
    data_train = data_train[mask]
    labels = labels[mask]
    
    #extract constant terms in p' and ff' expressions to place in rhs source. 
    pprime_const = config.pprime_const
    ffprime_const = config.ffprime_const  #check whether it should be pinn u or raw psi
    epsilon = config.epsilon
    
    g =  A * (ffprime_const + B*(1 + epsilon * data_train[:, 0:1])**2 * pprime_const)
    
    return data_train, labels, i_bc, g 


def generate_data(eq, epsilon, kappa, delta, B, n_x, n_y, data_train_all, task): #to be integrated into helena later 
    """
    Generate training data (coordinates, labels, and source term) for a given task.
    
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
    # Solve for coefficients using the linear system
    #c_n = jnp.linalg.solve(calc_A(epsilon, delta, kappa, P), calc_b(epsilon, delta, kappa, P))
    
    # Create a meshgrid in polar coordinates
    # radial = jnp.linspace(0, epsilon, n_radial)
    # angular = jnp.linspace(0, 2 * jnp.pi, n_angular)
    # radial_mesh, angular_mesh = jnp.meshgrid(radial, angular)
    
    R_data = eq["R"]
    Z_data = eq["Z"]
    psi_data = eq["psi"]
    print(R_data.shape, Z_data.shape, psi_data.shape)
    # cs = plt.pcolormesh(R_data[:], Z_data[:], psi_data[:])
    # plt.colorbar(cs)

    # lcfs_path = cs.collections[0].get_paths()[0]
    # lcfs_vertices = lcfs_path.vertices
    # plt.close()

    R_lcfs = R_data[0]
    Z_lcfs = Z_data[0]
    
    i_bc = np.column_stack([R_lcfs, Z_lcfs])
    labels_bc = np.ones((len(R_lcfs), 1))
    
    R_eq = R_data.ravel()
    Z_eq = Z_data.ravel()
    psi_eq = psi_data.ravel()
    
    #changing coordinate systems 
    x_eq = R_eq
    y_eq = Z_eq
    
    # # Define PINN training grid (in x,y)
    # Nx, Ny = n_x, n_y
    # x_lin = np.linspace(x_eq.min(), x_eq.max(), Nx)
    # y_lin = np.linspace(y_eq.min(), y_eq.max(), Ny)
    # xg, yg = np.meshgrid(x_lin, y_lin, indexing="xy")

    # data_train = np.column_stack([xg.ravel(), yg.ravel()])
    
    reference_inputs = data_train_all[task]

    
    # HELENA already uses normalized psī = ψ / ψ_boundary
    psi_bar_eq = psi_eq[::, None]#[:: -1, None]
    
    # Interpolate HELENA ψ onto PINN grid
    psi_grid = griddata(
        points=np.column_stack([x_eq, y_eq]),
        values=psi_bar_eq.ravel(),
        xi=reference_inputs,
        method="cubic"
    )[:, None]
    
    # Flatten interpolation 
    #inputs = np.stack([xg.flatten(), yg.flatten()], axis=1)
    labels = psi_grid.reshape(-1,1) #psi_grid.flatten()


    # Remove NaNs (outside LCFS)
    mask = ~np.isnan(labels[:, 0])
    data_train = reference_inputs
    labels = labels[mask]
    
    #extract constant terms in p' and ff' expressions to place in rhs source. 
    pprime_const = -1
    ffprime_const = -1  #check whether it should be pinn u or raw psi
    
    Gamma = Gamma_gen(config.A_helena, config.epsilon, pprime_const, ffprime_const)
    Pi = Pi_gen(config.A_helena, B, pprime_const)
    
    g =  config.A_helena * (ffprime_const + B*(1 + epsilon * data_train[:, 0:1])**2 * pprime_const)
    
    #g = config.A_helena * (Gamma + B * data_train[:, 0:1] * (1.0 + data_train[:, 0:1] * epsilon / 2.0) * Pi)
    
    # # Convert polar to Cartesian coordinates
    # R_train = 1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta) * jnp.sin(angular_mesh))
    # Z_train = radial_mesh * kappa * jnp.sin(angular_mesh)
    # data_train = jnp.column_stack([R_train.ravel(), Z_train.ravel()])
    
    # Compute analytic solution labels
    #labels = psi_analytic(data_train[:, 0:1], data_train[:, 1:2], *c_n, P)
    #g = P * data_train[:, 0:1]**2 + (1 - P)
    #i_bc = jnp.where(radial_mesh.ravel() == epsilon, 1, 0)
    return data_train, labels, g, i_bc

    
def plot_slice(inputs, labels, title, save_path=None):
    """
    Plot a slice of the solution at Z = 0.
    
    Parameters:
        inputs: Cartesian coordinates (R, Z).
        labels: Solution values.
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    mask = np.isclose(np.array(inputs[:, 1]), 0.0, atol=5e-3)
    R_line = np.array(inputs[:, 0])[mask]
    u_line = np.array(labels)[mask]
    sort_idx = np.argsort(R_line)
    plt.figure(figsize=(8, 5))
    plt.plot(R_line[sort_idx], u_line[sort_idx], 'b-', label="u(R,0)")
    plt.xlabel("R")
    plt.ylabel("u(R,0)")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def load_helena_equilibrium(
    psi_path,
    R_path,
    Z_path,
    R0,
    a
):
    """
    Load HELENA equilibrium and convert to normalized (x, y).
    """
    psi = np.load(psi_path)
    R = np.load(R_path)
    Z = np.load(Z_path)

    # Normalize coordinates
    x = (R - R0) / a
    y = Z / a

    # Flatten for PINN training
    X, Y = np.meshgrid(x, y, indexing="ij")

    coords = jnp.stack(
        [X.flatten(), Y.flatten()],
        axis=1
    )
    psi_flat = jnp.array(psi.flatten())

    return coords, psi_flat

def Gamma_const(A, eps): #where p' and ff' are constant
    return 2.0 / (A * eps)

def Pi_const(A, B): #where p' and ff' are constant
    return 2.0 / (A * B)

def Gamma_gen(A, eps, pprime, ffprime): #general p' and ff'
    return -1/(A * eps) * (pprime + ffprime)

def Pi_gen(A, B, pprime):   #general p' and ff'
    return -2 * pprime /(A * B)

def helena_rhs(x, psi, A, B, eps):
    """
    RHS of the normalized Grad–Shafranov equation.
    """
    # Gamma = Gamma_const(A, eps)
    # Pi = Pi_const(A, B)
    
    Gamma = Gamma_gen(A, eps, -1, -1) #update when psi can be parameterized
    Pi = Pi_gen(A, B, -1)
    
    # return A * (
    #     Gamma +
    #     B * x * (1.0 + eps / 2.0) * Pi
    # )
    
    return A * (Gamma + B * x * (1.0 + eps * x / 2.0) * Pi) 


    
def evaluate_task(task, data_train_all, label_train_all, source_term_all, i_bc,
                  model, params_flat, format_params_fn, config):

    inputs = jnp.asarray(data_train_all[task])   # (x, y)
    labels = jnp.asarray(label_train_all[task]) # HELENA ψ
    
    # Boundary data (lcfs)
    pred_bc = model.apply(format_params_fn(params_flat), i_bc)
    #u_bc = pred_bc[:, 0:1]              # ψ on LCFS
    u_bc, _, _, _ = jnp.split (pred_bc, 4, axis=1)
    print(u_bc.shape)
    psi_bc = jnp.ones((u_bc.shape[0], 1))        # ψ = 1 on LCFS

    pred = model.apply(format_params_fn(params_flat), inputs)
    print("pred shape:", pred.shape)
    u, u_x, u_xx, u_yy = jnp.split(pred, 4, axis=1)
    x = inputs[:, 0:1]

    # Inferred parameter
    B = config.B_vals
    #B = 2.0 * nn.sigmoid(params_flat[-3])

    # RHS
    rhs = helena_rhs(x, u, config.A_helena, B, config.epsilon)

    # PDE residual
    pde = u_xx - (config.epsilon / (1 + config.epsilon * x)) * u_x + u_yy #- rhs

    # Regularization parameters
    lmbda = 10 ** (nn.sigmoid(params_flat[-2]) * 8 - 6)
    lamb  = 10 ** (nn.sigmoid(params_flat[-1]) * 16 - 14)

    # Linear system
    A_sys = jnp.vstack([pde * lmbda, u_bc])#u[i_bc]])
    b_sys = jnp.vstack([rhs * lmbda, psi_bc]) # labels[i_bc]])

    w = jnp.linalg.solve(
        lamb * jnp.eye(A_sys.shape[1]) + A_sys.T @ A_sys,
        A_sys.T @ b_sys
    )

    u_pred = u @ w

    return {
        "task": task,
        "B": config.B_vals, #float(B),
        "mse": float(jnp.mean((labels - u_pred)**2)),
        "rl2": float(jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)),
        "pde_loss": float(jnp.mean(pde**2)),
        "bc_loss": float(jnp.mean((u_bc - psi_bc)**2)),#(jnp.mean((u[i_bc] - labels[i_bc])**2)),
        "u_pred": np.array(u_pred)
    }




# def evaluate_task(task, data_train_all, label_train_all, source_term_all, i_bc,
#                   model, params_flat, format_params_fn, config):
#     """
#     Evaluate the trained model on a given task using a regularized pseudo-inverse.
#     Also logs PDE loss, BC loss, lamb, and lmbda.
#     Returns a dictionary with task metrics.
#     """
#     # Extract data for the task.
#     inputs = jnp.asarray(data_train_all[task])
#     labels = jnp.asarray(label_train_all[task])
#     g = jnp.asarray(source_term_all[task])
    
#     # Compute model predictions.
#     pred = model.apply(format_params_fn(params_flat), inputs)
#     u, u_R, u_RR, u_ZZ = jnp.split(pred, 4, axis=1)
#     R = inputs[:, 0:1]
    
#     # Compute the PDE residual.
#     pde = u_RR - (1 / R) * u_R + u_ZZ
    
#     # Transform extra parameters.
#     lmbda = 10 ** (nn.sigmoid(params_flat[-2]) * 8 - 6)
#     lamb = 10 ** (nn.sigmoid(params_flat[-1]) * 16 - 14)
    
#     # Construct the linear system.
#     A = jnp.vstack([pde * lmbda, u[i_bc]])
#     b = jnp.vstack([g * lmbda, labels[i_bc]])
    
#     # Solve using the regularized pseudo-inverse (ridge solve).
#     As = lamb * jnp.eye(A.shape[1]) + (A.T @ A)
#     bs = A.T @ b
#     w = jnp.linalg.solve(As, bs)
    
#     # Compute predicted solution.
#     u_pred = u @ w

#     # Compute error metrics.
#     mse = float(jnp.mean((labels - u_pred)**2))
#     ssr = float(jnp.sum((b - A @ w)**2))
#     rl2 = float(jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels))
    
#     # Compute PDE and boundary (BC) losses.
#     pde_loss = float(jnp.mean((pde * lmbda - g * lmbda)**2))
#     bc_loss = float(jnp.mean((u[i_bc] - labels[i_bc])**2))
    
#     # Pack metrics into a dictionary.
#     result = {
#         'task': task,
#         'kappa': float(data_train_all[task][0, 0] - 1),
#         'delta': config.delta_vals,
#         'P': config.P_vals,
#         'mse': mse,
#         'ssr': ssr,
#         'rl2': rl2,
#         'pde_loss': pde_loss,
#         'bc_loss': bc_loss,
#         'lmbda': float(lmbda),
#         'lamb': float(lamb),
#         'u_pred': np.array(u_pred)  # predicted solution for plotting
#     }
#     return result


def evaluate_helena(model, params_flat, format_params_fn,
                    coords, psi_true, A, eps):
    params = format_params_fn(params_flat)
    pred = model.apply(params, coords)

    psi = pred[:, 0:1]
    psi_xx = pred[:, 2:3]
    psi_yy = pred[:, 3:4]
    psi_x = pred[:, 1:2]

    x = coords[:, 0:1]

    # infer B
    B = 2.0 * nn.sigmoid(params_flat[-3])

    rhs = A * (
        2.0 / (A * eps)
        + B * x * (1.0 + eps / 2.0) * (2.0 / (A * B))
    )

    pde = psi_xx + psi_yy - eps / (1 + eps * x) * psi_x - rhs

    mse = jnp.mean((psi - psi_true)**2)
    pde_loss = jnp.mean(pde**2)

    return {
        "mse": float(mse),
        "pde_loss": float(pde_loss),
        "B": float(B),
        "psi_pred": np.array(psi),
    }

def plot_evaluation(task, data_train_all, label_train_all, eval_result, config, save_path=None):
    import matplotlib.tri as tri

    plt.rcParams.update({
        'axes.labelsize': 20,
        'axes.titlesize': 14,
        'figure.titlesize': 14,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
    })

    fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)

    # Extract correct task
    inputs = np.array(data_train_all[task])
    labels = np.array(label_train_all[task]).squeeze()
    u_pred = eval_result['u_pred'].squeeze()

    x_train = inputs[:, 0]
    y_train = inputs[:, 1]

    # Triangulation for irregular domain
    triang = tri.Triangulation(x_train, y_train)

    mse_values = (labels - u_pred) ** 2

    vmin = min(labels.min(), u_pred.min())
    vmax = max(labels.max(), u_pred.max())

    # -------------------------
    # Analytical Solution
    # -------------------------
    cs1 = axs[0, 0].tricontourf(triang, labels, levels=40, cmap='rainbow',
                                vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('HELENA profile')#('Analytical Solution')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    fig.colorbar(cs1, ax=axs[0, 0])

    # -------------------------
    # Predicted Solution
    # -------------------------
    cs2 = axs[0, 1].tricontourf(triang, u_pred, levels=40, cmap='rainbow',
                                vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('PINN profile')#('Predicted Solution')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    fig.colorbar(cs2, ax=axs[0, 1])

    # -------------------------
    # MSE Contour
    # -------------------------
    cs3 = axs[1, 1].tricontourf(triang, mse_values, levels=40, cmap='rainbow')
    axs[1, 1].set_title('MSE Contour')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    fig.colorbar(cs3, ax=axs[1, 1])

    # -------------------------
    # Z = 0 Slice
    # -------------------------
    Z_line = 0.0
    tolerance = 5e-3
    mask = np.isclose(y_train, Z_line, atol=tolerance)

    R_line = x_train[mask]
    mse_line = mse_values[mask]

    if len(R_line) > 0:
        sort_idx = np.argsort(R_line)
        axs[1, 0].plot(R_line[sort_idx], mse_line[sort_idx])
        axs[1, 0].set_xlabel('R')
        axs[1, 0].set_ylabel('MSE')
        axs[1, 0].set_title('Slice at Z=0')
    else:
        axs[1, 0].set_visible(False)

    fig.suptitle(
        f'ε={config.epsilon}, δ={config.delta_vals}, '
        f'κ={config.kappa_vals}, B={eval_result["B"]:.4f}',
        fontsize=16
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.close()


# For non-HELENA
# def plot_evaluation(task, data_train_all, label_train_all, eval_result, config, save_path=None):
#     plt.rcParams.update({
#         'axes.labelsize': 20,
#         'axes.titlesize': 14,
#         'figure.titlesize': 14,
#         'xtick.labelsize': 18,
#         'ytick.labelsize': 18, 
#     })

#     fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)

#     inputs = np.array(data_train_all)#[task])
#     labels = np.array(label_train_all)#[task])
#     u_pred = eval_result['u_pred']

#     # R_train = inputs[:, 0].reshape(config.n_train_angular, config.n_train_radial)
#     # Z_train = inputs[:, 1].reshape(config.n_train_angular, config.n_train_radial)
#     x_train = inputs[:, 0]
#     y_train = inputs[:, 1]
#     analytical_grid = labels.reshape(config.n_train_y, config.n_train_x) #(config.n_train_angular, config.n_train_radial)
#     predicted_grid = u_pred.reshape(config.n_train_y, config.n_train_x)  #(config.n_train_angular, config.n_train_radial)
#     mse_values = jnp.square(labels - u_pred).reshape(config.n_train_y, config.n_train_x)

#     vmin = float(min(labels.min(), u_pred.min()))
#     vmax = float(max(labels.max(), u_pred.max()))

#     # Analytical
#     cs1 = axs[0, 0].contourf(x_train, y_train, analytical_grid, levels=20, cmap='rainbow', vmin=vmin, vmax=vmax)
#     axs[0, 0].set_title('Analytical Solution')
#     axs[0, 0].set_xlabel('x')
#     axs[0, 0].set_ylabel('y')
#     cbar1 = fig.colorbar(cs1, ax=axs[0, 0]); cbar1.ax.tick_params(labelsize=18)


#     # Predicted
#     cs2 = axs[0, 1].contourf(x_train, y_train, predicted_grid, levels=20, cmap='rainbow', vmin=vmin, vmax=vmax)
#     axs[0, 1].set_title('Predicted Solution')
#     axs[0, 1].set_xlabel('x')
#     axs[0, 1].set_ylabel('y')
#     cbar2 = fig.colorbar(cs2, ax=axs[0, 1]); cbar2.ax.tick_params(labelsize=18)


#     # MSE contour
#     cs3 = axs[1, 1].contourf(x_train, y_train, mse_values, cmap='rainbow')
#     axs[1, 1].set_title('MSE Contour')
#     axs[1, 1].set_xlabel('x')
#     axs[1, 1].set_ylabel('y')
#     cbar3 = fig.colorbar(cs3, ax=axs[1, 1]); cbar3.ax.tick_params(labelsize=18)

#     # Line plot (Z=0 slice)
#     Z_line = 0.0
#     tolerance = 5e-3
#     mask = np.isclose(inputs[:, 1], Z_line, atol=tolerance)
#     R_line = inputs[mask, 0]
#     mse_line = np.array(mse_values).flatten()[mask]
#     sort_idx = np.argsort(R_line)
#     axs[1, 0].plot(R_line[sort_idx], mse_line[sort_idx])
#     axs[1, 0].set_xlabel('R')   # explicit if you prefer
#     axs[1, 0].set_ylabel('MSE')
#     axs[1, 0].set_title('Slice at Z=0')

#     fig.suptitle(f'epsilon={config.epsilon}, delta={config.delta_vals}, '
#                  f'kappa={config.kappa_vals}, B={eval_result["B"]:.4f}', #P={eval_result["P"][task]:.4f}',  
#                  fontsize=16)

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path)
#     plt.close()
    
#HELENA inverse data inference      
def eval_inverse_loss(params_inv, label_train_all, task, model, fixed_params, i_bc):

    inputs = fixed_params["inputs"]
    nn_params = fixed_params["forward"]  # already formatted
    base_params_flat = fixed_params["params_flat"]

    labels = label_train_all[task]

    # Forward pass (NN frozen)
    pred = model.apply(nn_params, inputs)
    u, u_x, u_xx, u_yy = jnp.split(pred, 4, axis=1)
    
    # Boundary data (lcfs)
    pred_bc = model.apply(nn_params, i_bc)
    #u_bc = pred_bc[:, 0:1]              # ψ on LCFS
    u_bc, _, _, _ = jnp.split (pred_bc, 4, axis=1)
    psi_bc = jnp.ones((u_bc.shape[0], 1))        # ψ = 1 on LCFS

    x = inputs[:, 0:1]

    # Infer B from inverse parameter
    B = 2.0 * nn.sigmoid(params_inv[0])

    # Physics RHS
    rhs = 2 / config.epsilon + 2 * x *(1.0 + x*config.epsilon / 2.0)
    #rhs = config.A_helena * (1.0 + B * x * (1.0 + config.epsilon / 2.0))
    
    # extract constant parts of pprime and ffprime 
    pprime = -1 #change
    ffprime = -1 #change 
    
    Gamma = -1/(config.A_helena * config.epsilon) * (pprime + ffprime)
    Pi = -2 * pprime /(config.A_helena * config.B_vals) #check correctness of this 
    
    hel_rhs = config.A_helena * (ffprime + B*(1 + config.epsilon * x)**2 * pprime)
    
    #hel_rhs = config.A_helena * (Gamma + B * x * (1.0 + x*config.epsilon / 2.0) * Pi)
    
    
    #non-constant terms in u #change depending on p' and ff' profiles, consider placing in config?
    def nonlinff(z):
        return config.A_helena * z
    
    # PDE residual
    pde = u_xx - (config.epsilon / (1.0 + config.epsilon * x)) * u_x + u_yy + nonlinff(u) #- rhs
    #pde = u_xx - (config.epsilon / (1.0 + config.epsilon * x)) * u_x + u_yy # linear solovev case 

    # Use regularization scalars from trained model
    raw_lmbda = base_params_flat[-2]
    raw_lamb  = base_params_flat[-1]

    lmbda = 10 ** (nn.sigmoid(raw_lmbda) * 8 - 6)
    lamb  = 10 ** (nn.sigmoid(raw_lamb)  * 16 - 14)

    A_sys = jnp.vstack([pde * lmbda, u_bc])
    b_sys = jnp.vstack([hel_rhs*lmbda, psi_bc])#([jnp.zeros_like(pde), labels[i_bc]])

    w = jnp.linalg.solve(
        lamb * jnp.eye(A_sys.shape[1]) + A_sys.T @ A_sys,
        A_sys.T @ b_sys
    )

    u_pred = u @ w

    mse = jnp.mean((labels - u_pred)**2)
    ssr = jnp.sum((b_sys - A_sys @ w)**2)
    rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)

    return mse, (ssr, mse, rl2)


def plot_inverse_comparison(task, inputs, true_labels, infer_labels, result, config, trial):
    """
    Generate a 2x2 figure comparing true vs inferred analytic solutions:
      TL: true analytic contour
      TR: inferred analytic contour
      BR: squared-difference contour
      BL: slice of squared difference at Z=0
    """
    
    import matplotlib.tri as tri
    # Reshape to grids
    # R = inputs[:, 0].reshape(config.n_train_angular, config.n_train_radial)
    # Z = inputs[:, 1].reshape(config.n_train_angular, config.n_train_radial)
    x_train = inputs[:, 0]#.reshape(config.n_train_x, config.n_train_y) #is the reshape necessary?
    y_train = inputs[:, 1]#.reshape(config.n_train_x, config.n_train_y)

    # Triangulation for irregular domain
    triang = tri.Triangulation(x_train, y_train)
    true_grid = np.array(true_labels).squeeze()#.reshape(config.n_train_angular, config.n_train_radial)
    infer_grid = np.array(infer_labels).squeeze()#.reshape(config.n_train_angular, config.n_train_radial)
    diff_grid = (true_grid - infer_grid) ** 2
    
    print('true grid, infer grid, diff grid:', true_grid.shape, infer_grid.shape, diff_grid.shape)

    # Set up subplots
    fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)

    # Shared color range for analytic plots
    vmin = min(true_grid.min(), infer_grid.min())
    vmax = max(true_grid.max(), infer_grid.max())
    
    print("inputs shape:", inputs.shape)
    print("R range:", inputs[:,0].min(), inputs[:,0].max())
    print("Z range:", inputs[:,1].min(), inputs[:,1].max())

    print("true_grid range:", true_grid.min(), true_grid.max())
    print("infer_grid range:", infer_grid.min(), infer_grid.max())


    # Helena solution
    cs1 = axs[0, 0].tricontourf(triang, true_grid, levels=40, vmin=vmin, vmax=vmax)
    #cs1 = axs[0, 0].contourf(x_train, y_train, true_grid, levels=20, vmin=vmin, vmax=vmax)
    #axs[0, 0].set_title('Analytical Solution (True kappa)')
    axs[0, 0].set_title('HELENA Solution (True B)')
    fig.colorbar(cs1, ax=axs[0, 0])

    # Inferred solution
    cs2 = axs[0, 1].tricontourf(triang, infer_grid, levels=40, vmin=vmin, vmax=vmax)
    #cs2 = axs[0, 1].contourf(x_train, y_train, infer_grid, levels=20, vmin=vmin, vmax=vmax)
    #axs[0, 1].set_title('Analytical Solution (Inferred kappa)')
    axs[0, 1].set_title('PINN Solution (Inferred B)')
    fig.colorbar(cs2, ax=axs[0, 1])

    # Squared-difference contour
    cs3 = axs[1, 1].tricontourf(triang, diff_grid, levels=40)#, vmin=vmin, vmax=vmax)
    #cs3 = axs[1, 1].contourf(x_train, y_train, diff_grid, levels=20)
    axs[1, 1].set_title('Squared Difference')
    fig.colorbar(cs3, ax=axs[1, 1])

    # Slice at y=0
    y_line = 0.0
    tol = 0.05 #5e-3
    mask = np.isclose(inputs[:, 1], y_line, atol=tol)
    x_line = inputs[mask, 0]
    diff_line = diff_grid.flatten()[mask]
    sort_idx = np.argsort(x_line)
    axs[1, 0].plot(x_line[sort_idx], diff_line[sort_idx])
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('Squared Difference')
    axs[1, 0].set_title('Slice at y=0')

    # Super title
    fig.suptitle(f"Inverse Task {task:02d}: true B={result['true_B']:.4f}, inferred B={result['inferred_B']:.4f}")

    # Save figure
    save_dir = f"../results/{trial}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"inverse_task_{task:02d}_plot.png")
    plt.savefig(save_path)
    plt.close(fig)
   


# def eval_inverse_loss(params_inv, task, model, fixed_params, i_bc): 

#     """
#     Inverse loss function to infer P and delta.
    
#     Args:
#         params_inv: a vector of inverse parameters of shape (1,).
#         model: the pretrained forward PINN (fixed).
#         fixed_params: the fixed forward parameters (e.g. obtained as format_params_fn(params_flat[:-2])).
#         i_bc: boundary condition indicator (same as used in the forward loss).
    
#     Returns:
#         loss: the MSE loss (between the analytic solution and u_pred).
#         aux: a tuple of additional metrics (ssr, mse, relative L2 error).
#     """
#     # Map the inverse parameters:
#     P_infer = 0.8 + 0.2 * nn.sigmoid(params_inv[0])             # ### CHANGE
#     # delta_infer = 0.2 + 0.5 * nn.sigmoid(params_inv[0])  ### CHANGE HERE
#     delta_infer = config.delta_vals  # Assume delta is fixed for now.
#     # P_infer = config.P_vals  # Assume P is fixed for now.

#     # Analytics Coefficients
#     c_n = jnp.linalg.solve(
#         calc_A(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task]), # config.delta_vals[task]
#         calc_b(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task])  # config.delta_vals[task]
#     )
    
#     inputs = fixed_params['inputs']  
    
#     # Compute the analytic solution at these coordinates.
#     R = inputs[:, 0:1]
#     # R = jnp.ravel(
#     #     1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
#     # ).reshape(-1, 1)
#     Z = inputs[:, 1:2]
#     labels = psi_analytic(R, Z, *c_n, config.P_vals[task]) ### CHANGE  will need to put P_vals[task] if changing P
    
#     # Evaluate the forward model (which is fixed) with its parameters. ###CHECK HERE
#     pred = model.apply(fixed_params['forward'], inputs)
#     u, u_R, u_RR, u_ZZ = jnp.split(pred, 4, axis=1)
    
#     # Create a polar mesh to transform the radial coordinate using inferred delta.
#     radial = jnp.linspace(0, config.epsilon, config.n_train_radial)
#     angular = jnp.linspace(0, 2 * jnp.pi, config.n_train_angular)
#     radial_mesh, angular_mesh = jnp.meshgrid(radial, angular) ###
#     # Transform R using the inferred delta:
#     R_infer = jnp.ravel(
#         1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
#     ).reshape(-1, 1)

#     ### CHECK, transforming the coordinate
#     inputs_tr = jnp.column_stack([R_infer, Z])
#     pred = model.apply(fixed_params['forward'], inputs_tr)
#     u_tr, u_R_tr, u_RR_tr, u_ZZ_tr = jnp.split(pred, 4, axis=1)
    
#     pde = u_RR_tr - (1 / R_infer) * u_R_tr + u_ZZ_tr
#     g = P_infer * R_infer**2 + (1 - P_infer)
    
#     # Use the same regularization parameters from the forward part.
#     lmbda = 10 ** (nn.sigmoid(fixed_params['params_flat'][-2]) * 8 - 6)
#     lamb_reg = 10 ** (nn.sigmoid(fixed_params['params_flat'][-1]) * 16 - 14)
    
#     A = jnp.vstack([pde * lmbda, u_tr[i_bc]])
#     b = jnp.vstack([g * lmbda, labels[i_bc]])
#     As = lamb_reg * jnp.eye(A.shape[1]) + (A.T @ A)
#     bs = A.T @ b
#     w = jnp.linalg.solve(As, bs)
    
#     u_pred = u_tr @ w
    
#     ssr = jnp.sum((b - A @ w)**2)
#     mse = jnp.mean((labels - u_pred)**2)
#     rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
#     loss = mse  # Use MSE as the inverse loss.
#     return loss, (ssr, mse, rl2)


#Second inverse loss evaluation function 
# def eval_inverse_loss(params_inv, task, model, fixed_params, i_bc):
#     """
#     Inverse loss function to infer P and delta.
    
#     Args:
#         params_inv: a vector of inverse parameters of shape (1,).
#         model: the pretrained forward PINN (fixed).
#         fixed_params: the fixed forward parameters (e.g. obtained as format_params_fn(params_flat[:-2])).
#         i_bc: boundary condition indicator (same as used in the forward loss).
    
#     Returns:
#         loss: the MSE loss (between the analytic solution and u_pred).
#         aux: a tuple of additional metrics (ssr, mse, relative L2 error).
#     """
#     # Analytical Solutions
#     # ---------------------
#     # Training for P
#     # ---------------------
#     P_infer = 0.8 + 0.2 * nn.sigmoid(params_inv[0])
#     delta_infer = config.delta_vals
#     c_n = jnp.linalg.solve(
#         calc_A(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task]), 
#         calc_b(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task])  
#     )

#     # ---------------------
#     # Training for delta
#     # ---------------------
#     # delta_infer = 0.2 + 0.5 * nn.sigmoid(params_inv[0]
#     # P_infer = config.P_vals
#     # c_n = jnp.linalg.solve(
#     #     calc_A(config.epsilon, config.delta_vals[task], config.kappa_vals, config.P_vals), 
#     #     calc_b(config.epsilon, config.delta_vals[task], config.kappa_vals, config.P_vals)  
#     # )
    
#     inputs = fixed_params['inputs']  
    
#     R = inputs[:, 0:1]
#     # R = jnp.ravel(
#     #     1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
#     # ).reshape(-1, 1)
#     Z = inputs[:, 1:2]

#     # ---------------------
#     # Training for P
#     # ---------------------
#     labels = psi_analytic(R, Z, *c_n, config.P_vals[task]) 
    
#     # ---------------------
#     # Training for delta
#     # ---------------------
#     # labels = psi_analytic(R, Z, *c_n, config.P_vals) 

#     # Meta-PINN Prediction with transformed coordinates
#     radial = jnp.linspace(0, config.epsilon, config.n_train_radial)
#     angular = jnp.linspace(0, 2 * jnp.pi, config.n_train_angular)
#     radial_mesh, angular_mesh = jnp.meshgrid(radial, angular) 
#     R_infer = jnp.ravel(
#         1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
#     ).reshape(-1, 1)

#     inputs_tr = jnp.column_stack([R_infer, Z])
#     pred = model.apply(fixed_params['forward'], inputs_tr)
#     u_tr, u_R_tr, u_RR_tr, u_ZZ_tr = jnp.split(pred, 4, axis=1)
    
#     pde = u_RR_tr - (1 / R_infer) * u_R_tr + u_ZZ_tr
#     g = P_infer * R_infer**2 + (1 - P_infer)
    
#     lmbda = 10 ** (nn.sigmoid(fixed_params['params_flat'][-2]) * 8 - 6)
#     lamb_reg = 10 ** (nn.sigmoid(fixed_params['params_flat'][-1]) * 16 - 14)
    
#     A = jnp.vstack([pde * lmbda, u_tr[i_bc]])
#     b = jnp.vstack([g * lmbda, labels[i_bc]])
#     As = lamb_reg * jnp.eye(A.shape[1]) + (A.T @ A)
#     bs = A.T @ b
#     w = jnp.linalg.solve(As, bs)
    
#     u_pred = u_tr @ w
    
#     ssr = jnp.sum((b - A @ w)**2)
#     mse = jnp.mean((labels - u_pred)**2)
#     rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
#     loss = mse  
#     return loss, (ssr, mse, rl2)

def update_inverse(params_inv, opt_state_inv, task, model, fixed_params, label_train_all, i_bc, inverse_optimizer):
    (loss_val, aux), grad = jax.value_and_grad(eval_inverse_loss, has_aux=True)(
        params_inv, label_train_all, task, model, fixed_params, i_bc
    )
    updates, opt_state_inv = inverse_optimizer.update(grad, opt_state_inv)
    params_inv = optax.apply_updates(params_inv, updates)
    return params_inv, opt_state_inv, loss_val, aux

"""
Utility functions for saving and loading model parameters using Flax serialization.
"""

def save_meta_model(params_flat, filepath):
    """
    Save the meta model's flattened parameters using Flax serialization.
    
    Parameters:
        params_flat: The flat JAX array of meta model parameters.
        filepath: Path to save the serialized model.
    """
    serialized_bytes = flax.serialization.to_bytes(params_flat)
    with open(filepath, "wb") as f:
        f.write(serialized_bytes)
    print(f"Meta model saved to {filepath}")

def load_meta_model(filepath, target):
    """
    Load the meta model's flattened parameters using Flax serialization.
    
    Parameters:
        filepath: Path to the saved meta model file.
        target: A target object that has the same structure as the saved parameters.
        
    Returns:
        A JAX array of the flattened meta model parameters.
    """
    with open(filepath, "rb") as f:
        serialized_bytes = f.read()
    params_flat = flax.serialization.from_bytes(target, serialized_bytes)
    return params_flat
