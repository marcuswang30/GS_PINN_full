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

import config  

def calc_A(epsilon, delta, kappa, P):
    """
    Calculate the 7x7 matrix A (coefficients for the Grad–Shafranov equation).
    
    Returns:
        A: jnp.ndarray of shape (7, 7)
    """
    # Coefficients for the first row
    c11 = 1
    c12 = (epsilon + 1)**2
    c13 = -jnp.log(epsilon + 1) * (epsilon + 1)**2
    c14 = (epsilon + 1)**4
    c15 = 3 * jnp.log(epsilon + 1) * (epsilon + 1)**4
    c16 = (epsilon + 1)**6
    c17 = -15 * jnp.log(epsilon + 1) * (epsilon + 1)**6

    # Coefficients for the second row
    c21 = 1
    c22 = (epsilon - 1)**2
    c23 = -jnp.log(1 - epsilon) * (epsilon - 1)**2
    c24 = (epsilon - 1)**4
    c25 = 3 * jnp.log(1 - epsilon) * (epsilon - 1)**4
    c26 = (epsilon - 1)**6
    c27 = -15 * jnp.log(1 - epsilon) * (epsilon - 1)**6

    # Coefficients for the third row (example, similar structure follows for rows 4-7)
    c31 = 1
    c32 = (delta * epsilon - 1)**2
    c33 = -(jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2 - epsilon**2 * kappa**2)
    c34 = ((delta * epsilon - 1)**4 - 4 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**2)
    c35 = (3 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**4 + 2 * epsilon**4 * kappa**4
           - 9 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**2 - 12 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2)
    c36 = ((delta * epsilon - 1)**6 - 12 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**4 + 8 * epsilon**4 * kappa**4 * (delta * epsilon - 1)**2)
    c37 = -(15 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**6 - 8 * epsilon**6 * kappa**6 - 75 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**4
           + 140 * epsilon**4 * kappa**4 * (delta * epsilon - 1)**2 - 180 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**4
           + 120 * epsilon**4 * kappa**4 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2)
    
    c41 = 0
    c42 = -2 * (delta * epsilon - 1)
    c43 = (delta * epsilon + 2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1) - 1)
    c44 = -(4 * (delta * epsilon - 1)**3 - 8 * epsilon**2 * kappa**2 * (delta * epsilon - 1))
    c45 = -(12 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**3 + 3 * (delta * epsilon - 1)**3 - 30 * epsilon**2 * kappa**2 * (delta * epsilon - 1) - 24 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1))
    c46 = -(6 * (delta * epsilon - 1)**5 + 16 * epsilon**4 * kappa**4 * (delta * epsilon - 1) - 48 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**3)
    c47 = (90 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**5 + 15 * (delta * epsilon - 1)**5 + 400 * epsilon**4 * kappa**4 * (delta * epsilon - 1) - 480 * epsilon**2 * kappa**2 * (delta * epsilon - 1)**3 + 240 * epsilon**4 * kappa**4 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1) - 720 * epsilon**2 * kappa**2 * jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**3)

    c51 = 0
    c52 = -(jnp.arcsin(delta) + 1)**2 * 2*(epsilon + 1) /(epsilon*kappa**2)
    c53 = 2 -(jnp.arcsin(delta) + 1)**2 * -(epsilon + 2*jnp.log(epsilon + 1)*(epsilon + 1) + 1) /(epsilon*kappa**2)
    c54 = - 8*(epsilon + 1)**2 -(jnp.arcsin(delta) + 1)**2 * 4*(epsilon + 1)**3 /(epsilon*kappa**2)
    c55 = -(18*(epsilon + 1)**2 + 24*jnp.log(epsilon + 1)*(epsilon + 1)**2) -(jnp.arcsin(delta) + 1)**2 *(3*(epsilon + 1)**3 + 12*jnp.log(epsilon + 1)*(epsilon + 1)**3)/(epsilon*kappa**2)
    c56 = -24*(epsilon + 1)**4 -(jnp.arcsin(delta) + 1)**2 * 6*(epsilon + 1)**5 /(epsilon*kappa**2)
    c57 = (150*(epsilon + 1)**4 + 360*jnp.log(epsilon + 1)*(epsilon + 1)**4) -(jnp.arcsin(delta) + 1)**2 * -(15*(epsilon + 1)**5 + 90*jnp.log(epsilon + 1)*(epsilon + 1)**5)/(epsilon*kappa**2)

    c61 = 0
    c62 = -(jnp.arcsin(delta) - 1)**2 * 2*(epsilon - 1)/(epsilon*kappa**2)
    c63 = 2-(jnp.arcsin(delta) - 1)**2 *-(epsilon + 2*jnp.log(1 - epsilon)*(epsilon - 1) - 1) /(epsilon*kappa**2)
    c64 = -8*(epsilon - 1)**2-(jnp.arcsin(delta) - 1)**2 * 4*(epsilon - 1)**3/(epsilon*kappa**2)
    c65 = -(24*jnp.log(1 - epsilon)*(epsilon - 1)**2 + 18*(epsilon - 1)**2)-(jnp.arcsin(delta) - 1)**2 *(12*jnp.log(1 - epsilon)*(epsilon - 1)**3 + 3*(epsilon - 1)**3) /(epsilon*kappa**2)
    c66 = -24*(epsilon - 1)**4-(jnp.arcsin(delta) - 1)**2 * 6*(epsilon - 1)**5/(epsilon*kappa**2)
    c67 = (360*jnp.log(1 - epsilon)*(epsilon - 1)**4 + 150*(epsilon - 1)**4)-(jnp.arcsin(delta) - 1)**2 * -(90*jnp.log(1 - epsilon)*(epsilon - 1)**5 + 15*(epsilon - 1)**5)/(epsilon*kappa**2)

    c71 = 0
    c72 = 2
    c73 = -(2*jnp.log(1 - delta*epsilon) + 3) -kappa*-2*epsilon*kappa/(epsilon*(delta**2 - 1))
    c74 = (12*(delta*epsilon - 1)**2 - 8*epsilon**2*kappa**2) -kappa*8*epsilon*kappa*(delta*epsilon - 1)**2/(epsilon*(delta**2 - 1))
    c75 = (36*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2 + 21*(delta*epsilon - 1)**2 - 54*epsilon**2*kappa**2 - 24*epsilon**2*kappa**2*jnp.log(1 - delta*epsilon)) -kappa*(18*epsilon*kappa*(delta*epsilon - 1)**2 - 8*epsilon**3*kappa**3 + 24*epsilon*kappa*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2)/(epsilon*(delta**2 - 1))
    c76 = (30*(delta*epsilon - 1)**4 + 16*epsilon**4*kappa**4 - 144*epsilon**2*kappa**2*(delta*epsilon - 1)**2) -kappa*(24*epsilon*kappa*(delta*epsilon - 1)**4 - 32*epsilon**3*kappa**3*(delta*epsilon - 1)**2)/(epsilon*(delta**2 - 1))
    c77 =  -(450*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**4 + 165*(delta*epsilon - 1)**4 + 640*epsilon**4*kappa**4 - 2160*epsilon**2*kappa**2*(delta*epsilon - 1)**2 + 240*epsilon**4*kappa**4*jnp.log(1 - delta*epsilon) - 2160*epsilon**2*kappa**2*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2)     -kappa*-(48*epsilon**5*kappa**5 + 150*epsilon*kappa*(delta*epsilon - 1)**4 - 560*epsilon**3*kappa**3*(delta*epsilon - 1)**2 + 360*epsilon*kappa*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**4 - 480*epsilon**3*kappa**3*jnp.log(1 - delta*epsilon)*(delta*epsilon - 1)**2)/(epsilon*(delta**2 - 1))

    
    A = jnp.array([
        [c11, c12, c13, c14, c15, c16, c17],
        [c21, c22, c23, c24, c25, c26, c27],
        [c31, c32, c33, c34, c35, c36, c37],
        [c41, c42, c43, c44, c45, c46, c47],
        [c51, c52, c53, c54, c55, c56, c57],
        [c61, c62, c63, c64, c65, c66, c67],
        [c71, c72, c73, c74, c75, c76, c77]
    ]).reshape((7, 7))
    return A

def calc_b(epsilon, delta, kappa, P):
    """
    Calculate the 7x1 vector b (right-hand side of the Grad–Shafranov equation).
    
    Returns:
        b: jnp.ndarray of shape (7, 1)
    """
    b1 = (jnp.log(epsilon + 1) * (P - 1) * (epsilon + 1) ** 2) / 2 - (P * (epsilon + 1) ** 4) / 8
    b2 = (jnp.log(1 - epsilon) * (P - 1) * (epsilon - 1)**2) / 2 - (P * (epsilon - 1)**4) / 8
    b3 = (jnp.log(1 - delta * epsilon) * (delta * epsilon - 1)**2 * (P - 1)) / 2 - (P * (delta * epsilon - 1)**4) / 8
    b4 = (P * (delta * epsilon - 1)**3) / 2 - ((delta * epsilon - 1) * (P - 1)) / 2 - jnp.log(1 - delta * epsilon) * (delta * epsilon - 1) * (P - 1)
    b5 = -((jnp.arcsin(delta) + 1)**2 * (((P - 1) * (epsilon + 1)) / 2 - (P * (epsilon + 1)**3) / 2 + jnp.log(epsilon + 1) * (P - 1) * (epsilon + 1))) / (epsilon * kappa**2)
    b6 = -((jnp.arcsin(delta) - 1)**2 * (((P - 1) * (epsilon - 1))/2 - (P * (epsilon - 1)**3)/2 + jnp.log(1 - epsilon) * (P - 1) * (epsilon - 1)))/(epsilon*kappa**2)
    b7 = (3*P)/2 + jnp.log(1 - delta*epsilon)*(P - 1) - (3*P*(delta*epsilon - 1)**2)/2 - 3/2

    b = jnp.array([b1, b2, b3, b4, b5, b6, b7]).reshape((7, 1))
    return b

def psi_analytic(R, Z, c1, c2, c3, c4, c5, c6, c7, P):
    """
    Compute the analytic solution ψ as the sum of a particular solution (psi_P)
    and a homogeneous solution (psi_H).
    
    Parameters:
        R, Z: Coordinates.
        c1,...,c7: Coefficients from the linear system.
        P: Pressure parameter.
    
    Returns:
        psi: The computed analytic solution.
    """
    psi_P = (P * R**4) / 8 + (1 - P) * (R**2) / 2 * jnp.log(R)
    psi1 = 1
    psi2 = R**2
    psi3 = Z**2 - R**2 * jnp.log(R)
    psi4 = R**4 - 4 * R**2 * Z**2
    psi5 = 2 * Z**4 - 9 * Z**2 * R**2 + 3 * R**4 * jnp.log(R) - 12 * R**2 * Z**2 * jnp.log(R)
    psi6 = R**6 - 12 * R**4 * Z**2 + 8 * R**2 * Z**4
    psi7 = 8 * Z**6 - 140 * Z**4 * R**2 + 75 * Z**2 * R**4 - 15 * R**6 * jnp.log(R) \
           + 180 * R**4 * Z**2 * jnp.log(R) - 120 * R**2 * Z**4 * jnp.log(R)
    psi_H = c1 * psi1 + c2 * psi2 + c3 * psi3 + c4 * psi4 + c5 * psi5 + c6 * psi6 + c7 * psi7
    return psi_P + psi_H

def generate_data(epsilon, kappa, delta, P, n_radial, n_angular):
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
    c_n = jnp.linalg.solve(calc_A(epsilon, delta, kappa, P), calc_b(epsilon, delta, kappa, P))
    
    # Create a meshgrid in polar coordinates
    radial = jnp.linspace(0, epsilon, n_radial)
    angular = jnp.linspace(0, 2 * jnp.pi, n_angular)
    radial_mesh, angular_mesh = jnp.meshgrid(radial, angular)
    
    # Convert polar to Cartesian coordinates
    R_train = 1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta) * jnp.sin(angular_mesh))
    Z_train = radial_mesh * kappa * jnp.sin(angular_mesh)
    data_train = jnp.column_stack([R_train.ravel(), Z_train.ravel()])
    
    # Compute analytic solution labels
    labels = psi_analytic(data_train[:, 0:1], data_train[:, 1:2], *c_n, P)
    g = P * data_train[:, 0:1]**2 + (1 - P)
    i_bc = jnp.where(radial_mesh.ravel() == epsilon, 1, 0)
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

def evaluate_task(task, data_train_all, label_train_all, source_term_all, i_bc,
                  model, params_flat, format_params_fn, config):
    """
    Evaluate the trained model on a given task using a regularized pseudo-inverse.
    Also logs PDE loss, BC loss, lamb, and lmbda.
    Returns a dictionary with task metrics.
    """
    # Extract data for the task.
    inputs = jnp.asarray(data_train_all[task])
    labels = jnp.asarray(label_train_all[task])
    g = jnp.asarray(source_term_all[task])
    
    # Compute model predictions.
    pred = model.apply(format_params_fn(params_flat), inputs)
    u, u_R, u_RR, u_ZZ = jnp.split(pred, 4, axis=1)
    R = inputs[:, 0:1]
    
    # Compute the PDE residual.
    pde = u_RR - (1 / R) * u_R + u_ZZ
    
    # Transform extra parameters.
    lmbda = 10 ** (nn.sigmoid(params_flat[-2]) * 8 - 6)
    lamb = 10 ** (nn.sigmoid(params_flat[-1]) * 16 - 14)
    
    # Construct the linear system.
    A = jnp.vstack([pde * lmbda, u[i_bc]])
    b = jnp.vstack([g * lmbda, labels[i_bc]])
    
    # Solve using the regularized pseudo-inverse (ridge solve).
    As = lamb * jnp.eye(A.shape[1]) + (A.T @ A)
    bs = A.T @ b
    w = jnp.linalg.solve(As, bs)
    
    # Compute predicted solution.
    u_pred = u @ w

    # Compute error metrics.
    mse = float(jnp.mean((labels - u_pred)**2))
    ssr = float(jnp.sum((b - A @ w)**2))
    rl2 = float(jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels))
    
    # Compute PDE and boundary (BC) losses.
    pde_loss = float(jnp.mean((pde * lmbda - g * lmbda)**2))
    bc_loss = float(jnp.mean((u[i_bc] - labels[i_bc])**2))
    
    # Pack metrics into a dictionary.
    result = {
        'task': task,
        'kappa': float(data_train_all[task][0, 0] - 1),
        'delta': config.delta_vals,
        'P': config.P_vals,
        'mse': mse,
        'ssr': ssr,
        'rl2': rl2,
        'pde_loss': pde_loss,
        'bc_loss': bc_loss,
        'lmbda': float(lmbda),
        'lamb': float(lamb),
        'u_pred': np.array(u_pred)  # predicted solution for plotting
    }
    return result

def plot_evaluation(task, data_train_all, label_train_all, eval_result, config, save_path=None):
    plt.rcParams.update({
        'axes.labelsize': 20,
        'axes.titlesize': 14,
        'figure.titlesize': 14,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18, 
    })

    fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)

    inputs = np.array(data_train_all[task])
    labels = np.array(label_train_all[task])
    u_pred = eval_result['u_pred']

    R_train = inputs[:, 0].reshape(config.n_train_angular, config.n_train_radial)
    Z_train = inputs[:, 1].reshape(config.n_train_angular, config.n_train_radial)
    analytical_grid = labels.reshape(config.n_train_angular, config.n_train_radial)
    predicted_grid = u_pred.reshape(config.n_train_angular, config.n_train_radial)
    mse_values = jnp.square(labels - u_pred).reshape(config.n_train_angular, config.n_train_radial)

    vmin = float(min(labels.min(), u_pred.min()))
    vmax = float(max(labels.max(), u_pred.max()))

    # Analytical
    cs1 = axs[0, 0].contourf(R_train, Z_train, analytical_grid, levels=20, cmap='rainbow', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Analytical Solution')
    axs[0, 0].set_xlabel('R')
    axs[0, 0].set_ylabel('Z')
    cbar1 = fig.colorbar(cs1, ax=axs[0, 0]); cbar1.ax.tick_params(labelsize=18)


    # Predicted
    cs2 = axs[0, 1].contourf(R_train, Z_train, predicted_grid, levels=20, cmap='rainbow', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Predicted Solution')
    axs[0, 1].set_xlabel('R')
    axs[0, 1].set_ylabel('Z')
    cbar2 = fig.colorbar(cs2, ax=axs[0, 1]); cbar2.ax.tick_params(labelsize=18)

    # MSE contour
    cs3 = axs[1, 1].contourf(R_train, Z_train, mse_values, cmap='rainbow')
    axs[1, 1].set_title('MSE Contour')
    axs[1, 1].set_xlabel('R')
    axs[1, 1].set_ylabel('Z')
    cbar3 = fig.colorbar(cs3, ax=axs[1, 1]); cbar3.ax.tick_params(labelsize=18)

    # Line plot (Z=0 slice)
    Z_line = 0.0
    tolerance = 5e-3
    mask = np.isclose(inputs[:, 1], Z_line, atol=tolerance)
    R_line = inputs[mask, 0]
    mse_line = np.array(mse_values).flatten()[mask]
    sort_idx = np.argsort(R_line)
    axs[1, 0].plot(R_line[sort_idx], mse_line[sort_idx])
    axs[1, 0].set_xlabel('R')   # explicit if you prefer
    axs[1, 0].set_ylabel('MSE')
    axs[1, 0].set_title('Slice at Z=0')

    fig.suptitle(f'epsilon={config.epsilon}, delta={config.delta_vals}, '
                 f'kappa={config.kappa_vals}, P={eval_result["P"][task]:.4f}',  
                 fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def eval_inverse_loss(params_inv, task, model, fixed_params, i_bc):

    """
    Inverse loss function to infer P and delta.
    
    Args:
        params_inv: a vector of inverse parameters of shape (1,).
        model: the pretrained forward PINN (fixed).
        fixed_params: the fixed forward parameters (e.g. obtained as format_params_fn(params_flat[:-2])).
        i_bc: boundary condition indicator (same as used in the forward loss).
    
    Returns:
        loss: the MSE loss (between the analytic solution and u_pred).
        aux: a tuple of additional metrics (ssr, mse, relative L2 error).
    """
    # Map the inverse parameters:
    P_infer = 0.8 + 0.2 * nn.sigmoid(params_inv[0])             # ### CHANGE
    # delta_infer = 0.2 + 0.5 * nn.sigmoid(params_inv[0])  ### CHANGE HERE
    delta_infer = config.delta_vals  # Assume delta is fixed for now.
    # P_infer = config.P_vals  # Assume P is fixed for now.

    # Analytics Coefficients
    c_n = jnp.linalg.solve(
        calc_A(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task]), # config.delta_vals[task]
        calc_b(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task])  # config.delta_vals[task]
    )
    
    inputs = fixed_params['inputs']  
    
    # Compute the analytic solution at these coordinates.
    R = inputs[:, 0:1]
    # R = jnp.ravel(
    #     1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
    # ).reshape(-1, 1)
    Z = inputs[:, 1:2]
    labels = psi_analytic(R, Z, *c_n, config.P_vals[task]) ### CHANGE  will need to put P_vals[task] if changing P
    
    # Evaluate the forward model (which is fixed) with its parameters. ###CHECK HERE
    pred = model.apply(fixed_params['forward'], inputs)
    u, u_R, u_RR, u_ZZ = jnp.split(pred, 4, axis=1)
    
    # Create a polar mesh to transform the radial coordinate using inferred delta.
    radial = jnp.linspace(0, config.epsilon, config.n_train_radial)
    angular = jnp.linspace(0, 2 * jnp.pi, config.n_train_angular)
    radial_mesh, angular_mesh = jnp.meshgrid(radial, angular) ###
    # Transform R using the inferred delta:
    R_infer = jnp.ravel(
        1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
    ).reshape(-1, 1)

    ### CHECK, transforming the coordinate
    inputs_tr = jnp.column_stack([R_infer, Z])
    pred = model.apply(fixed_params['forward'], inputs_tr)
    u_tr, u_R_tr, u_RR_tr, u_ZZ_tr = jnp.split(pred, 4, axis=1)
    
    pde = u_RR_tr - (1 / R_infer) * u_R_tr + u_ZZ_tr
    g = P_infer * R_infer**2 + (1 - P_infer)
    
    # Use the same regularization parameters from the forward part.
    lmbda = 10 ** (nn.sigmoid(fixed_params['params_flat'][-2]) * 8 - 6)
    lamb_reg = 10 ** (nn.sigmoid(fixed_params['params_flat'][-1]) * 16 - 14)
    
    A = jnp.vstack([pde * lmbda, u_tr[i_bc]])
    b = jnp.vstack([g * lmbda, labels[i_bc]])
    As = lamb_reg * jnp.eye(A.shape[1]) + (A.T @ A)
    bs = A.T @ b
    w = jnp.linalg.solve(As, bs)
    
    u_pred = u_tr @ w
    
    ssr = jnp.sum((b - A @ w)**2)
    mse = jnp.mean((labels - u_pred)**2)
    rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
    loss = mse  # Use MSE as the inverse loss.
    return loss, (ssr, mse, rl2)

def eval_inverse_loss(params_inv, task, model, fixed_params, i_bc):
    """
    Inverse loss function to infer P and delta.
    
    Args:
        params_inv: a vector of inverse parameters of shape (1,).
        model: the pretrained forward PINN (fixed).
        fixed_params: the fixed forward parameters (e.g. obtained as format_params_fn(params_flat[:-2])).
        i_bc: boundary condition indicator (same as used in the forward loss).
    
    Returns:
        loss: the MSE loss (between the analytic solution and u_pred).
        aux: a tuple of additional metrics (ssr, mse, relative L2 error).
    """
    # Analytical Solutions
    # ---------------------
    # Training for P
    # ---------------------
    P_infer = 0.8 + 0.2 * nn.sigmoid(params_inv[0])
    delta_infer = config.delta_vals
    c_n = jnp.linalg.solve(
        calc_A(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task]), 
        calc_b(config.epsilon, config.delta_vals, config.kappa_vals, config.P_vals[task])  
    )

    # ---------------------
    # Training for delta
    # ---------------------
    # delta_infer = 0.2 + 0.5 * nn.sigmoid(params_inv[0]
    # P_infer = config.P_vals
    # c_n = jnp.linalg.solve(
    #     calc_A(config.epsilon, config.delta_vals[task], config.kappa_vals, config.P_vals), 
    #     calc_b(config.epsilon, config.delta_vals[task], config.kappa_vals, config.P_vals)  
    # )
    
    inputs = fixed_params['inputs']  
    
    R = inputs[:, 0:1]
    # R = jnp.ravel(
    #     1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
    # ).reshape(-1, 1)
    Z = inputs[:, 1:2]

    # ---------------------
    # Training for P
    # ---------------------
    labels = psi_analytic(R, Z, *c_n, config.P_vals[task]) 
    
    # ---------------------
    # Training for delta
    # ---------------------
    # labels = psi_analytic(R, Z, *c_n, config.P_vals) 

    # Meta-PINN Prediction with transformed coordinates
    radial = jnp.linspace(0, config.epsilon, config.n_train_radial)
    angular = jnp.linspace(0, 2 * jnp.pi, config.n_train_angular)
    radial_mesh, angular_mesh = jnp.meshgrid(radial, angular) 
    R_infer = jnp.ravel(
        1 + radial_mesh * jnp.cos(angular_mesh + jnp.arcsin(delta_infer) * jnp.sin(angular_mesh))
    ).reshape(-1, 1)

    inputs_tr = jnp.column_stack([R_infer, Z])
    pred = model.apply(fixed_params['forward'], inputs_tr)
    u_tr, u_R_tr, u_RR_tr, u_ZZ_tr = jnp.split(pred, 4, axis=1)
    
    pde = u_RR_tr - (1 / R_infer) * u_R_tr + u_ZZ_tr
    g = P_infer * R_infer**2 + (1 - P_infer)
    
    lmbda = 10 ** (nn.sigmoid(fixed_params['params_flat'][-2]) * 8 - 6)
    lamb_reg = 10 ** (nn.sigmoid(fixed_params['params_flat'][-1]) * 16 - 14)
    
    A = jnp.vstack([pde * lmbda, u_tr[i_bc]])
    b = jnp.vstack([g * lmbda, labels[i_bc]])
    As = lamb_reg * jnp.eye(A.shape[1]) + (A.T @ A)
    bs = A.T @ b
    w = jnp.linalg.solve(As, bs)
    
    u_pred = u_tr @ w
    
    ssr = jnp.sum((b - A @ w)**2)
    mse = jnp.mean((labels - u_pred)**2)
    rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
    loss = mse  
    return loss, (ssr, mse, rl2)

def update_inverse(params_inv, opt_state_inv, task, model, fixed_params, i_bc, inverse_optimizer):
    (loss_val, aux), grad = jax.value_and_grad(eval_inverse_loss, has_aux=True)(
        params_inv, task, model, fixed_params, i_bc
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
