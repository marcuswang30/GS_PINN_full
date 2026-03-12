# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 13:41:46 2026

@author: marcu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:30:51 2025

@author: marcus
"""

"""
Main script for meta-training the raw PINN for the Gradâ€“Shafranov equation
by varying different parameters, while keeping the rest fixed.
Then, the trained model is evaluated on all tasks:
- The linear system is solved using regularised pseudo-inverse.
- Metrics (MSE, SSR, relative L2 error) for each task are saved to a CSV.
- For each task, a 2x2 plot is generated showing the analytical solution,
  predicted solution, MSE contour, and an MSE slice at Z=0.
  
We use this version of the code to compare against HELENA computations
(with tweaks to original PINN). In this use case we attempt to replace 
instances of P with B.
"""

import time
import optax
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ["JAX_ENABLE_X64"] = "1"                
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"  

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")

import jax
import jax.numpy as jnp

from flax import linen as nn
from jax.scipy.linalg import solve

from model import PINN
import utilsv2 as utils 
import configv2 as config 

# ---------------------
# Setup: Random seed, key, trial number, and GPU check
# ---------------------
seed = config.seed
key = jax.random.PRNGKey(seed)
trial = config.trial
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

# ---------------------
# Build task parameters: Vary certain params, keep the other fixed
# ---------------------
# ---------------------
# Training for P
# ---------------------
fixed_delta = config.delta_vals  
fixed_B = config.B_vals
#P_vals = config.P_vals
#task_params = P_vals.reshape(-1, 1)  # Shape: (n_task, 1)
# ---------------------
# Training for delta
# ---------------------
# delta_vals = config.delta_vals
# fixed_P = config.P_vals         
# task_params = delta_vals.reshape(-1, 1)  # Shape: (n_task, 1)

fixed_kappa = config.kappa_vals    
# n_task = len(task_params)
# print("Number of tasks:", n_task)
# config.n_train_tasks = n_task

# HELENA / geometry parameters
R0 = config.R0        # major radius
a = config.a          # minor radius
epsilon = config.epsilon #a / R0      # inverse aspect ratio

A_helena = config.A_helena     # known from HELENA run


# If requiring interpolated data 
from scipy.interpolate import griddata


#importing HELENA data
eq = np.load('C:/Users/marcu/Downloads/b08.npy', allow_pickle=True).item() #change according to directory


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

# ---------------------
# Generate training data for each task
# ---------------------
# Change delta & P according to your task
# data_train_list, label_train_list, source_term_list = [], [], []
# for P_val in task_params:
#     data_train, labels, g, i_bc = utils.generate_data(
#         config.epsilon, fixed_kappa, fixed_delta, P_val.item(),
#         config.n_train_radial, config.n_train_angular
#     )
#     data_train_list.append(data_train)
#     label_train_list.append(labels)
#     source_term_list.append(g)

# data_train_all = jnp.array(data_train_list)
# label_train_all = jnp.array(label_train_list)
# source_term_all = jnp.array(source_term_list)



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


# # ---------------------
# # Define boundary condition indicator (assumed same for all tasks)
# # ---------------------
# radial = jnp.linspace(0, config.epsilon, config.n_train_radial)
# angular = jnp.linspace(0, 2*jnp.pi, config.n_train_angular)
# radial_mesh, _ = jnp.meshgrid(radial, angular, indexing="xy")
# i_bc = (radial_mesh.ravel() == config.epsilon)         
# print(i_bc.shape, jnp.sum(i_bc))

# #HELENA-defined boundary conditions
# i_bc = np.isclose(labels[:, 0], 1.0, atol=1e-3)
# print(np.sum(i_bc))



#Load input data onto arrays 
data_train_all = jnp.array([data_train])
label_train_all = jnp.array([labels])
source_term_all = None   # will be constructed inside loss
n_task = 1


# ---------------------
# Initialize the PINN model
# ---------------------
n_nodes = 500  # number of nodes in dense layer
model = PINN(n_nodes=n_nodes)
a = jax.random.normal(key, [1, 2])  # dummy input for initialization
params = model.init(key, a)
params_flat, unravel_fn = jax.flatten_util.ravel_pytree(params)
# Append two extra parameters for regularisation scaling
key, subkey = jax.random.split(key)
extra_params = jax.random.normal(subkey, [2])
params_flat = jnp.append(params_flat, extra_params)

# Parameter formatting 
def format_params_fn(p):
    base_params = p[:-2]
    return unravel_fn(base_params)

# Comment from here if you just want to run inverse
# ---------------------
# Set up the optimizer
# ---------------------
lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=config.max_lr,
    peak_value=config.max_lr,
    warmup_steps=int(config.max_iters * 0.4),
    decay_steps=config.max_iters,
    end_value=1e-6
)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params_flat)

# ---------------------
# Loss function for meta-learning (raw PINN component)
# ---------------------
def eval_loss(params_flat, task):
    inputs = data_train_all[task]
    labels = label_train_all[task]
    
    # Boundary data (lcfs)
    pred_bc = model.apply(format_params_fn(params_flat), i_bc)
    #u_bc = pred_bc[:, 0:1]              # ψ on LCFS
    u_bc, _, _, _ = jnp.split (pred_bc, 4, axis=1)
    print(u_bc.shape)
    psi_bc = jnp.ones((u_bc.shape[0], 1))        # ψ = 1 on LCFS

    pred = model.apply(format_params_fn(params_flat), inputs)
    u, u_x, u_xx, u_yy = jnp.split(pred, 4, axis=1)
    
    x = inputs[:, 0:1]
    
    #extract constant terms in p' and ff' expressions to place in rhs source. 
    pprime_const = -1
    ffprime_const = -1  #check whether it should be pinn u or raw psi
    
    Gamma = utils.Gamma_gen(config.A_helena, config.epsilon, pprime_const, ffprime_const)
    Pi = utils.Pi_gen(config.A_helena, fixed_B, pprime_const)
    
    #non-constant terms in u #change depending on p' and ff' profiles 
    def nonlin(z):
        return 0
    
    # HELENA GS operator
    pde = u_xx - (epsilon / (1.0 + epsilon * x)) * u_x + u_yy + nonlin(u)

    # Infer B, B is fixed for now, may update for intermediate testing
    # B = 2.0 * nn.sigmoid(params_flat[-3])

    # HELENA RHS
    g = A_helena * (ffprime_const + fixed_B*(1 + epsilon * x)**2 * pprime_const)
    #g = A_helena * (Gamma + fixed_B * x * (1.0 + x * epsilon / 2.0) * Pi)
    #g = 2 / epsilon + 2 * x *(1.0 + x*epsilon / 2.0)
    
    #verifying matrix dimensions
    print("pde shape:", pde.shape)
    print("g shape:", g.shape)
    print("psi_bc shape:", psi_bc.shape)
    print('u shape:', u.shape)
    print('x shape', x.shape)

    # Regularization parameters (unchanged)
    lmbda = 10 ** (nn.sigmoid(params_flat[-2]) * 8 - 6)
    lamb = 10 ** (nn.sigmoid(params_flat[-1]) * 16 - 14)

    A_mat = jnp.vstack([pde * lmbda, u_bc]) #u[i_bc]])
    b_vec = jnp.vstack([g * lmbda, psi_bc])#labels[i_bc]])

    As = lamb * jnp.eye(A_mat.shape[1]) + (A_mat.T @ A_mat)
    bs = A_mat.T @ b_vec
    w = solve(As, bs)

    u_pred = u @ w
    mse = jnp.mean((labels - u_pred) ** 2)
    ssr = jnp.sum((b_vec - A_mat @ w)**2)
    rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
    # Additional metrics for logging.
    pde_loss = jnp.mean((pde * lmbda - g * lmbda)**2)
    bc_loss = jnp.mean((u_bc - psi_bc)**2)
    
    loss = mse

    
    return loss, (ssr, mse, rl2, pde_loss, bc_loss, lamb, lmbda)#return mse, (B,)
# def eval_loss(params_flat, task):
#     inputs = data_train_all[task]
#     labels = label_train_all[task]
#     g = source_term_all[task]
    
#     pred = model.apply(format_params_fn(params_flat), inputs)
#     # Split predictions into u and its derivatives.
#     # u, u_R, u_RR, u_ZZ = jnp.split(pred, 4, axis=1)
#     # R = inputs[:, 0:1]
#     # pde = u_RR - (1 / R) * u_R + u_ZZ
#     u, u_x, u_xx, u_yy = jnp.split(pred, 4, axis=1)
#     x = inputs[:, 0:1]

#     pde = (u_xx - (epsilon / (1.0 + epsilon * x)) * u_x + u_yy)
    
    
#     # Transform extra parameters.
#     lmbda = 10 ** (nn.sigmoid(params_flat[-2]) * 8 - 6)
#     lamb = 10 ** (nn.sigmoid(params_flat[-1]) * 16 - 14)
    
#     A = jnp.vstack([pde * lmbda, u[i_bc]])
#     b = jnp.vstack([g * lmbda, labels[i_bc]])
    
#     # Regularized pseudo-inverse
#     As = lamb * jnp.eye(A.shape[1]) + (A.T @ A)
#     bs = A.T @ b
#     w = solve(As, bs)
    
#     u_pred = u @ w
#     mse = jnp.mean((labels - u_pred)**2)
#     ssr = jnp.sum((b - A @ w)**2)
#     rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
#     # Additional metrics for logging.
#     pde_loss = jnp.mean((pde * lmbda - g * lmbda)**2)
#     bc_loss = jnp.mean((u[i_bc] - labels[i_bc])**2)
    
#     loss = mse  
#     return loss, (ssr, mse, rl2, pde_loss, bc_loss, lamb, lmbda)




loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

# ---------------------
# Mini-batch update function over meta-training tasks
# ---------------------
@jax.jit
def update(params_flat, opt_state, key):
    # r_task = min(8, config.n_meta_train)
    r_task = config.n_meta_train
    train_tasks = jax.random.choice(key, jnp.arange(n_task), (r_task,), replace=False)
    losses = 0.0
    ssrs = 0.0
    mses = 0.0
    rl2s = 0.0
    pde_losses = 0.0
    bc_losses = 0.0
    lambs = 0.0
    lmbdas = 0.0
    grads = 0.0
    for task in train_tasks:
        (loss, (ssr, mse, rl2, pde_loss, bc_loss, lamb_val, lmbda_val)), grad = loss_grad(params_flat, task)
        grads += grad
        losses += loss
        ssrs += ssr
        mses += mse
        rl2s += rl2
        pde_losses += pde_loss
        bc_losses += bc_loss
        lambs += lamb_val
        lmbdas += lmbda_val
    losses /= r_task
    ssrs /= r_task
    mses /= r_task
    rl2s /= r_task
    pde_losses /= r_task
    bc_losses /= r_task
    lambs /= r_task
    lmbdas /= r_task
    grads /= r_task
    updates, opt_state = optimizer.update(grads, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)
    return params_flat, opt_state, losses, ssrs, mses, rl2s, pde_losses, bc_losses, lambs, lmbdas

# ---------------------
# Training loop for meta-learning
# ---------------------
training_iters = []
training_mse = []
training_pde_loss = []

runtime = 0
train_iters = 0
store = []
start_time = time.time()

while (train_iters <= config.max_iters) and (runtime < 9000):
    print(f"iter: {train_iters}")
    key, subkey = jax.random.split(key)
    (params_flat, opt_state, loss, ssr, mse, rl2, pde_loss, bc_loss, lamb_val, lmbda_val) = update(params_flat, opt_state, subkey)
    train_iters += 1
    runtime = time.time() - start_time
    store.append([train_iters, runtime, loss, ssr, mse, rl2, pde_loss, bc_loss, lamb_val, lmbda_val])
    
    training_iters.append(train_iters)
    training_mse.append(float(mse))
    training_pde_loss.append(float(pde_loss))
    
    # if train_iters % 50 == 0:
    #     print(f'iter. = {train_iters:05d}, time = {int(runtime):03d}s, loss = {loss:.2e} | ssr = {ssr:.2e}, mse = {mse:.2e}, rl2 = {rl2:.2e}')
    
    print(f'iter. = {train_iters:05d}, time = {int(runtime):03d}s, loss = {loss:.2e} | ssr = {ssr:.2e}, mse = {mse:.2e}, rl2 = {rl2:.2e}')
    
    
store = np.array(store)
os.makedirs(f"../results/{trial}", exist_ok=True)
np.savetxt(f"../results/{trial}/meta_train_metrics.txt", store)

plt.figure(figsize=(10, 5))
plt.plot(training_iters, training_mse, label='MSE')
plt.plot(training_iters, training_pde_loss, label='PDE Loss')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.yscale('log')  # <- This sets the y-axis to log scale
plt.title('Training Losses: MSE and PDE Loss')
plt.legend()
plt.savefig(f"../results/{trial}/training_losses.png")
# plt.show()

# Save model
os.makedirs(f"../model/{trial}", exist_ok=True)
meta_model_file = f"../model/{trial}/meta_model.flax"
utils.save_meta_model(params_flat, meta_model_file)


# ---------------------
# Evaluation: For each task, use a pseudo-inverse to solve the linear system,
# save metrics to CSV, and produce plots.
# ---------------------
meta_model_file = f"../model/{trial}/meta_model.flax"
target = jnp.zeros_like(params_flat)
params_flat = utils.load_meta_model(meta_model_file, target)
eval_results = []  
for task in range(n_task):
    result = utils.evaluate_task(
        task, data_train_all, label_train_all, source_term_all, i_bc,
        model, params_flat, format_params_fn, config
    )
    eval_results.append(result)
    utils.plot_evaluation(
        task, data_train_all, label_train_all, result, config,
        save_path=f"../results/{trial}/task_{task:02d}_plot.png",
    )

df = pd.DataFrame(eval_results)
df.to_csv(f"../results/{trial}/task_metrics.csv", index=False)

# print(f"Final regularization parameters: lamb = {result['lamb']:.2e}, lmbda = {result['lmbda']:.2e}")


# ---------------------
# Inverse Training: Infer P/ delta for each task
# ---------------------
print("Starting inverse training...")

target = jnp.zeros_like(params_flat)
params_flat = utils.load_meta_model(meta_model_file, target)
print("Meta model loaded for inverse training.")

inverse_lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=config.inverse_max_lr,
    peak_value=config.inverse_max_lr,
    warmup_steps=int(config.inverse_max_iters * 0.4),
    decay_steps=config.inverse_max_iters,
    end_value=1e-6
)
inverse_optimizer = optax.adam(learning_rate=inverse_lr_scheduler)

inverse_results = [] 

for task in range(n_task):
    print(f"\nInverse training for task {task}...")
    # Initialize inverse parameters randomly (shape (2,))
    key, subkey = jax.random.split(key)
    params_inv = jax.random.normal(subkey, (1,))
    opt_state_inv = inverse_optimizer.init(params_inv)

    fixed_params = {
        'inputs': jnp.array(data_train_all[task]), 
        'forward': format_params_fn(params_flat),     
        'params_flat': params_flat                     
    }
    
    # Inverse training loop.
    inv_train_iters = 0
    inv_training_iters = []
    inv_training_loss = []
    log = []
    while inv_train_iters < config.inverse_max_iters:
        params_inv, opt_state_inv, inv_loss, aux = utils.update_inverse(
            params_inv, opt_state_inv, task, model, fixed_params, label_train_all, i_bc, inverse_optimizer
        )
        inv_training_iters.append(inv_train_iters)
        inv_training_loss.append(inv_loss)
        if inv_train_iters % 50 == 0:
            ssr, mse_val, rl2_val = aux
            print(f"Task {task:02d} iter {inv_train_iters:03d}: inverse loss = {inv_loss:.2e}, mse = {mse_val:.2e}")
        
        # ---------------------
        # # Training for P
        # # ---------------------
        # P_infer = float(0.8 + 0.2 * nn.sigmoid(params_inv[0]))
        # Infer B from params
        B_infer = 2.0 * nn.sigmoid(params_inv[0])



        # ---------------------
        # Training for delta
        # ---------------------
        # delta_infer = float(0.2 + 0.5 * nn.sigmoid(params_inv[0]))  ### CHANGE

        log.append([inv_train_iters, float(inv_loss), float(ssr), float(mse_val), float(B_infer)])#float(P_infer)])
        inv_train_iters += 1
        
    # After training, store the final inferred values.
    
    # ---------------------
    # Training for delta
    # ---------------------
    # P_infer_final = float(config.P_vals)
    # delta_infer_final = delta_infer
        
    # ---------------------
    # Training for P
    # ---------------------
    B_infer_final = B_infer
    delta_infer_final = float(config.delta_vals)
    
    result = {
        'task': task,
        # ---------------------
        # Training for P
        # ---------------------
        'true_B': float(config.B_vals),
        'inferred_B': B_infer_final,
        # 'true_P': float(config.P_vals[task]),
        # 'inferred_P': P_infer_final,

        # ---------------------
        # Training for delta
        # ---------------------
        # 'true_delta': float(config.delta_vals[task]),
        # 'inferred_delta': delta_infer_final,

        'inverse_loss': float(inv_loss)
    }
    
    inverse_results.append(result)
    
    # inverse_results.append({
    #     'task': task,
    #     # ---------------------
    #     # Training for P
    #     # ---------------------
    #     'true_B': float(config.B_vals),
    #     'inferred_B': B_infer_final,
    #     # 'true_P': float(config.P_vals[task]),
    #     # 'inferred_P': P_infer_final,

    #     # ---------------------
    #     # Training for delta
    #     # ---------------------
    #     # 'true_delta': float(config.delta_vals[task]),
    #     # 'inferred_delta': delta_infer_final,

    #     'inverse_loss': float(inv_loss)
    # })
    # ---------------------
    # Training for delta
    # ---------------------
    # print(f"Task {task:02d}: True delta = {config.delta_vals[task]:.6f}, Inferred delta = {delta_infer_final:.6f}, Loss = {inv_loss:.2e}")

    # ---------------------
    # Training for P or B
    # ---------------------
    print(f"Task {task:02d}: True B = {config.B_vals:.6f}, Inferred B = {B_infer_final:.6f}, Loss = {inv_loss:.2e}")
    #print(f"Task {task:02d}: True P = {config.P_vals[task]:.6f}, Inferred P = {P_infer_final:.6f}, Loss = {inv_loss:.2e}")
    
    # generate log-scale plot of inverse loss against epochs 
    plt.figure(figsize=(10, 5))
    plt.plot(inv_training_iters, inv_training_loss)
    m, b = np.polyfit(inv_training_iters, np.log(inv_training_loss), 1)  # Optional: fit in log-space
    print((m, b))
    plt.xlabel('Training Iteration')
    plt.ylabel('Inverse Loss')
    plt.yscale('log')  # <- This sets the y-axis to log scale
    plt.title('Inverse Training Losses (Log Scale)')
    plt.savefig(f"../results/{trial}/task_{task:02d}_training_losses_logscale.png")
    
    data_infer, labels_infer, _, _ = utils.generate_data(
        eq,
        config.epsilon,
        config.kappa_vals,
        config.delta_vals,
        B_infer_final,
        config.n_train_x,
        config.n_train_y
    )
    
    
    #inputs = np.array(data_infer[task])
    inputs = np.array(data_train_all[task])
    true_labels = np.array(label_train_all[task])
    infer_labels = np.array(labels_infer)
    
    utils.plot_inverse_comparison(
        task,
        inputs,
        true_labels,
        infer_labels,
        result,
        config,
        trial
    )
    
    # data_infer, labels_infer, _, _ = utils.generate_data(
    #     config.epsilon,
    #     fixed_kappa,
    #     fixed_delta,
    #     B_infer_final,
    #     config.n_train_x,
    #     config.n_train_y
    # )
    # inputs = np.array(data_train_all[task])
    # true_labels = np.array(label_train_all[task])
    # infer_labels = np.array(labels_infer)

    # utils.plot_inverse_comparison(
    #     task,
    #     inputs,
    #     true_labels,
    #     infer_labels,
    #     result,
    #     config,
    #     trial
    # )
    
    
    log = np.array(log)
    np.savetxt(f"../results/{trial}/inverse_task_{task:02d}_log.txt", log)


df_inv = pd.DataFrame(inverse_results)
print("Inverse training results:")
print(df_inv)
df_inv.to_csv(f"../results/{trial}/inverse_results.csv", index=False)