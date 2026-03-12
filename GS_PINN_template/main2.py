# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:30:51 2025

@author: marcu
"""

"""
Main script for meta-training the raw PINN for the Gradâ€“Shafranov equation
by varying different parameters, while keeping the rest fixed.
Then, the trained model is evaluated on all tasks:
- The linear system is solved using regularised pseudo-inverse.
- Metrics (MSE, SSR, relative L2 error) for each task are saved to a CSV.
- For each task, a 2x2 plot is generated showing the analytical solution,
  predicted solution, MSE contour, and an MSE slice at Z=0.
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
import utils
import config

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
P_vals = config.P_vals
task_params = P_vals.reshape(-1, 1)  # Shape: (n_task, 1)
# ---------------------
# Training for delta
# ---------------------
# delta_vals = config.delta_vals
# fixed_P = config.P_vals         
# task_params = delta_vals.reshape(-1, 1)  # Shape: (n_task, 1)

fixed_kappa = config.kappa_vals    
n_task = len(task_params)
print("Number of tasks:", n_task)
config.n_train_tasks = n_task

# ---------------------
# Generate training data for each task
# ---------------------
# Change delta & P according to your task
data_train_list, label_train_list, source_term_list = [], [], []
for P_val in task_params:
    data_train, labels, g, i_bc = utils.generate_data(
        config.epsilon, fixed_kappa, fixed_delta, P_val.item(),
        config.n_train_radial, config.n_train_angular
    )
    data_train_list.append(data_train)
    label_train_list.append(labels)
    source_term_list.append(g)

data_train_all = jnp.array(data_train_list)
label_train_all = jnp.array(label_train_list)
source_term_all = jnp.array(source_term_list)

# ---------------------
# Define boundary condition indicator (assumed same for all tasks)
# ---------------------
radial = jnp.linspace(0, config.epsilon, config.n_train_radial)
angular = jnp.linspace(0, 2*jnp.pi, config.n_train_angular)
radial_mesh, _ = jnp.meshgrid(radial, angular, indexing="xy")
i_bc = (radial_mesh.ravel() == config.epsilon)         
print(i_bc.shape, jnp.sum(i_bc))

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
    g = source_term_all[task]
    
    pred = model.apply(format_params_fn(params_flat), inputs)
    # Split predictions into u and its derivatives.
    u, u_R, u_RR, u_ZZ = jnp.split(pred, 4, axis=1)
    R = inputs[:, 0:1]
    pde = u_RR - (1 / R) * u_R + u_ZZ
    
    # Transform extra parameters.
    lmbda = 10 ** (nn.sigmoid(params_flat[-2]) * 8 - 6)
    lamb = 10 ** (nn.sigmoid(params_flat[-1]) * 16 - 14)
    
    A = jnp.vstack([pde * lmbda, u[i_bc]])
    b = jnp.vstack([g * lmbda, labels[i_bc]])
    
    # Regularized pseudo-inverse
    As = lamb * jnp.eye(A.shape[1]) + (A.T @ A)
    bs = A.T @ b
    w = solve(As, bs)
    
    u_pred = u @ w
    mse = jnp.mean((labels - u_pred)**2)
    ssr = jnp.sum((b - A @ w)**2)
    rl2 = jnp.linalg.norm(labels - u_pred) / jnp.linalg.norm(labels)
    
    # Additional metrics for logging.
    pde_loss = jnp.mean((pde * lmbda - g * lmbda)**2)
    bc_loss = jnp.mean((u[i_bc] - labels[i_bc])**2)
    
    loss = mse  
    return loss, (ssr, mse, rl2, pde_loss, bc_loss, lamb, lmbda)

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
    
    if train_iters % 50 == 0:
        print(f'iter. = {train_iters:05d}, time = {int(runtime):03d}s, loss = {loss:.2e} | ssr = {ssr:.2e}, mse = {mse:.2e}, rl2 = {rl2:.2e}')

store = np.array(store)
os.makedirs(f"../results/{trial}", exist_ok=True)
np.savetxt(f"../results/{trial}/meta_train_metrics.txt", store)

plt.figure(figsize=(10, 5))
plt.plot(training_iters, training_mse, label='MSE')
plt.plot(training_iters, training_pde_loss, label='PDE Loss')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
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
    log = []
    while inv_train_iters < config.inverse_max_iters:
        params_inv, opt_state_inv, inv_loss, aux = utils.update_inverse(
            params_inv, opt_state_inv, task, model, fixed_params, i_bc, inverse_optimizer
        )
        if inv_train_iters % 50 == 0:
            ssr, mse_val, rl2_val = aux
            print(f"Task {task:02d} iter {inv_train_iters:03d}: inverse loss = {inv_loss:.2e}, mse = {mse_val:.2e}")
        
        # ---------------------
        # Training for P
        # ---------------------
        P_infer = float(0.8 + 0.2 * nn.sigmoid(params_inv[0]))

        # ---------------------
        # Training for delta
        # ---------------------
        # delta_infer = float(0.2 + 0.5 * nn.sigmoid(params_inv[0]))  ### CHANGE

        log.append([inv_train_iters, float(inv_loss), float(ssr), float(mse_val), float(P_infer)])
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
    P_infer_final = P_infer
    delta_infer_final = float(config.delta_vals)

    inverse_results.append({
        'task': task,
        # ---------------------
        # Training for P
        # ---------------------
        'true_P': float(config.P_vals[task]),
        'inferred_P': P_infer_final,

        # ---------------------
        # Training for delta
        # ---------------------
        # 'true_delta': float(config.delta_vals[task]),
        # 'inferred_delta': delta_infer_final,

        'inverse_loss': float(inv_loss)
    })
    # ---------------------
    # Training for delta
    # ---------------------
    # print(f"Task {task:02d}: True delta = {config.delta_vals[task]:.6f}, Inferred delta = {delta_infer_final:.6f}, Loss = {inv_loss:.2e}")

    # ---------------------
    # Training for P
    # ---------------------
    print(f"Task {task:02d}: True P = {config.P_vals[task]:.6f}, Inferred P = {P_infer_final:.6f}, Loss = {inv_loss:.2e}")

    log = np.array(log)
    np.savetxt(f"../results/{trial}/inverse_task_{task:02d}_log.txt", log)


df_inv = pd.DataFrame(inverse_results)
print("Inverse training results:")
print(df_inv)
df_inv.to_csv(f"../results/{trial}/inverse_results.csv", index=False)