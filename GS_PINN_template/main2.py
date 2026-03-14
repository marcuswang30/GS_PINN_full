import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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
trial = 'iPINN_08'
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

# ---------------------
# Build task parameters: Vary both delta and P
# ---------------------
# Use all delta values from config.delta_vals and generate a range for P.
# (Here we generate a 1D array for P. Adjust the number of P samples as needed.)
delta_vals = config.delta_vals                   # shape: (n_delta,)
P_vals = config.P_vals           # pressure profile parameter range (excluding zero)

# Create a grid (Cartesian product) of [delta, P] values.
delta_grid, P_grid = jnp.meshgrid(delta_vals, P_vals)
task_params = jnp.stack([delta_grid.ravel(), P_grid.ravel()], axis=-1)  # each row is [delta, P]
n_task = task_params.shape[0]
print("Number of tasks:", n_task)
config.n_train_tasks = n_task

print("delta_vals shape:", delta_vals.shape)
print("P_vals shape:", P_vals.shape)
print("Task parameters shape:", task_params.shape)
print("Sample task parameters (first 5):", task_params[:5])
# ---------------------
# Generate training data for each task
# ---------------------
data_train_list, label_train_list, source_term_list = [], [], []
for i in range(n_task):
    delta_val = task_params[i, 0]
    P_val = task_params[i, 1]
    data_train, labels, g, i_bc = utils.generate_data(
        config.epsilon, config.kappa_vals, delta_val.item(), P_val.item(),
        config.n_train_radial, config.n_train_angular
    )
    data_train_list.append(data_train)
    label_train_list.append(labels)
    source_term_list.append(g)

data_train_all = jnp.array(data_train_list)
label_train_all = jnp.array(label_train_list)
source_term_all = jnp.array(source_term_list)

print("data_train_all shape:", data_train_all.shape)
print("label_train_all shape:", label_train_all.shape)
print("source_term_all shape:", source_term_all.shape)
# ---------------------
# Define boundary condition indicator (assumed same for all tasks)
# ---------------------
_, angular_mesh = jnp.meshgrid(
    jnp.linspace(0, config.epsilon, config.n_train_radial),
    jnp.linspace(0, 2 * jnp.pi, config.n_train_angular)
)
i_bc = jnp.where(angular_mesh.ravel() == config.epsilon, 1, 0)

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

# Create a parameter formatting function
def format_params_fn(p):
    base_params = p[:-2]
    return unravel_fn(base_params)

# Comment from here if you just want to run inverse
# ---------------------
# Set up the optimizer for meta-training
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
    # Get the training data for this task.
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
    
    # Use regularized pseudo-inverse (ridge solve).
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
    
    loss = mse  # the training loss is defined as MSE.
    return loss, (ssr, mse, rl2, pde_loss, bc_loss, lamb, lmbda)

loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

# ---------------------
# Vectorized mini-batch update function over meta-training tasks using jax.vmap
# ---------------------
@jax.jit
def update(params_flat, opt_state, key):
    r_task = config.n_meta_train
    # Sample a batch of task indices.
    tasks_batch = jax.random.choice(key, jnp.arange(n_task), shape=(r_task,), replace=False)
    
    # Define a function for loss and gradient per task.
    def loss_and_grad_fn(task):
        return loss_grad(params_flat, task)
    
    # Vectorize the computation over the batch.
    batched_out = jax.vmap(loss_and_grad_fn)(tasks_batch)
    # batched_out is a tuple: ( (losses, aux_metrics), grads )
    # Extract losses and aux metrics.
    losses_array = batched_out[0][0]  # shape: (r_task,)
    aux_metrics = batched_out[0][1]   # each is an array of shape (r_task,)
    # Average aux metrics.
    avg_aux = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), aux_metrics)
    avg_loss = jnp.mean(losses_array)
    
    # Average gradients over tasks.
    avg_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), batched_out[1])
    
    updates, opt_state = optimizer.update(avg_grad, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)
    return params_flat, opt_state, avg_loss, avg_aux

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

while (train_iters <= config.max_iters) and (runtime < 12000):
    print(f"iter: {train_iters}")
    key, subkey = jax.random.split(key)
    (params_flat, opt_state, loss, aux) = update(params_flat, opt_state, subkey)
    train_iters += 1
    runtime = time.time() - start_time
    # Unpack auxiliary metrics.
    ssr, mse, rl2, pde_loss, bc_loss, lamb_val, lmbda_val = aux
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

# ---------------------
# Evaluation: For each task, use a pseudo-inverse to solve the linear system,
# save metrics to CSV, and produce plots.
# ---------------------
eval_results = []  
for task in range(n_task):
    # Here we pass the true parameters for the task from task_params.
    result = utils.evaluate_task(
        task, data_train_all, label_train_all, source_term_all, i_bc,
        model, params_flat, format_params_fn, config
    )
    # Overwrite the task parameters in the result with the true ones.
    result['delta'] = float(task_params[task, 0])
    result['P'] = float(task_params[task, 1])
    
    print(f"Task {task}: True delta = {result['delta']:.4f}, True P = {result['P']:.4f}, MSE = {result['mse']:.4e}")
    eval_results.append(result)
    utils.plot_evaluation(
        task, data_train_all, label_train_all, result, config,
        save_path=f"../results/{trial}/task_{task:02d}_plot.png"
    )

df = pd.DataFrame(eval_results)
df.to_csv(f"../results/{trial}/task_metrics.csv", index=False)

print(f"Final regularization parameters: lamb = {result['lamb']:.2e}, lmbda = {result['lmbda']:.2e}")

# ---------------------
# Save the meta model parameters after meta training
# ---------------------
os.makedirs(f"../model/{trial}", exist_ok=True)
meta_model_file = f"../model/{trial}/meta_model.flax"
utils.save_meta_model(params_flat, meta_model_file)


# ---------------------
# Inverse Training: Infer P and delta for each task
# ---------------------
print("Starting inverse training...")

# Load the saved meta model parameters using Flax serialization.
target = jnp.zeros_like(params_flat)
params_flat = utils.load_meta_model(meta_model_file, target)
print("Meta model loaded for inverse training.")

# Set up inverse optimizer using inverse hyperparameters from config
inverse_lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=config.inverse_max_lr,
    peak_value=config.inverse_max_lr,
    warmup_steps=int(config.inverse_max_iters * 0.4),
    decay_steps=config.inverse_max_iters,
    end_value=1e-6
)
inverse_optimizer = optax.adam(learning_rate=inverse_lr_scheduler)

# For each task, run an inverse training loop.
inverse_results = []  # to store inferred parameters and losses

for task in range(n_task):
    print(f"\nInverse training for task {task}...")
    # Initialize inverse parameters randomly (shape (2,)): params_inv[0] for P and params_inv[1] for delta.
    key, subkey = jax.random.split(key)
    params_inv = jax.random.normal(subkey, (2,))
    opt_state_inv = inverse_optimizer.init(params_inv)
    
    # Prepare fixed_params dictionary for inverse loss.
    fixed_params = {
        'inputs': jnp.array(data_train_all[task]),  # training data inputs for this task
        'forward': format_params_fn(params_flat),     # fixed forward network parameters
        'params_flat': params_flat                     # original forward flat parameters (for lmbda and lamb)
    }
    
    # Inverse training loop.
    inv_train_iters = 0
    while inv_train_iters < config.inverse_max_iters:
        params_inv, opt_state_inv, inv_loss, aux = utils.update_inverse(
            params_inv, opt_state_inv, task, model, fixed_params, i_bc, inverse_optimizer
        )
        if inv_train_iters % 50 == 0:
            ssr, mse_val, rl2_val = aux
            print(f"Task {task:02d} iter {inv_train_iters:03d}: inverse loss = {inv_loss:.2e}, mse = {mse_val:.2e}")
        inv_train_iters += 1
        
    # After training, compute the final inferred values.
    P_infer_final = float(nn.sigmoid(params_inv[0]))          # Map raw value to (0,1)
    delta_infer_final = float(0.3 + 0.2 * nn.sigmoid(params_inv[1]))  ### CHANGE
    inverse_results.append({
        'task': task,
        'true_delta': float(task_params[task, 0]),
        'inferred_delta': delta_infer_final,
        'true_P': float(task_params[task, 1]),
        'inferred_P': P_infer_final,
        'inverse_loss': float(inv_loss)
    })

# Print inverse results:
df_inv = pd.DataFrame(inverse_results)
print("Inverse training results:")
print(df_inv)
df_inv.to_csv(f"../results/{trial}/inverse_results.csv", index=False)