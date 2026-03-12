# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 00:58:11 2026

@author: marcu
"""

"""
Configuration file for GS-PINN.

Contains all the hyperparameters and settings used in training.
"""

import jax.numpy as jnp
import os
import jax

# Physical and domain parameters
epsilon = 0.32          # Elongation parameter
n_train_radial = 10     # Number of radial grid points, 50
n_train_angular = 360   # Number of angular grid points, 720
n_train_x = 100
n_train_y = 100


# delta_vals = jnp.linspace(0.2, 0.7, 11)[1:]  # Triangularity parameter (excluding first point)
delta_vals = 0.7
# P_vals = 0.5        # Pressure profile parameter (excluding zero)
#P_vals = jnp.linspace(0.8, 1.0, 11)[1:]
kappa_vals = 1.0 #1.7
R0 = 1 #giving it an arbitrary value for now 
a = R0 * epsilon #see above with R0
B_vals = 1
A_helena = 1.77 #1.99 (ff'=-1)

# Training hyperparameters
n_train_tasks = None    # Will be computed based on the grid of task parameters
n_meta_train = 1        # Number of tasks for meta-training
max_iters = 300  #100       # Maximum iterations for meta-learning training loop
max_lr = 1e-1           # Maximum learning rate for the scheduler

# Trial number
trial = 'iPINN_11_helena' ### CHANGE

# Random seed for reproducibility
seed = 9

# GPU configuration (if applicable)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# Inverse problem training hyperparameters
inverse_max_iters = 200 #200        # Maximum iterations for inverse training loop
inverse_max_lr = 5e-2          # Maximum learning rate for the inverse optimizer