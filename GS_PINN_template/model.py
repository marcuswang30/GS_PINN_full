"""
Model definition for the Physics-Informed Neural Network (PINN).

The model uses a simple dense layer followed by a sinusoidal activation.
It also computes the derivatives required to form the PDE residual.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jacfwd, vmap

class PINN(nn.Module):
    n_nodes: int  
    
    def setup(self):
        self.layers = [
            nn.Dense(self.n_nodes, kernel_init=jax.nn.initializers.he_uniform()),
            jnp.sin,  # Sinusoidal activation
            # Uncomment below to add more layers if needed
            # nn.Dense(self.n_nodes, kernel_init=jax.nn.initializers.he_uniform()),
            # jnp.sin,
        ]
        
    @nn.compact
    def __call__(self, inputs):
        """
        Forward pass: Computes the neural network output and its derivatives.
        
        Parameters:
            inputs: jnp.ndarray of shape [batch_size, 2] with [R, Z] coordinates.
            
        Returns:
            A concatenated array of u, u_R, u_RR, u_ZZ.
        """
        R, Z = inputs[:, 0:1], inputs[:, 1:2]

        def get_u(R, Z):
            u = jnp.hstack([R, Z])
            ul = []
            for i, layer in enumerate(self.layers):
                u = layer(u)
                if i == 0:  # Scale output of the first layer
                    u = jnp.pi * u
                if i % 2 != 0:
                    ul.append(u)
            return u

        u = get_u(R, Z)

        def get_u_dir(get_u, R, Z):
            u_R = jacfwd(get_u)(R, Z)
            u_RR = jacfwd(jacfwd(get_u))(R, Z)
            u_ZZ = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(R, Z)
            return u_R, u_RR, u_ZZ

        # Vectorize derivative computations over the batch
        u_R, u_RR, u_ZZ = vmap(get_u_dir, in_axes=(None, 0, 0))(get_u, R, Z)
        u_R, u_RR, u_ZZ = u_R[:, :, 0], u_RR[:, :, 0, 0], u_ZZ[:, :, 0, 0]
        outputs = jnp.hstack([u, u_R, u_RR, u_ZZ])
        return outputs