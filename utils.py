import jax
import jax.numpy as jnp
def random_jnp_tensor(shape:tuple):
    key = jax.random.PRNGKey(0)

    # Input tensor has shape (32, 3, 256, 256)
    random_tensor = jax.random.normal(key, shape)
    return random_tensor