import jax.numpy as jnp
from flax import nnx
import jax
class ResidualLayer(nnx.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """
    in_dim: int
    h_dim: int
    res_h_dim: int

    def __init__(self, in_dim, h_dim, res_h_dim):
        rngs = nnx.Rngs(0)

        self.conv1 = nnx.Conv(in_dim,res_h_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,rngs=rngs)
        self.conv2 = nnx.Conv(res_h_dim,h_dim, kernel_size=(1, 1), strides=(1, 1), use_bias=False,rngs=rngs)

    def __call__(self, x):
        residual = x
        x = jax.nn.relu(x)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        return x + residual


class ResidualStack(nnx.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    in_dim: int
    h_dim: int
    res_h_dim: int
    n_res_layers: int

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        self.layers = [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        x = jax.nn.relu(x)
        return x