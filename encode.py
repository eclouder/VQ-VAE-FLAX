from flax import nnx
from residual import ResidualStack
import jax

import jax
import jax.numpy as jnp


class Encoder(nnx.Module):
    def __init__(self, in_dim, h_dim, n_res_layer, res_h_dim, conv_kernel: int = 4, conv_stride: int = 2):
        super().__init__()
        rngs = nnx.Rngs(0)

        # Adjust in_dim to match the number of input channels (3 for RGB images)
        self.conv_1 = nnx.Conv(in_dim, h_dim // 2, kernel_size=(conv_kernel, conv_kernel), strides=(conv_stride, conv_stride), padding=1, rngs=rngs)
        self.conv_2 = nnx.Conv(h_dim // 2, h_dim, kernel_size=conv_kernel, strides=conv_stride, padding=1, rngs=rngs)
        self.conv_3 = nnx.Conv(h_dim, h_dim, kernel_size=conv_kernel - 1, strides=conv_stride - 1, padding=1, rngs=rngs)
        self.residual_layer = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layer)

    def __call__(self, x):
        x = self.conv_1(x)
        x = nnx.relu(x)
        x = self.conv_2(x)
        x = nnx.relu(x)
        x = self.conv_3(x)
        x = self.residual_layer(x)
        return x

if __name__ == '__main__':
    # Set in_dim to 3 to match the input tensor
    test_model = Encoder(3, 128, 3, 64)
    key = jax.random.PRNGKey(0)

    # Input tensor has shape (32, 3, 256, 256)
    random_tensor = jax.random.normal(key, (32, 256, 256, 3))
    y = test_model(random_tensor)

    print(y.shape)