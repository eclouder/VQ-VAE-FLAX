from flax import nnx
from residual import ResidualStack
import jax

import jax.numpy as jnp


class Decoder(nnx.Module):
    def __init__(self, in_dim, h_dim, n_res_layer, res_h_dim, conv_kernel: int = 4, conv_stride: int = 2):
        super().__init__()
        rngs = nnx.Rngs(0)

        # Adjust in_dim to match the number of input channels (3 for RGB images)
        self.conv_1 = nnx.ConvTranspose(in_dim, h_dim, kernel_size=(conv_kernel - 1, conv_kernel - 1),
                                        strides=(conv_stride - 1, conv_stride - 1), padding=1, rngs=rngs)
        self.conv_2 = nnx.ConvTranspose(h_dim, h_dim // 2, kernel_size=conv_kernel, strides=conv_stride, padding=1,
                                        rngs=rngs)
        self.conv_3 = nnx.ConvTranspose(h_dim // 2, 3, kernel_size=conv_kernel, strides=conv_stride, padding=1,
                                        rngs=rngs)
        self.residual_layer = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layer)

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.residual_layer(x)
        x = self.conv_2(x)
        x = nnx.relu(x)
        x = self.conv_3(x)
        return x
