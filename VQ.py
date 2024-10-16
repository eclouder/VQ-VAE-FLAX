import jax.lax
from flax import nnx
from typing import Optional
from jax import numpy as jnp
from torchmetrics.functional.text import perplexity

from utils import random_jnp_tensor
class VQ(nnx.Module):
    def __init__(self,embedding_num:int,embedding_dim:int,beta,rngs: Optional[nnx.Rngs]=nnx.Rngs(0)):
        super().__init__()
        # self.embedding_num = embedding_num
        self.beta = beta
        self.embedding_dim = embedding_dim
        self.embedding = nnx.Embed(embedding_num,embedding_dim,rngs= rngs)
    def __call__(self, x):
        z_fla = jnp.reshape(x,(-1,self.embedding_dim))
        dis = jnp.sum(z_fla ** 2, axis=1, keepdims=True) +\
            jnp.sum(self.embedding.embedding.value **2, axis=1) -\
            2 * jnp.matmul(z_fla,self.embedding.embedding.value.T)
        min_encoding_indices = jnp.argmin(dis, axis=1) # shape:[(batch,w,h),embed_num]
        z_q = self.embedding(min_encoding_indices)
        z_q = jnp.reshape(z_q,(x.shape))
        loss = jnp.mean((jax.lax.stop_gradient(z_q) - x) ** 2 ) + \
            self.beta * jnp.mean((z_q -jax.lax.stop_gradient(x)) ** 2)
        z_q = x + jax.lax.stop_gradient(z_q - x)
        e_min = jnp.mean(min_encoding_indices,axis=0)
        _perplexity = jnp.exp(-jnp.sum(e_min * jnp.log(e_min + 1e-10)))

        return loss,z_q,_perplexity,min_encoding_indices

if __name__ == '__main__':
    t = random_jnp_tensor((32,256,256,256))
    Vq = VQ(128,256)
    Vq(t)