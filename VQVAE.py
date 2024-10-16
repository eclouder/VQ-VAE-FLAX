from flax import nnx
from encode import Encoder
from typing import Optional
import jax
from VQ import VQ
from decode import Decoder
class VQVAE(nnx.Module):
    def __init__(self,h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim,  rngs:Optional[nnx.Rngs]=nnx.Rngs(0)):
        super().__init__()
        self.Encoder = Encoder(3,h_dim,n_res_layers,res_h_dim)
        self.conv2embed_dim = nnx.Conv(h_dim,embedding_dim,1,1,rngs=rngs)
        self.VQ = VQ(h_dim, res_h_dim)
        self.Decoder = Decoder(embedding_dim, h_dim,n_res_layers,res_h_dim)
    def __call__(self,x):
        z_e = self.Encoder(x)
        z_e = self.conv2embed_dim(z_e)
        embed_loss,z_q,perp,_ =self.VQ(z_e)
        x_hat = self.Decoder(z_q)
        return embed_loss,x_hat,perp