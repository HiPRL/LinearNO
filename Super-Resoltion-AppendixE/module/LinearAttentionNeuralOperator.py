import numpy as np
from timm.layers.weight_init import trunc_normal_
# from model.Embedding import timestep_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class LinearNO(nn.Module):

    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,key_ratio=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim_head, key_ratio*dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, key_ratio*dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.GELU(),
                                    nn.Linear(dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x_mid = self.in_project_x(x)
        x_mid = rearrange(x_mid, 'b n (h d) -> b h n d',h=self.heads, d=self.dim_head)
        q=self.to_q(x_mid)
        k=self.to_k(x_mid)
        v=self.to_v(x_mid)

        q=F.softmax(q,dim=-1)
        k=F.softmax(k,dim=-2)

        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        qkv = torch.einsum("bhnd,bhdc->bhnc", q, kv)
        qkv=rearrange(qkv, 'b h n d -> b n (h d)')
        return self.to_out(qkv)
class CrossLinearNO(nn.Module):

    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,key_ratio=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_y = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim_head, key_ratio*dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, key_ratio*dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.GELU(),
                                    nn.Linear(dim, dim), nn.Dropout(dropout))
    def forward(self, x,y):
        # B N C
        B, N, C = x.shape
        x_mid = self.in_project_x(x)
        y_mid = self.in_project_y(y)
        x_mid = rearrange(x_mid, 'b n (h d) -> b h n d',h=self.heads, d=self.dim_head)
        y_mid = rearrange(y_mid, 'b n (h d) -> b h n d',h=self.heads, d=self.dim_head)
        q=self.to_q(x_mid)
        k=self.to_k(y_mid)
        v=self.to_v(y_mid)

        q=F.softmax(q,dim=-1)
        k=F.softmax(k,dim=-2)

        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        qkv = torch.einsum("bhnd,bhdc->bhnc", q, kv)
        qkv=rearrange(qkv, 'b h n d -> b n (h d)')
        return self.to_out(qkv)

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

# Linear Neural Operator
class LinearNO_block(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=4,
        key_ratio=4,
        last_layer=False,
        out_dim=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = LinearNO(hidden_dim,
                                                      heads=num_heads,
                                                      dim_head=hidden_dim //
                                                      num_heads,
                                                      dropout=dropout,
                                                      key_ratio=key_ratio)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim,
                       hidden_dim * mlp_ratio,
                       hidden_dim,
                       n_layers=0,
                       res=False,
                       act=act)



    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx
class CrossLinearNO_block(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=4,
        key_ratio=4,
        last_layer=False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.Attn=CrossLinearNO(hidden_dim,
                                                      heads=num_heads,
                                                      dim_head=hidden_dim //
                                                      num_heads,
                                                      dropout=dropout,
                                                      key_ratio=key_ratio)
        self.mlp = MLP(hidden_dim,
                       hidden_dim * mlp_ratio,
                       hidden_dim,
                       n_layers=0,
                       res=False,
                       act=act)
    def forward(self, query,y):
        fx = self.Attn(self.ln_1(query),self.ln_2(y)) + query
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class LinearAttentionNeuralOperator(nn.Module):

    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act='gelu',
        mlp_layer=1,
        mlp_ratio=3,
        fun_dim=1,
        out_dim=1,
        key_ratio=4,
        ref=8,
    ):
        super(LinearAttentionNeuralOperator, self).__init__()
        self.ref = ref
        #funcution projection
        self.preprocess = MLP(fun_dim,
                              n_hidden * 2,
                              n_hidden,
                              n_layers=0,
                              res=False,
                              act=act)
        #coordinate projection
        self.query_MLP = MLP(space_dim,
                             n_hidden * 2,
                             n_hidden,
                             n_layers=0,
                             res=False,
                             act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            LinearNO_block(num_heads=n_head,
                                       hidden_dim=n_hidden,
                                       dropout=dropout,
                                       act=act,
                                       mlp_ratio=mlp_ratio,
                                       out_dim=out_dim,
                                       key_ratio=key_ratio,
                                       last_layer=(_ == n_layers - 1))
            for _ in range(n_layers)
        ])
        self.cross_block = CrossLinearNO_block(
            num_heads=n_head,
            hidden_dim=n_hidden,
            dropout=dropout,
            act=act,
            mlp_ratio=mlp_ratio,
            key_ratio=key_ratio,
        )
        self.ln = nn.LayerNorm(n_hidden)
        self.mlp2 = nn.Linear(n_hidden, out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, query, x):
        # function projection + coordinate projection
        fx = self.preprocess(x[..., 2:]) + self.query_MLP(x[..., :2])

        for block in self.blocks:
            fx = block(fx)  # fx: B N C

        query = self.query_MLP(query)

        out = self.cross_block(query, fx)

        out = self.mlp2(self.ln(out))

        return out
