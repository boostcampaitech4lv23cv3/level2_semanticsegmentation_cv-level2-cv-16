import copy
import math

import torch
import torch.nn.functional as F
from detectron2.layers import DepthwiseSeparableConv2d
from timm.models.layers import DropPath
from torch import Tensor, nn
from torch.functional import Tensor
from torch.nn import MultiheadAttention


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk -> ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if self.bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)
                
    def forward(self, x):
        # x: B, K, d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class MultiLabelHead(nn.Module):
    def __init__(self, num_classes, d_encoder, hidden_dim, n_heads, d_ff, dropout, share_embedding, downsample=None, mlp=True, droppath=0):
        super().__init__()
        self.share_embedding = share_embedding
        self.mlp = mlp
        self.block = Block(hidden_dim, n_heads, d_ff, dropout, droppath)
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_classes = num_classes
        self.fc = GroupWiseLinear(num_classes, hidden_dim)
        
        if not share_embedding:
            self.cls_emb = nn.Parameter(torch.randn(1, num_classes, hidden_dim))
            from torch.nn.init import trunc_normal_
            
            trunc_normal_(self.cls_emb, std=0.02)
        self.scale = hidden_dim ** -0.5
        
        self.proj_dec = nn.Linear(d_encoder, hidden_dim)
        self.downsample = downsample
        if downsample:
            self.pooling = nn.AdaptiveAvgPool2d(downsample)
            
    def forward(self, x):
        if self.share_embedding:
            x, cls_emb = x
            cls_emb = cls_emb.unsqueeze(0)
        else:
            cls_emb = self.cls_emb
            
        if self.downsample:
            x.self.pooling(x)
            
        B, C = x.size()[:2]
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.proj_dec(x)
        
        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        x = self.block(x)
        x = self.norm(x)
        cls_emb = x[:, -self.num_classes :]
        img_pred = self.fc(cls_emb)
        
        return img_pred
    

class FeedFoward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        
    @property
    def unwrapped(self):
        return self
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    @property
    def unwrapped(self):
        return self
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])
        
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
    
    
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedFoward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()
        
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        
        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return self
    
    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        B, HW, C = x.size()
        x = x.view(B, GS, HW // GS, C).permute(0, 3, 1, 2)
        
        return x

class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mizer and related networks"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU()):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x