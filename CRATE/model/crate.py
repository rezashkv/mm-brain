import math
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
from .baseline_modules import MLP, ResNet, NODE, SNN
from .node import entmax15, entmoid15


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False, return_key=False):
        if return_key:
            return self.qkv(x)
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        if return_attention:
            return attn
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x

            x = ff(grad_x)
        return x


class InterAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, seq_length=1339, dropout=0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista)),
                        PreNorm(dim * seq_length, Attention(dim * seq_length, heads=heads, dim_head=dim_head,
                                                            dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )

    def forward(self, x):
        for attn1, ff1, attn2, ff2 in self.layers:
            grad_x = attn1(x) + x
            grad_x = ff1(grad_x) + grad_x

            b, n, d = grad_x.shape
            grad_x = torch.reshape(grad_x, (1, b, n * d))
            grad_x = attn2(grad_x) + grad_x
            grad_x = torch.reshape(grad_x, (b, n, d))
            x = ff2(grad_x) + grad_x

        return x


class CRATE(nn.Module):
    def __init__(
            self, *, image_size, patch_size, num_classes, dim, depth, heads, pool='cls', channels=3, dim_head=64,
            dropout=0., emb_dropout=0., ista=0.1
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature_last = x
        return self.mlp_head(x)


class TabularCRATE(nn.Module):
    def __init__(self, *, seq_length, num_features, dim, depth, heads, pool='cls', dim_head=64, dropout=0.,
                 emb_dropout=0., ista=0.1):
        super().__init__()
        self.to_embedding = nn.Sequential(
            nn.Linear(num_features, dim),
            nn.LayerNorm(dim),
        )

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature_last = x
        return x, feature_pre, feature_last


class TabularCRATEPredictor(nn.Module):
    def __init__(self, *, mri_kwargs, snp_kwargs, num_classes=1):
        super().__init__()
        self.mri_model = TabularCRATE(**mri_kwargs)
        self.snp_model = TabularCRATE(**snp_kwargs)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mri_kwargs['dim'] + snp_kwargs['dim']),
            nn.Linear(mri_kwargs['dim'] + snp_kwargs['dim'], num_classes)
        )

    def forward(self, mri, snp):
        mri_out, _, _ = self.mri_model(mri)
        snp_out, _, _ = self.snp_model(snp)
        x = torch.cat([mri_out, snp_out], dim=1)
        return self.mlp_head(x)


class SingleTransformerTabularCRATE(nn.Module):
    def __init__(self, *, mri_seq_len, snp_seq_len, dim, depth, heads, pool='cls', dim_head=64, dropout=0.,
                 emb_dropout=0., ista=0.1, num_classes=1, inter_attention=False):
        super().__init__()
        self.snp_seq_len = snp_seq_len
        self.mri_seq_len = mri_seq_len
        self.to_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim),
            ) for _ in range(mri_seq_len + snp_seq_len)
        ])

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.pos_embedding = nn.Parameter(torch.randn(1, self.mri_seq_len + self.snp_seq_len + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.inter_attention = inter_attention

        if not inter_attention:
            self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista)
        else:
            self.transformer = InterAttentionTransformer(dim, depth, heads, dim_head,
                                                         seq_length=mri_seq_len + snp_seq_len + 1, dropout=dropout,
                                                         ista=ista)

        self.num_classes = num_classes
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, mri, snp):
        mri = torch.stack([self.to_embeddings[i](mri[:, i]) for i in range(self.mri_seq_len)], dim=1)
        snp = torch.stack([self.to_embeddings[self.mri_seq_len + i](snp[:, i]) for i in range(self.snp_seq_len)], dim=1)
        x = torch.cat([mri, snp], dim=1)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        out = self.mlp_head(x)
        feature_last = x
        return out

    def get_last_key(self, mri, snp, depth=11):
        mri = torch.stack([self.to_embeddings[i](mri[:, i]) for i in range(self.mri_seq_len)], dim=1)
        snp = torch.stack([self.to_embeddings[self.mri_seq_len + i](snp[:, i]) for i in range(self.snp_seq_len)], dim=1)
        x = torch.cat([mri, snp], dim=1)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for i, (attn1, ff1, attn2, ff2) in enumerate(self.transformer.layers):
            if i < depth:
                grad_x = attn1(x) + x
                grad_x = ff1(grad_x) + grad_x

                b, n, d = grad_x.shape
                grad_x = torch.reshape(grad_x, (1, b, n * d))
                grad_x = attn2(grad_x) + grad_x
                grad_x = torch.reshape(grad_x, (b, n, d))
                x = ff2(grad_x) + grad_x
            else:
                key = attn1(x, return_key=True)
                return key

    def get_last_sa(self, mri, snp, layer=11):
        mri = torch.stack([self.to_embeddings[i](mri[:, i]) for i in range(self.mri_seq_len)], dim=1)
        snp = torch.stack([self.to_embeddings[self.mri_seq_len + i](snp[:, i]) for i in range(self.snp_seq_len)], dim=1)
        x = torch.cat([mri, snp], dim=1)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        if self.inter_attention:
            for i, (attn1, ff1, attn2, ff2) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn1(x) + x
                    grad_x = ff1(grad_x) + grad_x

                    b, n, d = grad_x.shape
                    grad_x = torch.reshape(grad_x, (1, b, n * d))
                    grad_x = attn2(grad_x) + grad_x
                    grad_x = torch.reshape(grad_x, (b, n, d))
                    x = ff2(grad_x) + grad_x
                else:
                    att_map = attn1(x, return_attention=True)
                    return att_map
        else:
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x) + grad_x
                else:
                    att_map = attn(x, return_attention=True)
                    return att_map


class MMMLP(nn.Module):
    def __init__(self, *, mri_kwargs, snp_kwargs, num_classes=1):
        super().__init__()
        self.mri_model = MLP.make_baseline(**mri_kwargs)
        self.snp_model = MLP.make_baseline(**snp_kwargs)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mri_kwargs['d_out'] + snp_kwargs['d_out']),
            nn.Linear(mri_kwargs['d_out'] + snp_kwargs['d_out'], num_classes)
        )

    def forward(self, mri, snp):
        if mri.dim() > 2:
            mri = mri.squeeze(2)
        if snp.dim() > 2:
            snp = snp.squeeze(2)
        mri_out = self.mri_model(mri)
        snp_out = self.snp_model(snp)
        x = torch.cat([mri_out, snp_out], dim=1)
        return self.mlp_head(x)


class MM_ResNet(nn.Module):
    def __init__(self, *, mri_kwargs, snp_kwargs, num_classes=1):
        super().__init__()
        self.mri_model = ResNet.make_baseline(**mri_kwargs)
        self.snp_model = ResNet.make_baseline(**snp_kwargs)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mri_kwargs['d_out'] + snp_kwargs['d_out']),
            nn.Linear(mri_kwargs['d_out'] + snp_kwargs['d_out'], num_classes)
        )

    def forward(self, mri, snp):
        if mri.dim() > 2:
            mri = mri.squeeze(2)
        if snp.dim() > 2:
            snp = snp.squeeze(2)

        mri_out = self.mri_model(mri)
        snp_out = self.snp_model(snp)
        x = torch.cat([mri_out, snp_out], dim=1)
        return self.mlp_head(x)


def CRATE_tiny(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6
    )


def CRATE_small(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12
    )


def CRATE_base(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12
    )


def CRATE_large(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16
    )


def CRATE_tabular_tiny(num_classes=1):
    mri_kwargs = {
        'seq_length': 259,
        'num_features': 1,
        'dim': 64,
        'depth': 4,
        'heads': 4,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }

    snp_kwargs = {
        'seq_length': 1079,
        'num_features': 1,
        'dim': 64,
        'depth': 4,
        'heads': 4,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }
    return TabularCRATEPredictor(mri_kwargs=mri_kwargs, snp_kwargs=snp_kwargs, num_classes=num_classes)


def CRATE_tabular_small(num_classes=1):
    mri_kwargs = {
        'seq_length': 259,
        'num_features': 1,
        'dim': 96,
        'depth': 6,
        'heads': 6,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }

    snp_kwargs = {
        'seq_length': 1079,
        'num_features': 1,
        'dim': 96,
        'depth': 6,
        'heads': 6,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }
    return TabularCRATEPredictor(mri_kwargs=mri_kwargs, snp_kwargs=snp_kwargs, num_classes=num_classes)


def CRATE_tabular_base(num_classes=1):
    mri_kwargs = {
        'seq_length': 259,
        'num_features': 1,
        'dim': 128,
        'depth': 6,
        'heads': 6,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }

    snp_kwargs = {
        'seq_length': 1079,
        'num_features': 1,
        'dim': 128,
        'depth': 6,
        'heads': 6,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }
    return TabularCRATEPredictor(mri_kwargs=mri_kwargs, snp_kwargs=snp_kwargs, num_classes=num_classes)


def CRATE_tabular_large(num_classes=1):
    mri_kwargs = {
        'seq_length': 259,
        'num_features': 1,
        'dim': 256,
        'depth': 12,
        'heads': 8,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }

    snp_kwargs = {
        'seq_length': 1079,
        'num_features': 1,
        'dim': 256,
        'depth': 12,
        'heads': 8,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'ista': 0.1
    }
    return TabularCRATEPredictor(mri_kwargs=mri_kwargs, snp_kwargs=snp_kwargs, num_classes=num_classes)


def MLP_baseline(num_classes=1):
    mri_kwargs = {
        'd_in': 259,
        'd_layers': [1024, 1024, 1024, 1024],
        'dropout': 0.1,
        'd_out': 288
    }
    snp_kwargs = {
        'd_in': 1079,
        'd_layers': [1024, 1024, 1024, 1024],
        'dropout': 0.1,
        'd_out': 288
    }

    return MMMLP(mri_kwargs=mri_kwargs, snp_kwargs=snp_kwargs, num_classes=num_classes)


def ResNet_baseline(num_classes=1):
    mri_kwargs = {
        'd_in': 259,
        'n_blocks': 12,
        'd_main': 312,
        'd_hidden': 512,
        'dropout_first': 0.1,
        'dropout_second': 0.1,
        'd_out': 256
    }
    snp_kwargs = {
        'd_in': 1079,
        'n_blocks': 12,
        'd_main': 312,
        'd_hidden': 512,
        'dropout_first': 0.1,
        'dropout_second': 0.1,
        'd_out': 256
    }

    return MM_ResNet(mri_kwargs=mri_kwargs, snp_kwargs=snp_kwargs, num_classes=num_classes)


def NODE_baseline(num_classes=1):
    kwargs = {
        'd_in': 259,
        'num_layers': 6,
        'layer_dim': 68,
        'd_embedding': 4,
        'depth': 4,
        'tree_dim': 3,
        'choice_function': entmax15,
        'bin_function': entmoid15,
        'd_out': num_classes,
        'categories': [3 for _ in range(1079)],
    }

    return NODE(**kwargs)


def SNN_baseline(num_classes=1):
    kwargs = {
        'd_in': 259,
        'd_layers': [908 for _ in range(6)],
        'dropout': 0.1,
        'd_embedding': 4,
        'd_out': num_classes,
        'categories': [3 for _ in range(1079)],
    }
    return SNN(**kwargs)


def SingleTransformerTabularCRATE_large(num_classes=1):
    return SingleTransformerTabularCRATE(
        mri_seq_len=259,
        snp_seq_len=1079,
        dim=512,
        depth=7,
        heads=8,
        pool='cls',
        dim_head=512 // 8,
        dropout=0.,
        emb_dropout=0.,
        ista=0.1,
        num_classes=num_classes,
        inter_attention=False
    )


def InterAttentionSingleTransformerTabularCRATE_large(num_classes=1):
    return SingleTransformerTabularCRATE(
        mri_seq_len=259,
        snp_seq_len=1079,
        dim=64,
        depth=6,
        heads=8,
        pool='cls',
        dim_head=64 // 8,
        dropout=0.1,
        emb_dropout=0.,
        ista=0.1,
        num_classes=num_classes,
        inter_attention=True
    )
