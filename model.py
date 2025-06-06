import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: (batch, seq_len)
        # output: (batch, seq_len, emb_size) scaled by sqrt(emb_size)
        return self.embedding(tokens) * self.scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 2*k) → return (..., 2*k)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(dim: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # dim: 헤드당 차원(d_k), seq_len: 최대 시퀀스 길이
    inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )  # (dim/2,)
    t = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
    sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
    return cos, sin


def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:, :, :seq_len, :].to(dtype=dtype)
    sin = sin[:, :, :seq_len, :].to(dtype=dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.float()
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        context = torch.matmul(p_attn, value)
        return context, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_model: int,
            dropout: float = 0.1,
            max_seq_len: int = 2048,
            use_rope: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rope = use_rope

        self.lin_q = nn.Linear(d_model, d_model, bias=False)
        self.lin_k = nn.Linear(d_model, d_model, bias=False)
        self.lin_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn = Attention(dropout)

        if use_rope:
            cos, sin = build_rotary_pos_emb(self.d_k, max_seq_len)
            self.register_buffer("cos", cos)  # float32
            self.register_buffer("sin", sin)  # float32

    def forward(
            self,
            x_q: torch.Tensor,
            x_k: torch.Tensor,
            x_v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x_q.size()
        dtype = x_q.dtype

        q = self.lin_q(x_q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.lin_k(x_k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.lin_v(x_v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin, seq_len, dtype)

        context, _ = self.attn(q, k, v, mask)  # context: (batch, n_heads, seq_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)

        out = self.out_proj(context)  # (batch, seq_len, d_model)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SublayerConnection(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)  # (batch, seq_len, 2*d_ff)
        x = self.activation(x)  # (batch, seq_len, d_ff)
        x = self.linear2(x)  # (batch, seq_len, d_model)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout, max_seq_len, use_rope=True)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer_attn = SublayerConnection(d_model, dropout)
        self.sublayer_ffn = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.sublayer_attn(x, lambda _x: self.self_attn(_x, _x, _x, mask))
        x = self.sublayer_ffn(x, self.ffn)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return self.pos_embedding[:, :seq_len, :]


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 16,
            emb_size: int = 768,
            img_size: int = 224
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (batch, emb_size, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)  # (batch, emb_size, n_patches)
        return x.transpose(1, 2)  # (batch, n_patches, emb_size)


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


class PositionwiseFeedForwardViT(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.dropout(x)


class EncoderBlockViT(nn.Module):
    def __init__(
            self,
            hidden: int,
            attn_heads: int,
            feed_forward_hidden: int,
            dropout: float,
            max_seq_len: int = 2048
    ):
        super().__init__()
        self.attn = MultiHeadedAttention(attn_heads, hidden, dropout, max_seq_len, use_rope=False)
        self.ffn = PositionwiseFeedForwardViT(hidden, feed_forward_hidden, dropout)
        self.sublayer1 = SublayerConnection(hidden, dropout)
        self.sublayer2 = SublayerConnection(hidden, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.sublayer1(x, lambda _x: self.attn(_x, _x, _x, mask))
        x = self.sublayer2(x, self.ffn)
        return x


class ViT(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            emb_size: int = 768,
            n_layers: int = 12,
            attn_heads: int = 12,
            dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        num_patches = (img_size // patch_size) ** 2
        max_len = num_patches + 1
        self.pos_embedding = PositionalEmbedding(d_model=emb_size, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            EncoderBlockViT(emb_size, attn_heads, emb_size * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding(x)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x, mask=None)
        return self.norm(x)  # (batch, 1 + n_patches, emb_size)


class LLaVAConfig:
    def __init__(
            self,
            vocab_size: int = 32000,
            d_model: int = 512,
            n_layers: int = 12,
            n_heads: int = 8,
            d_ff: int = 2048,
            max_text_len: int = 512,
            img_size: int = 224,
            patch_size: int = 16,
            img_emb_size: int = 768,
            img_n_layers: int = 12,
            img_n_heads: int = 12,
            dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_text_len = max_text_len

        self.img_size = img_size
        self.patch_size = patch_size
        self.img_emb_size = img_emb_size
        self.img_n_layers = img_n_layers
        self.img_n_heads = img_n_heads

        self.dropout = dropout


class LLaVAModel(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config

        self.vit = ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=3,
            emb_size=config.img_emb_size,
            n_layers=config.img_n_layers,
            attn_heads=config.img_n_heads,
            dropout=config.dropout
        )
        self.vit_norm = RMSNorm(config.img_emb_size)
        self.img_proj = nn.Linear(config.img_emb_size, config.d_model, bias=False)

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.dec_pos_embedding = PositionalEmbedding(d_model=config.d_model, max_len=config.max_text_len + 1)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_seq_len=(config.max_text_len + 1)
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(
            self,
            images: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, T = input_ids.size()

        vit_feats = self.vit(images)
        cls_feats = vit_feats[:, 0, :]  # (batch, img_emb_size)
        cls_feats = self.vit_norm(cls_feats)  # (batch, img_emb_size)

        img_prefix = self.img_proj(cls_feats)  # (batch, d_model)
        img_prefix = img_prefix.unsqueeze(1)  # (batch, 1, d_model)

        txt_emb = self.token_embedding(input_ids)  # (batch, T, d_model)

        x = torch.cat([img_prefix, txt_emb], dim=1)  # (batch, 1 + T, d_model)

        x = x + self.dec_pos_embedding(x)  # (batch, 1 + T, d_model)

        full_len = T + 1
        device = x.device
        causal = torch.tril(torch.ones((full_len, full_len), device=device, dtype=torch.bool))
        causal = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, full_len, full_len)

        if attention_mask is not None:
            padding_mask = attention_mask.view(batch_size, 1, 1, T).to(torch.bool)  # (batch,1,1,T)
            causal_mask = causal.expand(batch_size, -1, -1, -1).clone()  # (batch,1,full_len,full_len)
            causal_mask[:, :, 1:, 1:] = causal_mask[:, :, 1:, 1:] & padding_mask
            mask = causal_mask
        else:
            mask = causal.expand(batch_size, -1, -1, -1)  # (batch,1,full_len,full_len)

        for block in self.decoder_blocks:
            x = block(x, mask)  # x: (batch, 1+T, d_model)

        x = self.final_norm(x)

        txt_feats = x[:, 1:, :]  # (batch, T, d_model)
        logits = self.output_proj(txt_feats) + self.lm_bias  # (batch, T, vocab_size)

        return logits
