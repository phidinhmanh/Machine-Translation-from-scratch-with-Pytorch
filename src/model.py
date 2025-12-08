import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x, mask=None, context=None):
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_proj(x)
        if context is None:
            kv = self.kv_proj(x)
        else:
            kv = self.kv_proj(context)

        k, v = kv.chunk(2, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -float("inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        # recombie head
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        feedforward_expansion=3,
        is_decoder=False,
    ):
        super(TransformerBlock, self).__init__()
        self.is_decoder = is_decoder  # ty:ignore[unresolved-attribute]
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        if is_decoder:
            self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
            self.norm2 = nn.RMSNorm(embed_dim)
            self.dropout2 = nn.Dropout(dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_expansion * embed_dim, embed_dim),
        )
        self.norm3 = nn.RMSNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mask, context=None, src_mask=None):
        # Pre-LN: Norm -> Attn -> Add
        norm_x = self.norm1(x)
        atten_out = self.self_attn(norm_x, mask=mask)
        x = x + self.dropout1(atten_out)

        if self.is_decoder:
            norm_x = self.norm2(x)
            cross_atten_out = self.cross_attn(norm_x, mask=src_mask, context=context)
            x = x + self.dropout2(cross_atten_out)

        norm_x = self.norm3(x)
        feedforward_out = self.feedforward(norm_x)
        x = x + self.dropout3(feedforward_out)
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, heads, blocks, device, dropout=0.1, ff_expansion=3
    ) -> None:
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerBlock(embed_dim, heads, dropout, ff_expansion)
                for _ in range(blocks)
            ]
        )
        self.transformer_decoder = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, heads, dropout, ff_expansion, is_decoder=True
                )
                for _ in range(blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.final_norm = nn.RMSNorm(embed_dim)
        self.linear.weight = self.embedding.weight
        self.apply(self._init_weights)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = (trg != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        tril = torch.tril(
            torch.ones((trg_len, trg_len), dtype=torch.bool, device=self.device)
        ).expand(N, 1, trg_len, trg_len)
        trg_mask = trg_mask & tril
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_out = self.dropout(
            self.pos_encoder(self.embedding(src) * math.sqrt(self.embed_dim))
        )
        dec_out = self.dropout(
            self.pos_encoder(self.embedding(trg) * math.sqrt(self.embed_dim))
        )

        for layer in self.transformer_encoder:
            enc_out = layer(enc_out, src_mask)
        for layer in self.transformer_decoder:
            dec_out = layer(dec_out, trg_mask, context=enc_out, src_mask=src_mask)
        dec_out = self.final_norm(dec_out)
        out = self.linear(dec_out)
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

        elif isinstance(module, nn.RMSNorm):
            nn.init.constant_(module.weight, 1.0)
