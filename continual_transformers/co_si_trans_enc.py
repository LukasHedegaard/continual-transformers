from functools import reduce
from typing import Sequence

import continual as co
import torch
from torch import nn
from torch.functional import Tensor

from continual_transformers.co_mha_base import MaybeTensor

from .co_si_mha import CoSiMultiheadAttention


class SelectOrDelay(co.Delay):
    """Select a temporal index during forward, or delay correspondingly during forward_step(s)"""

    def forward(self, x: Tensor) -> MaybeTensor:
        assert len(x.shape) >= 3  # N, C, T
        return x[:, :, -1 - self._delay].unsqueeze(2)


def CoSiTransformerEncoder(
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    dropout=0.0,
    in_proj_bias=True,
    query_index=-1,
    ff_hidden_dim: int = None,
    ff_activation=nn.GELU(),
    device=None,
    dtype=None,
):
    """Create a Continual Single-Output Transformer Encoder block.
    The block assumes inputs of shape `(N, E, T)` during `forward` and `forward_steps` and (N, E) during `forward_step`.

    Args:
        sequence_len (int): Sequence length.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of head in self-attention
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        in_proj_bias (bool, optional): Whether to include bias in initial linear projection. Defaults to True.
        query_index (int, optional): Temporal index over which the attention query should be made. Defaults to -1.
        ff_hidden_dim (int, optional): Hidden dimension for feed-forward network. If None, the `embed_dim` is used. Defaults to None.
        ff_activation (optional): Which activation to applu in feed-forward. Defaults to nn.GELU().
    """

    mha = CoSiMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=in_proj_bias,
        batch_first=True,
        embed_dim_second=True,
        query_index=query_index,
        device=device,
        dtype=dtype,
        sequence_len=sequence_len,
        forward_returns_attn_mask=False,
    )

    ff_hidden_dim = ff_hidden_dim or embed_dim

    ff = co.Sequential(
        co.Linear(embed_dim, ff_hidden_dim, channel_dim=1),
        ff_activation,
        nn.Dropout(p=dropout),
        co.Linear(ff_hidden_dim, embed_dim, channel_dim=1),
        nn.Dropout(p=dropout),
    )

    def sum_last_pairs(inputs: Sequence[Tensor]) -> Tensor:
        if inputs[0].shape != inputs[1].shape:
            T_min = min(inputs[i].shape[2] for i in range(len(inputs)))
            inputs = [inp[:, :, -T_min:] for inp in inputs]
        return reduce(torch.Tensor.add, inputs[1:], inputs[0])

    return co.Sequential(
        co.BroadcastReduce(
            SelectOrDelay(mha.delay),
            mha,
            reduce=sum_last_pairs,
            auto_delay=False,
        ),
        co.Lambda(nn.LayerNorm(embed_dim), takes_time=False),
        co.Residual(ff),
        co.Lambda(nn.LayerNorm(embed_dim), takes_time=False),
    )
