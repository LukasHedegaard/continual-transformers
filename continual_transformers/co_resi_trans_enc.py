import continual as co
from torch import nn

from continual_transformers.co_re_trans_enc import (
    CoReTransformerEncoder,
    RetroactiveLambda,
)
from continual_transformers.co_si_trans_enc import CoSiTransformerEncoder


def CoReSiTransformerEncoder(
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
    """Create a Two-block Continual Transformer Encoder with a Retroactive and a Single-output block.
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

    core = CoReTransformerEncoder(
        sequence_len,
        embed_dim,
        num_heads,
        dropout,
        in_proj_bias,
        ff_hidden_dim,
        ff_activation,
        device,
        dtype,
    )

    cosi = CoSiTransformerEncoder(
        sequence_len,
        embed_dim,
        num_heads,
        dropout,
        in_proj_bias,
        query_index,
        ff_hidden_dim,
        ff_activation,
        device,
        dtype,
    )

    return co.Sequential(
        core,
        RetroactiveLambda(cosi, takes_time=True),
    )
