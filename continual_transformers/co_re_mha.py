import math
from functools import partial
from typing import Optional, Tuple

import torch
from torch import Tensor
from continual.module import CallMode
from continual_transformers.co_mha_base import CoMultiheadAttentionBase

State = Tuple[
    Tensor,  # d_mem, (B, Nt-1)
    Tensor,  # AV_mem, (B, Ns-1, E)
    Tensor,  # Q_mem, (B, Nt-1, E)
    Tensor,  # K_T_mem, (B, E, Ns)
    Tensor,  # V_mem, (B, Ns, E)
]


def _scaled_dot_product_attention_default_state(
    batch_size: int,
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    init_fn=torch.zeros,
    dtype=None,
    device=None,
):
    init_fn = partial(init_fn, dtype=dtype, device=device)
    E = embed_dim // num_heads
    B = batch_size * num_heads
    N = sequence_len
    d_mem = init_fn((B, N - 1, 1))
    AV_mem = init_fn((B, N - 1, E))
    Q_mem = init_fn((B, N - 1, E))
    K_T_mem = init_fn((B, E, N))
    V_mem = init_fn((B, N, E))
    return (d_mem, AV_mem, Q_mem, K_T_mem, V_mem)


def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, State]:
    """
    Computes the Continual Retroactive Scaled Dot-Product Attention on query, key and value tensors.
    Returns attended values and updated states.

    Args:
        q_step, k_step, v_step: query, key and value tensors for a step. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q_step: :math:`(B, E)` where B is batch size and E is embedding dimension.
        - k_step: :math:`(B, E)` where B is batch size and E is embedding dimension.
        - v_step: :math:`(B, E)` where B is batch size and E is embedding dimension.

        - Output: attention values have shape :math:`(B, Nt, E)`; new state
    """
    assert attn_mask is None, "attn_mask is not supported yet."
    assert dropout_p == 0.0, "dropout_p is not supported yet."

    (
        d_mem,  # (B, Nt-1)
        AV_mem,  # (B, Ns-1, E)
        Q_mem,  # (B, Nt-1, E)
        K_T_mem,  # (B, E, Ns)
        V_mem,  # (B, Ns, E)
    ) = prev_state

    B, E = q_step.shape
    q_step = q_step / math.sqrt(E)

    # Compute oldest and newest entries in attention matrix A:
    # L . . . R
    # L . . . R
    # L . . . R
    #   B B B B

    # Left column attention values
    A_left = torch.exp(torch.bmm(Q_mem, K_T_mem[:, :, 0].unsqueeze(-1)))

    # Right column attention values
    A_right = torch.exp(torch.bmm(Q_mem, k_step.unsqueeze(-1)))

    # Update Q_mem and K_mem
    Q_mem_new = torch.roll(Q_mem, shifts=-1, dims=(1,))
    Q_mem_new[:, -1] = q_step

    K_T_mem_new = torch.roll(K_T_mem, shifts=-1, dims=(2,))
    K_T_mem_new[:, :, -1] = k_step

    # Bottom row attention values
    A_bottom = torch.exp(torch.bmm(q_step.unsqueeze(1), K_T_mem_new))

    # Compute normalisation
    d = torch.cat(
        (
            d_mem - A_left + A_right,
            (A_bottom.sum(-1)).unsqueeze(-1),
        ),
        dim=1,
    )

    # Compute AV matrix top
    AV_sub = torch.bmm(A_left, V_mem[:, 0].unsqueeze(1))
    AV_add = torch.bmm(A_right, v_step.unsqueeze(1))
    AV_top = AV_mem - AV_sub + AV_add

    # Update V_mem
    V_mem_new = torch.roll(V_mem, shifts=-1, dims=(1,))
    V_mem_new[:, -1] = v_step

    # Compute AV_bottom
    AV_bottom = torch.bmm(A_bottom, V_mem_new)

    AV_new = torch.cat((AV_top, AV_bottom), dim=1)

    # Compute final output
    output = AV_new / d

    new_states = (
        d[:, 1:],  # (B, Nt-1)
        AV_new[:, 1:],  # (B, Ns-1, E)
        Q_mem_new,
        K_T_mem_new,
        V_mem_new,
    )

    return output, new_states


class CoReMultiheadAttention(CoMultiheadAttentionBase):
    """
    Continual Retroactive MultiHeadAttention.
    It augments the MultiHeadAttention in PyTorch with
    `forward_step` / `forward_steps` functions, in which one / more
    query, key, and value tokens are passed to yield the multihead attention
    corresponding to the last (self.sequence_len) tokens.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        sequence_len: Length of token sequence

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        sequence_len=None,
        forward_returns_attn_mask=True,
    ) -> None:
        CoMultiheadAttentionBase.__init__(
            self,
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
            sequence_len,
            partial(
                _scaled_dot_product_attention_default_state,
                sequence_len=sequence_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
            ),
            _scaled_dot_product_attention_step,
            forward_returns_attn_mask,
        )

    def get_state(self) -> Optional[State]:
        """Get model state

        Returns:
            Optional[State]: A State tuple if the model has been initialised and otherwise None.
        """
        if (
            getattr(self, "d_mem", None) is not None
            and getattr(self, "AV_mem", None) is not None
            and getattr(self, "Q_mem", None) is not None
            and getattr(self, "K_T_mem", None) is not None
            and getattr(self, "V_mem", None) is not None
            and getattr(self, "stride_index", None) is not None
        ):
            return (
                self.d_mem,
                self.AV_mem,
                self.Q_mem,
                self.K_T_mem,
                self.V_mem,
                self.stride_index,
            )

    def set_state(self, state: State):
        """Set model state

        Args:
            state (State): State tuple to set as new internal internal state
        """
        (
            self.d_mem,
            self.AV_mem,
            self.Q_mem,
            self.K_T_mem,
            self.V_mem,
            self.stride_index,
        ) = state

    def clean_state(self):
        """Clean model state"""
        if hasattr(self, "d_mem"):
            del self.d_mem
        if hasattr(self, "AV_mem"):
            del self.AV_mem
        if hasattr(self, "Q_mem"):
            del self.Q_mem
        if hasattr(self, "K_T_mem"):
            del self.K_T_mem
        if hasattr(self, "V_mem"):
            del self.V_mem
        if hasattr(self, "stride_index"):
            del self.stride_index

    def flops(self, include_muls=True, include_adds=False, include_exps=False):
        return {
            CallMode.FORWARD: forward_flops,
            CallMode.FORWARD_STEP: forward_step_flops,
        }[self.call_mode](
            self.sequence_len, self.embed_dim, include_muls, include_adds, include_exps
        )


def forward_flops(
    sequence_len, embed_dim, include_muls=True, include_adds=False, include_exps=False
):
    n = sequence_len
    d = embed_dim

    flops = 0

    if include_muls:
        flops += 2 * n * n * d + 2 * n * d
    if include_adds:
        flops += 2 * n * n - n * d - n
    if include_exps:
        flops += n * n

    return flops


def forward_step_flops(
    sequence_len, embed_dim, include_muls=True, include_adds=False, include_exps=False
):
    n = sequence_len
    d = embed_dim

    flops = 0

    if include_muls:
        flops += 7 * n * d + 2 * n - 3 * d
    if include_adds:
        flops += 6 * n * d + 3 * n - 6 * d - 3
    if include_exps:
        flops += 3 * n - 2

    return flops
