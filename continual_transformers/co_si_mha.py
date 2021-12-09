import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from continual.module import CallMode
from .co_mha_base import CoMultiheadAttentionBase

State = Tuple[
    Tensor,  # Q_mem, (B, Nt-1, E)
    Tensor,  # K_T_mem, (B, E, Ns)
    Tensor,  # V_mem, (B, Ns, E)
]


def _scaled_dot_product_attention_default_state(
    batch_size: int,
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    query_index=-1,
    init_fn=torch.zeros,
    dtype=None,
    device=None,
):
    init_fn = partial(init_fn, dtype=dtype, device=device)
    E = embed_dim // num_heads
    B = batch_size * num_heads
    N = sequence_len
    Nq = sequence_len - query_index - 1 if query_index >= 0 else -query_index - 1
    Q_mem = init_fn((B, Nq, E))
    K_T_mem = init_fn((B, E, N))
    V_mem = init_fn((B, N, E))
    return (Q_mem, K_T_mem, V_mem)


def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, State]:
    """
    Computes the Continual Singe-output Scaled Dot-Product Attention on query, key and value tensors.
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
        Q_mem,  # (B, Nq, E)
        K_T_mem,  # (B, E, Ns)
        V_mem,  # (B, Ns, E)
    ) = prev_state

    B, E = q_step.shape
    q_step = q_step / math.sqrt(E)
    q_sel = (Q_mem[:, 0] if Q_mem.shape[1] > 0 else q_step).unsqueeze(1)

    # Update states
    # Note: We're allowing the K and V mem to have one more entry than
    # strictly necessary to simplify computatations.

    K_T_new = torch.roll(K_T_mem, shifts=-1, dims=(2,))
    K_T_new[:, :, -1] = k_step

    V_new = torch.roll(V_mem, shifts=-1, dims=(1,))
    V_new[:, -1] = v_step

    attn = torch.bmm(q_sel, K_T_new)
    attn_sm = F.softmax(attn, dim=-1)

    if dropout_p > 0.0:
        attn_sm = F.dropout(attn_sm, p=dropout_p)

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn_sm, V_new)

    if Q_mem.shape[1] > 0:
        Q_new = torch.roll(Q_mem, shifts=-1, dims=(1,))
        Q_new[:, -1] = q_step
    else:
        Q_new = Q_mem

    new_states = (Q_new, K_T_new, V_new)

    return output, new_states


class CoSiMultiheadAttention(CoMultiheadAttentionBase):
    """
    Continual Single-output MultiHeadAttention.

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
        query_index=-1,
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
                query_index=query_index,
            ),
            _scaled_dot_product_attention_step,
        )
        assert query_index < sequence_len
        self._query_index = query_index

    def get_state(self) -> Optional[State]:
        """Get model state

        Returns:
            Optional[State]: A State tuple if the model has been initialised and otherwise None.
        """
        if (
            getattr(self, "Q_mem", None) is not None
            and getattr(self, "K_T_mem", None) is not None
            and getattr(self, "V_mem", None) is not None
            and getattr(self, "stride_index", None) is not None
        ):
            return (
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
            self.Q_mem,
            self.K_T_mem,
            self.V_mem,
            self.stride_index,
        ) = state

    def clean_state(self):
        """Clean model state"""
        if hasattr(self, "Q_mem"):
            del self.Q_mem
        if hasattr(self, "K_T_mem"):
            del self.K_T_mem
        if hasattr(self, "V_mem"):
            del self.V_mem
        if hasattr(self, "stride_index"):
            del self.stride_index

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # Select a single query entry
        if self.batch_first:
            query = query[:, self._query_index].unsqueeze(1)
        else:
            query = query[self._query_index].unsqueeze(0)

        return CoMultiheadAttentionBase.forward(
            self, query, key, value, key_padding_mask, need_weights, attn_mask
        )

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
        flops += 2 * n * d + 2 * d
    if include_adds:
        flops += 2 * n * d - d - 1
    if include_exps:
        flops += n

    return flops


def forward_step_flops(
    sequence_len, embed_dim, include_muls=True, include_adds=False, include_exps=False
):
    return forward_flops(
        sequence_len, embed_dim, include_muls, include_adds, include_exps
    )
