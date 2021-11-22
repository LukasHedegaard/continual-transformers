import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from continual.module import CoModule, TensorPlaceholder
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.activation import MultiheadAttention as _MultiheadAttention
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

MaybeTensor = Union[Tensor, TensorPlaceholder]

#
# multihead attention
#


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,
    ), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,
    ), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,
    ), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention_mod(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask

    attn_exp = torch.exp(attn)
    norm = attn_exp.sum(dim=-1)  # over key dim
    if dropout_p > 0.0:
        attn_exp = F.dropout(attn_exp, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    av = torch.bmm(attn_exp, v)
    output = av / norm.unsqueeze(-1)

    return output, attn


State = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int]


def _scaled_dot_product_attention_default_state(
    B: int,  # batch_size
    Nt: int,  # num_target
    Ns: int,  # num_source
    E: int,  # embedding_dim
    H: int,  # num_heads
    init_fn=torch.zeros,
    dtype=None,
    device=None,
):
    init_fn = partial(init_fn, dtype=dtype, device=device)
    E = E // H
    B = B * H
    a_sum_mem = init_fn((B, Nt - 1))
    av_mem = init_fn((B, Ns - 1, E))
    q_mem = init_fn((B, Nt, E))
    k_t_mem = init_fn((B, E, Ns))
    v_mem = init_fn((B, Ns, E))
    return (a_sum_mem, av_mem, q_mem, k_t_mem, v_mem)


def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, State]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

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
        # a,  # (B, Nt, Ns)
        a_sum_mem,  # (B, Nt-1)
        av_mem,  # (B, Ns-1, E)
        q_mem,  # (B, Nt, E)
        k_t_mem,  # (B, E, Ns)
        v_mem,  # (B, Ns, E)
        # q_i,  # idx of oldest value
        # k_i,  # idx of oldest value
        # v_i,  # idx of oldest value
    ) = prev_state

    B, E = q_step.shape
    q_step = q_step / math.sqrt(E)

    # Get oldest values
    q_step_old = q_mem[:, 0]  # q_i
    k_t_step_old = k_t_mem[:, :, 0]  # k_i

    # Compute oldest and new attn values
    # Old values are always subtracted, so we can do it here already
    q_sel = torch.stack((q_step_old, q_step), dim=1)
    k_sel = torch.stack((k_t_step_old, k_step), dim=2)

    # Compute oldest and newest entries in exp attention matrix
    #   T T T
    # L . . . R
    # L . . . R
    # L . . . R
    #   B B B BR

    # [T T T; B B B], shape=(B, 2, Ns-1)
    a_top_bottom = torch.exp(torch.bmm(q_sel, k_t_mem[:, :, 1:]))

    # [L L L; R R R]^T, shape=(B, Nt-1, 2)
    a_left_right = torch.exp(torch.bmm(q_mem[:, 1:, :], k_sel))

    # BR
    a_bottom_right = torch.exp((q_step * k_step).sum(dim=1))

    # TODO
    # if a_mask is not None:
    #     a += a_mask

    # TODO
    # if dropout_p > 0.0:
    #     a_exp = F.dropout(a_exp, p=dropout_p)

    # Compute normalisation constants
    a_sum = torch.cat(
        (
            a_sum_mem + a_left_right[:, :, 1] - a_left_right[:, :, 0],
            (a_top_bottom[:, 1].sum(1) + a_bottom_right).unsqueeze(1),
        ),
        dim=1,
    )

    # Compute av matrix
    av_sub = torch.bmm(
        a_left_right[:, :, 0].unsqueeze(2),
        v_mem[:, 0].unsqueeze(1),
    )

    av_add = torch.bmm(
        a_left_right[:, :, 1].unsqueeze(2),
        v_step.unsqueeze(1),
    )

    av_top = av_mem - av_sub + av_add

    av_bottom = torch.bmm(a_top_bottom[:, 1].unsqueeze(1), v_mem[:, 1:]) + (
        a_bottom_right.unsqueeze(1) * v_step
    ).unsqueeze(1)

    av = torch.cat(
        (av_top, av_bottom),
        dim=1,
    )

    # Compute final output
    output = av / a_sum.unsqueeze(-1)

    # Update states
    q_mem_new = torch.cat((q_mem[:, 1:], q_step.unsqueeze(1)), dim=1)
    k_t_mem_new = torch.cat((k_t_mem[:, :, 1:], k_step.unsqueeze(2)), dim=2)
    v_mem_new = torch.cat((v_mem[:, 1:], v_step.unsqueeze(1)), dim=1)

    # q_mem_new = q_mem
    # k_t_mem_new = k_t_mem
    # v_mem_new = v_mem

    # Update states for circular buffer
    # q_mem[:, q_i] = q_step
    # k_t_mem[:, k_i] = k_step
    # v_mem[:, v_i] = v_step

    # q_i = (q_i + 1) % q_mem.shape[1]
    # k_i = (k_i + 1) % k_t_mem.shape[2]
    # v_i = (v_i + 1) % v_mem.shape[1]

    return output, (
        a_sum[:, 1:],  # (B, Nt-1)
        av[:, 1:],  # (B, Ns-1, E)
        q_mem_new,
        k_t_mem_new,
        v_mem_new,
    )


"""
def mul_saving_scaled_dot_product_attention(
    self,
    q_step: Tensor,  # step input
    k_step: Tensor,  # step input
    v_step: Tensor,  # step input
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
):
    assert attn_mask is None, "attn_mask is not supported yet."
        assert dropout_p == 0.0, "dropout_p is not supported yet."

        B, _, E = q_step.shape
        q_step = q_step / math.sqrt(E)

        (
            a,  # (B, Nt, Ns)
            a_sum_mem,  # (B, Nt)
            av_mem, # (B, Ns, E)
            q_mem,  # (B, Nt-1, E)
            k_mem,  # (B, Ns-1, E)
            v_mem,  # (B, Ns, E)
        ) = self.get_state()

        # Update Query and Key matrices
        q = torch.cat((q_mem, q_step), dim=1)
        k = torch.cat((k_mem, k_step), dim=1)

        # Attention layout O (old), R (new_a_row), C (new_a_col)
        # O O O C
        # O O O C
        # O O O C
        # R R R R

        # (B, 1, E) x (B, E, Ns) -> (B, 1, Ns)
        new_a_row = torch.exp(torch.bmm(q_step, k.transpose(-2, -1)))

        # (B, Nt-1, E) x (B, E, 1) -> (B, Nt-1, 1)
        new_a_col = torch.exp(torch.bmm(q_mem, k_step.transpose(-2, -1)))

        # TODO: impl
        # if a_mask is not None:
        #     a += a_mask

        new_a_sum = torch.cat(
            (
                a_sum_mem - a[:, :, -1] + new_a_row[:, :, :-1],
                new_a_col.sum() + new_a_row[:, :, -1],
            ),
            dim=1,
        )

        # TODO: impl
        # if dropout_p > 0.0:
        #     a_exp = F.dropout(a_exp, p=dropout_p)


        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        av = torch.bmm(a_exp, v)
        output = av / norm.unsqueeze(-1)

        # Update states
        # Attention update: old values slide up left
        # 1 2 3 4
        # 2 2 3 4
        # 3 3 3 4
        # 4 4 4 4
        a[:, :-1, :-1] = a[:, 1:, 1:]
        a[:, -1, :] = new_a_row
        a[:, :-1, -1] = new_a_col

        # Attention sum: override
        a_sum_mem = new_a_sum[:, 1:]

        # Q, K, V update: slide up
        # 1 1 1 1
        # 2 2 2 2
        # 3 3 3 3
        # 4 4 4 4
        q_mem = q[:, 1:]
        k_mem = k[:, 1:]
        v_mem[:, :-1] = v_mem[:, 1:]
        v_mem[:, -1] = v_step

        self.set_state((a, a_sum_mem, q_mem, k_mem, v_mem))
        return output, attn
"""


def multi_head_attention_forward_step(  # noqa: C901
    prev_state: State,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, State]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.

    Shape:
        Inputs:
        - query: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        - key: :math:`(N, E)`, where N is the batch size and E is the embedding dimension.
        - value: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        - key_padding_mask: :math:`(N)` where N is the batch size.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size and E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size and E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        - state: Internal state for continual computataion.
    """
    assert add_zero_attn is False, "add_zero_attn is not supported"
    assert key_padding_mask is None, "key_padding_mask is not supported"
    assert attn_mask is None, "attn_mask is not supported"
    assert static_k is None, "static_k is not supported"
    assert static_v is None, "static_v is not supported"

    # set up shape vars
    assert len(query.shape) == 2, "query should have shape (N, E)"
    assert len(key.shape) == 2, "key should have shape (N, E)"
    assert len(value.shape) == 2, "value should have shape (N, E)"
    query = query.unsqueeze(0)  # shape = (1, N, E)
    key = key.unsqueeze(0)  # shape = (1, N, E)
    value = value.unsqueeze(0)  # shape = (1, N, E)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    # compute in-projection
    if not use_separate_proj_weight:
        # Note: Also works for single step (unqueeze dim 0)
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # add bias along batch dimension (currently second) TODO: Handle this branch
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # reshape q, k, v for multihead attention and make em batch first
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:  # TODO: Handle this branch
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float  # TODO: Handle this branch
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # (deep breath) calculate attention and out projection
    q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)
    attn_output, new_state = _scaled_dot_product_attention_step(
        prev_state, q, k, v, attn_mask, dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output, new_state


# Corresponds to MultiheadAttention in Pytorch v1.9
class CoReMultiheadAttention(CoModule, torch.nn.Module):
    r"""
    Continual Retrospective MultiHeadAttention.
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

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

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
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        torch.nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.sequence_len = sequence_len

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def get_state(self) -> Optional[State]:
        """Get model state

        Returns:
            Optional[State]: A State tuple if the model has been initialised and otherwise None.
        """
        if (
            getattr(self, "a_sum_mem", None) is not None
            and getattr(self, "av_mem", None) is not None
            and getattr(self, "q_mem", None) is not None
            and getattr(self, "k_t_mem", None) is not None
            and getattr(self, "v_mem", None) is not None
            and getattr(self, "stride_index", None) is not None
        ):
            return (
                self.a_sum_mem,
                self.av_mem,
                self.q_mem,
                self.k_t_mem,
                self.v_mem,
                self.stride_index,
            )

    def set_state(self, state: State):
        """Set model state

        Args:
            state (State): State tuple to set as new internal internal state
        """
        (
            self.a_sum_mem,
            self.av_mem,
            self.q_mem,
            self.k_t_mem,
            self.v_mem,
            self.stride_index,
        ) = state

    def clean_state(self):
        """Clean model state"""
        if hasattr(self, "a_sum_mem"):
            del self.a_sum_mem
        if hasattr(self, "av_mem"):
            del self.av_mem
        if hasattr(self, "q_mem"):
            del self.q_mem
        if hasattr(self, "k_t_mem"):
            del self.k_t_mem
        if hasattr(self, "v_mem"):
            del self.v_mem
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
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shapes for inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
              source sequence length.

              If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
              length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
              the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        return _MultiheadAttention.forward(
            self, query, key, value, key_padding_mask, need_weights, attn_mask
        )

    def _forward_step(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        prev_state: State = None,
        *args,
        **kwargs,
    ) -> Tuple[MaybeTensor, State]:
        """Forward computation for a single step with state initialisation

        Args:
            query, key, value: step inputs of shape `(B, E)` where B is the batch size and E is the embedding dimension.

        Returns:
            Tuple[MaybeTensor, State]: Step output and new state.
        """
        batch_size = query.shape[0]
        if prev_state is None:
            prev_state = (
                *_scaled_dot_product_attention_default_state(
                    batch_size,
                    self.sequence_len,
                    self.sequence_len,
                    self.embed_dim,
                    self.num_heads,
                    dtype=query.dtype,
                    device=query.device,
                ),
                -self.sequence_len,
            )

        o, new_state = multi_head_attention_forward_step(
            prev_state[:-1],
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
        )
        stride_index = prev_state[-1]
        if stride_index < 0:
            stride_index += 1

        new_state = (*new_state, stride_index)

        return (
            TensorPlaceholder(o.shape) if stride_index < 0 else o,
            new_state,
        )

    def forward_step(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        update_state=True,
        *args,
        **kwargs,
    ) -> MaybeTensor:
        r"""
        Args:
            query, key, value: step_inputs for mapping a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.

        Shapes for inputs:
            - query: :math:`(N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - new_state: Tuple of internal states.
        """
        tmp_state = self.get_state()
        # TODO: For the  efficient memory implementation, we will want to clone state here
        # if not update_state and state:
        #     state = clone(state)
        o, tmp_state = self._forward_step(query, key, value, tmp_state)

        if self.batch_first and not isinstance(o, TensorPlaceholder):
            o = o.transpose(1, 0)

        if update_state:
            self.set_state(tmp_state)

        return o

    def forward_steps(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pad_end=False,
        update_state=True,
        *args,
        **kwargs,
    ) -> MaybeTensor:
        """Forward computation for multiple steps with state initialisation

        Args:
            query (Tensor): query.
            key (Tensor): key.
            value (Tensor): value.
            pad_end (bool): Dummy parameter added to fulfill interface.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Layer output corresponding to the self-attention for the last step
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tmp_state = self.get_state()

        # TODO: For the efficient memory implementation, we will want to clone state here
        # if not update_state and tmp_state:
        #     tmp_state = clone(tmp_state)

        L = query.shape[0]
        assert L == key.shape[0]
        assert L == value.shape[0]
        for t in range(L):
            o, tmp_state = self._forward_step(query[t], key[t], value[t], tmp_state)

        if self.batch_first and not isinstance(o, TensorPlaceholder):
            o = o.transpose(1, 0)

        if update_state:
            self.set_state(tmp_state)

        return o

    @property
    def receptive_field(self) -> int:
        return self.sequence_len

    @staticmethod
    def build_from(
        module: _MultiheadAttention, sequence_len: int, **kwargs
    ) -> "CoReMultiheadAttention":
        comodule = CoReMultiheadAttention(
            **{
                **dict(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    dropout=module.dropout,
                    bias=module.in_proj_bias is not None,
                    add_bias_kv=module.bias_k is not None,
                    add_zero_attn=module.add_zero_attn,
                    kdim=module.kdim,
                    vdim=module.vdim,
                    batch_first=module.batch_first,
                    device=module.out_proj.weight.device,
                    dtype=module.out_proj.weight.dtype,
                    sequence_len=sequence_len,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            if module.in_proj_weight is not None:
                comodule.in_proj_weight.copy_(module.in_proj_weight)

            if module.q_proj_weight is not None:
                comodule.q_proj_weight.copy_(module.q_proj_weight)
            if module.k_proj_weight is not None:
                comodule.k_proj_weight.copy_(module.k_proj_weight)
            if module.v_proj_weight is not None:
                comodule.v_proj_weight.copy_(module.v_proj_weight)

            if module.in_proj_bias is not None:
                comodule.in_proj_bias.copy_(module.in_proj_bias)
            if module.out_proj is not None:
                comodule.out_proj.weight.copy_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    comodule.out_proj.bias.copy_(module.out_proj.bias)
            if module.bias_k is not None:
                comodule.bias_k.copy_(module.bias_k)
            if module.bias_v is not None:
                comodule.bias_v.copy_(module.bias_v)
        return comodule
