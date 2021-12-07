import math

import torch
from continual.module import TensorPlaceholder

from modules.co_mha import (
    CoReMultiheadAttention,
    _scaled_dot_product_attention_default_state,
    _scaled_dot_product_attention_step,
)
from modules.mha import MultiheadAttention, _scaled_dot_product_attention

torch.manual_seed(42)


def test_multi_head_attention():
    L = 10  # target sequence length
    S = L  # source sequence length
    E = 4  # embedding dimension
    N = 4  # batch size
    H = 2  # num heads
    mha = MultiheadAttention(
        embed_dim=E,
        num_heads=H,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    )
    comha = CoReMultiheadAttention.build_from(mha, sequence_len=L)

    query = torch.randn((L, N, E))
    key = torch.randn((S, N, E))
    value = torch.randn((S, N, E))

    # forward description
    # query: (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension. (N,L,E) if batch_first is True.
    # key: (S,N,E), where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    # value: (S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    attn_output, attn_weight = mha.forward(query, key, value)

    # Forward is identical
    attn_output2, attn_weight2 = comha.forward(query, key, value)
    assert torch.equal(attn_output, attn_output2)
    assert torch.equal(attn_weight, attn_weight2)

    # Forward steps gives same output for full sequence
    attn_output3 = comha.forward_steps(query, key, value)
    assert torch.allclose(attn_output, attn_output3)

    # Initialise, then use forward_step
    comha.clean_state()
    attn_output_dummy = comha.forward_steps(query[:-1], key[:-1], value[:-1])
    assert isinstance(attn_output_dummy, TensorPlaceholder)
    attn_output4 = comha.forward_step(query[-1], key[-1], value[-1])
    assert torch.allclose(attn_output, attn_output4)

    # Shift query, key and value by a time-step
    query_step = torch.randn((N, E))
    key_step = torch.randn((N, E))
    value_step = torch.randn((N, E))

    query2 = torch.cat((query[1:], query_step.unsqueeze(0)), dim=0)
    key2 = torch.cat((key[1:], key_step.unsqueeze(0)), dim=0)
    value2 = torch.cat((value[1:], value_step.unsqueeze(0)), dim=0)

    assert torch.equal(query[1:], query2[:-1])
    assert torch.equal(key[1:], key2[:-1])
    assert torch.equal(value[1:], value2[:-1])

    attn_output_next, _ = mha.forward(query2, key2, value2)

    # Continual MHA should yield the same result by using query_step, key_step, and value_step
    attn_output_next2 = comha.forward_step(query_step, key_step, value_step)
    assert torch.allclose(attn_output_next, attn_output_next2)


def test_scaled_dot_product_attention_step():
    Nt = 10  # target sequence length
    Ns = Nt  # source sequence length
    E = 5  # embedding dimension
    B = 2  # batch size
    H = 1  # num heads

    query1 = torch.randn((B, Nt, E))
    key1 = torch.randn((B, Ns, E))
    value1 = torch.randn((B, Ns, E))

    # Manually compute first output
    q = query1 / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn_exp = torch.exp(torch.bmm(q, key1.transpose(-2, -1)))
    attn_sum = attn_exp.sum(dim=-1)  # over key dim

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    av = torch.bmm(attn_exp, value1)
    output1 = av / attn_sum.unsqueeze(-1)

    # Sanity check
    target1, _ = _scaled_dot_product_attention(query1, key1, value1)
    assert torch.allclose(target1, output1, atol=1e-7)

    # == Ready for actual test! ==

    # Shift query, key and value by a time-step
    query_step = torch.randn((B, E))
    key_step = torch.randn((B, E))
    value_step = torch.randn((B, E))

    query2 = torch.cat((query1[:, 1:], query_step.unsqueeze(1)), dim=1)
    key2 = torch.cat((key1[:, 1:], key_step.unsqueeze(1)), dim=1)
    value2 = torch.cat((value1[:, 1:], value_step.unsqueeze(1)), dim=1)

    # Manually compute first output
    q = query2 / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # attn_exp2 = torch.exp(torch.bmm(q, key2.transpose(-2, -1)))
    # av2 = torch.bmm(attn_exp2, value2)

    target2, _ = _scaled_dot_product_attention(query2, key2, value2)

    prev_state = (
        # attn_sum[:, 1:],
        attn_sum[:, 1:].unsqueeze(-1),
        av[:, 1:],
        # query1 / math.sqrt(E),
        query1[:, 1:] / math.sqrt(E),
        key1.transpose(-2, -1),
        value1,
        # 0,
        # 0,
        # 0,
    )
    output2, new_state = _scaled_dot_product_attention_step(
        prev_state, query_step, key_step, value_step
    )

    for p, n in zip(prev_state, new_state):
        assert p.shape == n.shape

    assert torch.allclose(target2, output2, atol=1e-7)

    # Now, let's try from zero-init
    state = _scaled_dot_product_attention_default_state(
        B, Nt, Ns, E, H, dtype=torch.float
    )
    for i in range(Nt):
        output_step, state = _scaled_dot_product_attention_step(
            state, query1[:, i], key1[:, i], value1[:, i]
        )

    assert torch.allclose(output_step, target1, atol=1e-7)

    output_step, state = _scaled_dot_product_attention_step(
        state, query_step, key_step, value_step
    )
    assert torch.allclose(output_step, target2, atol=1e-7)
