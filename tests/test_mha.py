import math

import torch

from modules.co_mha import (  # _scaled_dot_product_attention_mod,
    _scaled_dot_product_attention_step,
)
from modules.mha import MultiheadAttention, _scaled_dot_product_attention

torch.manual_seed(42)


def test_scaled_dot_product_attention():
    Nt = 1000  # target sequence length
    Ns = 1000  # source sequence length
    E = 100  # embedding dimension
    B = 1  # batch size

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
        attn_sum[:, 1:],
        av[:, 1:],
        query1 / math.sqrt(E),
        key1.transpose(-2, -1),
        value1,
        # 0,
        # 0,
        # 0,
    )
    output2, new_state = _scaled_dot_product_attention_step(
        prev_state, query_step, key_step, value_step
    )

    assert torch.allclose(target2, output2, atol=1e-7)

    # Timing comparison
    # import time
    # num_runs = 100

    # s = time.time()
    # for _ in range(num_runs):
    #     _scaled_dot_product_attention(query2, key2, value2)
    # time_all = time.time() - s

    # s = time.time()
    # for _ in range(num_runs):
    #     _scaled_dot_product_attention_mod(query2, key2, value2)
    # time_all_mod = time.time() - s

    # s = time.time()
    # for _ in range(num_runs):
    #     _scaled_dot_product_attention_step(prev_state, query_step, key_step, value_step)
    # time_step = time.time() - s

    # Compare time_all, time_all_mod, time_step

    assert True


def test_multi_head_attention():
    L = 6  # target sequence length
    S = 5  # source sequence length
    E = 4  # embedding dimension
    N = 1  # batch size
    H = 1  # num heads
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
    # forward description
    # query: (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension. (N,L,E) if batch_first is True.
    # key: (S,N,E), where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    # value: (S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.

    query = torch.randn((L, N, E))
    key = torch.randn((S, N, E))
    value = torch.randn((S, N, E))

    attn_output, attn_output_weights = mha.forward(query, key, value)
    # attn_output: (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension. (N,L,E) if batch_first is True.
    # attn_output_weights: (N,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length.

    # Shift query, key and value by a time-step
    query_step = torch.randn((1, N, E))
    key_step = torch.randn((1, N, E))
    value_step = torch.randn((1, N, E))

    query2 = torch.cat((query[1:], query_step), dim=0)
    key2 = torch.cat((key[1:], key_step), dim=0)
    value2 = torch.cat((value[1:], value_step), dim=0)

    assert torch.equal(query[1:], query2[:-1])
    assert torch.equal(key[1:], key2[:-1])
    assert torch.equal(value[1:], value2[:-1])

    attn_output2, attn_output_weights2 = mha.forward(query, key, value)

    # Continual MHA should yield the same result by using query_step, key_step, and value_step
