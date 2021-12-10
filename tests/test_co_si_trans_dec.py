import torch
from ptflops import get_model_complexity_info

from continual_transformers.co_si_trans_dec import CoSiTransformerEncoder


def test_CoSiTransformerEncoder():
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    N = 1  # batch size
    H = 2  # num heads

    enc = CoSiTransformerEncoder(
        sequence_len=T,
        embed_dim=E,
        num_heads=H,
        query_index=-2,
    )

    query = torch.randn((N, E, T))

    o = enc.forward(query)

    enc.forward_steps(query[:, :, :-1])

    o_step = enc.forward_step(query[:, :, -1])

    assert torch.allclose(o, o_step.unsqueeze(-1))

    flops, _ = get_model_complexity_info(
        enc,
        (E, T),
        as_strings=False,  # input_constructor=input_constructor,
    )

    enc.call_mode = "forward_step"
    step_flops, _ = get_model_complexity_info(
        enc,
        (E,),
        as_strings=False,  # input_constructor=input_constructor,
    )

    assert step_flops < flops
