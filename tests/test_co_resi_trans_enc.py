import torch
from ptflops import get_model_complexity_info

from continual_transformers.co_resi_trans_enc import CoReSiTransformerEncoder


def test_CoReTransformerEncoder():
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    N = 1  # batch size
    H = 2  # num heads

    enc = CoReSiTransformerEncoder(
        sequence_len=T,
        embed_dim=E,
        num_heads=H,
    )

    query = torch.randn((N, E, T))

    # Baseline
    o = enc.forward(query)

    # Forward step
    enc.forward_steps(query[:, :, :-1])  # init

    o_step = enc.forward_step(query[:, :, -1])

    assert torch.allclose(o, o_step)

    # Forward steps
    enc.clean_state()
    o_steps = enc.forward_steps(query)

    assert torch.allclose(o, o_steps.squeeze(2))

    # FLOPs
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

    assert step_flops < flops  # flops / step_flops = 1.66
