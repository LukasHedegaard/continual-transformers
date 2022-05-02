import torch
from ptflops import get_model_complexity_info
from torch.functional import Tensor

from continual_transformers.co_re_trans_enc import (
    CoReTransformerEncoder,
    RetroactiveUnity,
)


def test_RetroactiveUnity():
    ru = RetroactiveUnity(delay=2)
    assert not isinstance(ru.forward_step(torch.tensor([1])), Tensor)
    assert not isinstance(ru.forward_step(torch.tensor([2])), Tensor)
    o = ru.forward_step(torch.tensor([3]))
    assert torch.equal(o, torch.tensor([1, 2, 3]).unsqueeze(0))
    o = ru.forward_step(torch.tensor([4]))
    assert torch.equal(o, torch.tensor([2, 3, 4]).unsqueeze(0))


def test_CoReTransformerEncoder():
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    N = 1  # batch size
    H = 2  # num heads

    enc = CoReTransformerEncoder(
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

    assert torch.allclose(o, o_step, atol=1e-6)

    # Forward steps
    enc.clean_state()
    o_steps = enc.forward_steps(query)

    assert torch.allclose(o, o_steps.squeeze(2), atol=1e-6)

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

    assert step_flops < flops  # flops / step_flops = 2.15
