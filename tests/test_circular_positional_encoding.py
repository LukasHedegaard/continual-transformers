import torch

from continual_transformers.circular_positional_encoding import (
    CircularPositionalEncoding,
)


def test_CircularPositionalEncoding():
    B, C, T = 2, 3, 4

    x = torch.zeros((B, C, T))

    cpe = CircularPositionalEncoding(embed_dim=C, num_embeds=T)

    o_forward = cpe.forward(x)

    o_forward_steps = cpe.forward_steps(x[:, :, :-1])
    assert torch.equal(o_forward[:, :, :-1], o_forward_steps)

    o_forward_step = cpe.forward_step(x[:, :, -1])
    assert torch.equal(o_forward[:, :, -1], o_forward_step)
