import continual as co
from torch import nn
from torch.functional import Tensor
import torch
from typing import Tuple, Callable
from .co_re_mha import CoReMultiheadAttention
from continual.delay import State as DelayState, PaddingMode


class RetroactiveUnity(co.Delay):
    """Unity mapping during forward. During forward_step(s), a single-to-many mapping is assumed,
    and all cached values are output."""

    def __init__(
        self,
        delay: int,
        temporal_fill: PaddingMode = "zeros",
        auto_shrink: bool = False,
        time_dim=-1,
    ):
        """Initialise Delay block

        Args:
            delay (int): the number of steps to delay an output.
            temporal_fill (PaddingMode, optional): Temporal state initialisation mode ("zeros" or "replicate"). Defaults to "zeros".
            auto_shrink (int, optional): Whether to shrink the temporal dimension of the feature map during forward.
                This is handy for residuals that are parallel to modules which reduce the number of temporal steps. Defaults to False.
            time_dim (int, optional): Which dimension to concatenate step outputs along
        """
        self.time_dim = time_dim
        co.Delay.__init__(self, delay, temporal_fill, auto_shrink)

    def init_state(
        self,
        first_output: Tensor,
    ) -> DelayState:
        padding = self.make_padding(first_output)
        state_buffer = torch.stack([padding for _ in range(self.delay + 1)], dim=0)
        state_index = -self.delay
        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
        return state_buffer, state_index

    def _forward_step(
        self, input: Tensor, prev_state: DelayState
    ) -> Tuple[Tensor, DelayState]:
        if prev_state is None:
            buffer, index = self.init_state(input)
        else:
            buffer, index = prev_state

        # Update state
        buffer[index % (self.delay + 1)] = input
        new_index = index + 1
        if new_index > 0:
            new_index = new_index % self.delay

        # Get output
        if index >= 0:
            output = buffer.clone().roll(shifts=-index - 1, dims=0)
            idx = (
                self.time_dim + len(output.shape)
                if self.time_dim < 0
                else self.time_dim
            )
            output = output.permute(
                list(range(1, idx + 1)) + [0] + list(range(idx + 1, len(output.shape)))
            )
        else:
            output = co.TensorPlaceholder(buffer[0].shape)

        return output, (buffer, new_index)


class RetroactiveLambda(co.Lambda):
    """
    Wrapper for functions that are applied after retroactive modules.
    """

    def forward_step(self, input: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(input)

    def forward_steps(self, input: Tensor, *args, **kwargs) -> Tensor:
        return torch.stack(
            [self.forward(input[:, :, t]) for t in range(input.shape[2])], dim=2
        )

    @staticmethod
    def build_from(
        fn: Callable[[Tensor], Tensor], takes_time=False
    ) -> "RetroactiveLambda":
        return RetroactiveLambda(fn, takes_time)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def CoReTransformerEncoder(
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    dropout=0.0,
    in_proj_bias=True,
    ff_hidden_dim: int = None,
    ff_activation=nn.GELU(),
    device=None,
    dtype=None,
):
    """Create a Continual Retroactive Transformer Encoder block.
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

    mha = CoReMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=in_proj_bias,
        batch_first=True,
        embed_dim_second=True,
        device=device,
        dtype=dtype,
        sequence_len=sequence_len,
        forward_returns_attn_mask=False,
    )

    ff_hidden_dim = ff_hidden_dim or embed_dim

    ff = nn.Sequential(
        co.Linear(embed_dim, ff_hidden_dim, channel_dim=1),
        ff_activation,
        nn.Dropout(p=dropout),
        co.Linear(ff_hidden_dim, embed_dim, channel_dim=1),
        nn.Dropout(p=dropout),
    )

    return co.Sequential(
        co.BroadcastReduce(
            RetroactiveUnity(mha.delay),
            mha,
            reduce="sum",
            auto_delay=False,
        ),
        RetroactiveLambda(
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                Residual(ff),
                nn.LayerNorm(embed_dim),
            )
        ),
    )
