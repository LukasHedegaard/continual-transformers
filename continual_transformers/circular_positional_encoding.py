import continual as co
import torch
from torch import Tensor, nn


class CircularPositionalEncoding(co.CoModule, nn.Module):
    def __init__(self, embed_dim: int, num_embeds: int, forward_update_index_steps=1):
        nn.Module.__init__(self)
        self.pe = nn.Embedding(num_embeds, embed_dim)
        self.index = 0
        self.forward_update_index_steps = forward_update_index_steps

    def forward(self, input: Tensor, update_index_steps: int = None) -> Tensor:
        T = input.shape[2]
        assert T <= self.pe.num_embeddings
        position_ids = (
            torch.arange(T, device=self.pe.weight.device).unsqueeze(0) + self.index
        ) % self.pe.num_embeddings

        index_update = (
            self.forward_update_index_steps
            if update_index_steps is None
            else update_index_steps
        )
        self.index = (self.index + index_update) % self.pe.num_embeddings

        position_embeddings = self.pe(position_ids).transpose(1, 2)
        return input + position_embeddings

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        return self.forward(
            input, update_index_steps=input.shape[2] if update_state else 0
        )

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        output = input + self.pe(torch.tensor([self.index]))

        if update_state:
            self.index = (self.index + 1) % self.pe.num_embeddings
        return output

    def clean_state(self):
        """Clean model state"""
        self.index = 0
