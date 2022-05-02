import continual as co
import torch
from torch import nn

from continual_transformers import (
    CoReSiTransformerEncoder,
    CoSiTransformerEncoder,
    RecyclingPositionalEncoding,
)


def CoTransformerModel(
    embed_dim,
    depth,
    heads,
    mlp_dim,
    dropout_rate=0.1,
    sequence_len=64,
):
    assert depth in {1, 2}

    if depth == 1:
        return CoSiTransformerEncoder(
            sequence_len=sequence_len,
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=dropout_rate,
            in_proj_bias=False,
            query_index=-1,
            ff_hidden_dim=mlp_dim,
            ff_activation=nn.GELU(),
            device=None,
            dtype=None,
        )

    # depth == 2
    return CoReSiTransformerEncoder(
        sequence_len=sequence_len,
        embed_dim=embed_dim,
        num_heads=heads,
        dropout=dropout_rate,
        in_proj_bias=False,
        query_index=-1,
        ff_hidden_dim=mlp_dim,
        ff_activation=nn.GELU(),
        device=None,
        dtype=None,
    )


def CoVisionTransformer(
    sequence_len,
    input_dim,
    embedding_dim,
    attn_ff_hidden_dim,
    out_dim,
    num_heads,
    num_layers,
    dropout_rate=0.1,
):

    assert embedding_dim % num_heads == 0

    linear_encoding = co.Linear(input_dim, embedding_dim, channel_dim=1)
    position_encoding = RecyclingPositionalEncoding(
        embedding_dim,
        int(embedding_dim * 1.0),  # Change num pos enc to cycle between
        forward_update_index_steps=1,
    )

    pe_dropout = nn.Dropout(p=dropout_rate)

    encoder = CoTransformerModel(
        embedding_dim,
        num_layers,
        num_heads,
        attn_ff_hidden_dim,
        dropout_rate,
        sequence_len,
    )
    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(embedding_dim, out_dim, channel_dim=1)

    return co.Sequential(
        linear_encoding,
        position_encoding,
        pe_dropout,
        encoder,
        pre_head_ln,
        mlp_head,
    )


if __name__ == "__main__":
    SEQ_LEN = 120
    INPUT_DIM = 128

    model = CoVisionTransformer(
        sequence_len=SEQ_LEN,
        input_dim=INPUT_DIM,
        embedding_dim=192,
        attn_ff_hidden_dim=192,
        out_dim=10,
        num_heads=16,
        num_layers=1,
        dropout_rate=0.1,
    )

    try:
        from ptflops import get_model_complexity_info

        # Compute flops for regular inference mode
        flops, params = get_model_complexity_info(
            model, (INPUT_DIM, SEQ_LEN), as_strings=False, print_per_layer_stat=False
        )
        print(f"Model Params: {params}")
        print(f"Regular Model FLOPs: {flops}")

        # Compute flops for continual inference mode
        model.forward_steps(torch.randn(1, INPUT_DIM, SEQ_LEN))  # Warm up model
        model.call_mode = "forward_step"

        co_flops, _ = get_model_complexity_info(
            model, (INPUT_DIM,), as_strings=False, print_per_layer_stat=False
        )
        print(f"Continual Model FLOPs: {flops}")
        print(f"Reg/Co: {flops / co_flops}x")

    except Exception as e:
        print(e)
        ...
