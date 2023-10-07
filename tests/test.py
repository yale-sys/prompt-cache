import torch
import math

def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8, device='cpu'):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32, device=device).view(1, 1, 1, sequence_length)
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.float32, device=device)
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:num_heads]

    alibi = alibi * slopes
    return alibi.squeeze(0)


table = build_mpt_alibi_tensor(16, 20)
print(table.shape)
print(table)