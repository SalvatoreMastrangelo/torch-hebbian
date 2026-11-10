"""Shared helpers for the test-suite."""

from __future__ import annotations

import torch

from torch_hebbian import HebbianLinear


def make_layer(weight, bias=None, **kwargs) -> HebbianLinear:
    """A ``HebbianLinear`` with hand-picked weights (``weight`` is ``(out, in)``)."""
    weight = torch.as_tensor(weight, dtype=torch.float32)
    out_features, in_features = weight.shape
    layer = HebbianLinear(in_features, out_features, bias=bias is not None, **kwargs)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return layer


def set_rule(layer: HebbianLinear, name: str, value) -> None:
    """Overwrite one of the rule tensors ``A``/``B``/``C``/``D`` in place."""
    with torch.no_grad():
        getattr(layer, name).copy_(torch.as_tensor(value, dtype=layer.weight.dtype))


def rollout(layer, inputs):
    """Feed ``inputs`` one after the other; return the list of outputs."""
    return [layer(torch.as_tensor(x, dtype=layer.weight.dtype)) for x in inputs]
