"""Model-level helpers: every function takes any ``nn.Module`` and acts on all
the :class:`~torch_hebbian.HebbianLinear` layers found inside it (a single
``HebbianLinear`` is a valid argument too).

The rules of a model form one flat vector: the layers in ``model.modules()``
order, each contributing ``A | B | C | D`` flattened row-major. That vector is
what an evolutionary algorithm optimises and what the ``.npy`` files contain.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from .layer import RULE_NAMES, HebbianLinear

PathLike = Union[str, os.PathLike]


def hebbian_layers(model: nn.Module) -> List[Tuple[str, HebbianLinear]]:
    """``(name, layer)`` for every Hebbian layer, in rule-vector order."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, HebbianLinear)]


def hebbianize(
    model: nn.Module,
    layers: Union[str, Iterable[str], Mapping],
    **settings,
) -> nn.Module:
    """Replace ``nn.Linear`` layers of an existing model by ``HebbianLinear``.

    The model is modified in place and returned. The new layers reuse the
    weight and bias parameters of the layers they replace and start with zero
    rules, so the model computes exactly what it did before until rules are
    set or trained.

    Args:
        layers: which layers, by their name in ``model.named_modules()``.
            Either one name or several, all configured by ``settings``; or a
            dict ``{name: {"eta": ..., "decay": ..., "w_max": ...}}`` to
            configure each layer on its own.
        **settings: ``eta``, ``decay``, ``w_max`` shared by all named layers.
    """
    if isinstance(layers, str):
        layers = [layers]
    if isinstance(layers, Mapping):
        if settings:
            raise TypeError(
                "give the settings either per layer (dict) or as keyword arguments, not both"
            )
        plan = dict(layers)
    else:
        plan = {name: settings for name in layers}

    modules = dict(model.named_modules())
    for name in plan:  # check everything before touching the model
        module = modules.get(name) if name else None
        if module is None:
            linear = [n for n, m in modules.items() if n and isinstance(m, nn.Linear)]
            raise ValueError(f"the model has no layer named {name!r}; its linear layers are {linear}")
        if isinstance(module, HebbianLinear):
            raise ValueError(f"layer {name!r} is already a HebbianLinear")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"layer {name!r} is a {type(module).__name__}, not an nn.Linear")

    for name, layer_settings in plan.items():
        parent_name, _, child_name = name.rpartition(".")
        replacement = HebbianLinear.from_linear(modules[name], **layer_settings)
        setattr(modules[parent_name], child_name, replacement)
    return model


def _require_layers(model: nn.Module) -> List[HebbianLinear]:
    layers = [layer for _, layer in hebbian_layers(model)]
    if not layers:
        raise ValueError("the model has no HebbianLinear layers")
    return layers


def num_rules(model: nn.Module) -> int:
    """Length of the model's rule vector."""
    return sum(layer.num_rules for _, layer in hebbian_layers(model))


def rule_slices(model: nn.Module) -> Dict[str, slice]:
    """Where each layer's rules sit in the rule vector, keyed by module name."""
    slices, start = {}, 0
    for name, layer in hebbian_layers(model):
        slices[name] = slice(start, start + layer.num_rules)
        start += layer.num_rules
    return slices


def rule_parameters(model: nn.Module) -> List[nn.Parameter]:
    """The ``A, B, C, D`` parameters of every layer, e.g. to hand to an optimiser."""
    return [getattr(layer, name) for _, layer in hebbian_layers(model) for name in RULE_NAMES]


def get_rules(model: nn.Module) -> np.ndarray:
    """The model's rules as an array of shape ``(num_rules,)``.

    While a population is loaded (see :func:`set_rules`) the result has shape
    ``(K, num_rules)``.
    """
    parts = [layer._read_rules().cpu().numpy() for layer in _require_layers(model)]
    if len({part.shape[:-1] for part in parts}) > 1:
        raise ValueError("the layers of this model hold different numbers of rule sets")
    return np.concatenate(parts, axis=-1)


def set_rules(model: nn.Module, rules, *, freeze: bool = True) -> None:
    """Write rules into the model.

    Args:
        rules: array-like, copied. Shape ``(num_rules,)`` sets the rules of the
            model. Shape ``(K, num_rules)``, or a list of K vectors, loads a
            population of K rule sets evaluated side by side: the first batch
            dimension is split in K contiguous blocks and block k uses rule
            set k. A population stays active until a single vector is set.
        freeze: set ``requires_grad=False`` on the rule parameters, so that
            optimisers leave them alone. Other parameters are not touched.

    The plastic state is left as it is: call :func:`reset_state` before the
    next episode.
    """
    layers = _require_layers(model)
    total = sum(layer.num_rules for layer in layers)
    if isinstance(rules, torch.Tensor):
        rules = rules.detach().cpu().numpy()
    rules = np.array(rules, dtype=np.float64)  # always a copy
    if rules.ndim not in (1, 2) or rules.shape[-1] != total:
        raise ValueError(
            f"expected rules of shape ({total},), or (K, {total}) for K rule sets, "
            f"got {rules.shape}"
        )
    start = 0
    for layer in layers:
        layer._write_rules(torch.from_numpy(rules[..., start:start + layer.num_rules]))
        start += layer.num_rules
    if freeze:
        for parameter in rule_parameters(model):
            parameter.requires_grad_(False)


def save_rules(model: nn.Module, path: PathLike) -> None:
    """Save the model's rule vector to a ``.npy`` file."""
    np.save(path, get_rules(model))


def load_rules(model: nn.Module, path: PathLike, *, freeze: bool = True) -> None:
    """Load a rule vector from a ``.npy`` file; see :func:`set_rules`."""
    _require_layers(model)
    set_rules(model, np.load(path, allow_pickle=False), freeze=freeze)


def reset_state(model: nn.Module, rows=None) -> None:
    """Put the plastic weights of every layer back to the layer's ``weight``.

    Call it at the start of every episode. ``rows`` restricts the reset to some
    entries of the first batch dimension (see ``HebbianLinear.reset_state``).
    """
    for _, layer in hebbian_layers(model):
        layer.reset_state(rows)


def detach_state(model: nn.Module) -> None:
    """Cut the autograd history of every layer's plastic weights."""
    for _, layer in hebbian_layers(model):
        layer.detach_state()
