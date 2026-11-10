"""``HebbianLinear`` must reproduce, step by step, the implementation it was
extracted from (``reference_hebbian.HebbianLastLayer`` with the Oja option off)."""

from __future__ import annotations

import pytest
import torch

from reference_hebbian import HebbianLastLayer
from torch_hebbian import HebbianLinear

OUT, IN, ROWS, STEPS = 3, 5, 6, 25
ETA, DECAY = 0.05, 0.1


def reference_step(reference: HebbianLastLayer, bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """One control step exactly as ``IsolatedPopulationActor.act`` performs it."""
    y = torch.einsum("ni,noi->no", x, reference.W) + bias.unsqueeze(0)
    reference.hebbian_update(x, y)
    return y


def random_rules(*shape: int) -> dict:
    return {name: torch.randn(*shape, OUT, IN, dtype=torch.float64) for name in "ABCD"}


@pytest.mark.parametrize("w_max", [None, 0.6], ids=["unclipped", "clipped"])
def test_single_rule_set_matches_reference(w_max):
    torch.manual_seed(0)
    layer = HebbianLinear(IN, OUT, eta=ETA, decay=DECAY, w_max=w_max, dtype=torch.float64)
    rules = random_rules()
    with torch.no_grad():
        for name, value in rules.items():
            getattr(layer, name).copy_(value)
    reference = HebbianLastLayer(
        layer.weight,
        {**rules, "lam": torch.full((OUT, IN), DECAY, dtype=torch.float64)},
        eta=ETA,
        w_max=1e12 if w_max is None else w_max,
        use_oja_coefficient=False,
        num_envs=ROWS,
    )

    for _ in range(STEPS):
        x = torch.randn(ROWS, IN, dtype=torch.float64)
        expected = reference_step(reference, layer.bias.detach(), x)
        assert torch.allclose(layer(x), expected, rtol=1e-10, atol=1e-12)
        assert torch.allclose(layer.plastic_weight, reference.W, rtol=1e-10, atol=1e-12)

    # the rollout must have been a real test: weights moved, and clipping bit
    assert (reference.W - layer.weight).abs().max() > 0.1
    if w_max is not None:
        assert (reference.W.abs() == w_max).any()
