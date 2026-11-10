"""Several rule sets in one batch: a ``(K, num_rules)`` array gives rule set k
to the k-th contiguous block of batch rows."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torch_hebbian as hb
from helpers import make_layer
from reference_hebbian import HebbianLastLayer
from torch_hebbian import HebbianLinear


def model():
    torch.manual_seed(0)
    return nn.Sequential(
        HebbianLinear(3, 4, eta=0.1, decay=0.1),
        nn.Tanh(),
        HebbianLinear(4, 2, eta=0.1, decay=0.1),
    )


N_RULES = 4 * 4 * 3 + 4 * 2 * 4  # 80


def rollout(net, inputs):
    hb.reset_state(net)
    with torch.no_grad():
        return torch.stack([net(x) for x in inputs])


def test_blocks_of_rows_follow_their_own_rule_set():
    # D-only rules on w0 = 2 with eta = 0.5, no decay: rule set k drifts by 0.5*D_k.
    # Vector layout for a 1x1 layer is [A, B, C, D].
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    hb.set_rules(layer, [[0, 0, 0, 2.0], [0, 0, 0, 4.0], [0, 0, 0, -2.0]])

    layer(torch.ones(6, 1))
    second = layer(torch.ones(6, 1))

    assert second.flatten().tolist() == [3.0, 3.0, 4.0, 4.0, 1.0, 1.0]


def test_population_equals_separate_runs_one_rule_set_at_a_time():
    K, S, steps = 3, 4, 6
    population = np.random.default_rng(0).normal(size=(K, N_RULES))
    generator = torch.Generator().manual_seed(1)
    inputs = [torch.randn(K * S, 3, generator=generator) for _ in range(steps)]

    together = model()
    hb.set_rules(together, population)
    batched = rollout(together, inputs)  # (steps, K*S, 2)

    for k in range(K):
        alone = model()
        hb.set_rules(alone, population[k])
        rows = slice(k * S, (k + 1) * S)
        expected = rollout(alone, [x[rows] for x in inputs])
        assert torch.allclose(batched[:, rows], expected, atol=1e-6)


def test_population_matches_reference_with_per_env_rules():
    K, S, OUT, IN, steps = 3, 2, 3, 5, 20
    torch.manual_seed(0)
    layer = HebbianLinear(IN, OUT, eta=0.05, decay=0.1, dtype=torch.float64)
    population = torch.randn(K, 4, OUT, IN, dtype=torch.float64)
    hb.set_rules(layer, population.reshape(K, -1))
    per_env = population.repeat_interleave(S, dim=0)  # (K*S, 4, out, in)
    reference = HebbianLastLayer(
        layer.weight,
        {
            "A": per_env[:, 0], "B": per_env[:, 1], "C": per_env[:, 2], "D": per_env[:, 3],
            "lam": torch.full((OUT, IN), 0.1, dtype=torch.float64),
        },
        eta=0.05, w_max=1e12, use_oja_coefficient=False, num_envs=K * S,
    )

    for _ in range(steps):
        x = torch.randn(K * S, IN, dtype=torch.float64)
        expected = torch.einsum("ni,noi->no", x, reference.W) + layer.bias.detach()
        reference.hebbian_update(x, expected)
        with torch.no_grad():
            assert torch.allclose(layer(x), expected, rtol=1e-10, atol=1e-12)


def test_batch_must_split_evenly_among_the_rule_sets():
    layer = HebbianLinear(2, 2)
    hb.set_rules(layer, np.zeros((3, 16)))

    with pytest.raises(RuntimeError, match="3 rule sets"):
        layer(torch.zeros(4, 2))


def test_the_first_batch_dimension_is_the_one_that_is_split():
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    hb.set_rules(layer, [[0, 0, 0, 2.0], [0, 0, 0, 4.0]])

    layer(torch.ones(2, 3, 1))
    second = layer(torch.ones(2, 3, 1))

    assert second.squeeze(-1).tolist() == [[3.0] * 3, [4.0] * 3]


def test_get_rules_returns_the_whole_population():
    net = model()
    population = np.random.default_rng(0).normal(size=(5, N_RULES)).astype(np.float32)

    hb.set_rules(net, population)

    assert np.array_equal(hb.get_rules(net), population)


def test_a_list_of_vectors_is_a_population():
    # pycma's ask() returns a list of 1-D arrays
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    hb.set_rules(layer, [np.array([0, 0, 0, 2.0]), np.array([0, 0, 0, 4.0])])

    layer(torch.ones(2, 1))

    assert layer(torch.ones(2, 1)).flatten().tolist() == [3.0, 4.0]


def test_a_flat_vector_replaces_the_population():
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    hb.set_rules(layer, [[0, 0, 0, 2.0], [0, 0, 0, 4.0]])

    hb.set_rules(layer, [0, 0, 0, 8.0])
    hb.reset_state(layer)
    layer(torch.ones(3, 1))  # 3 rows: would not split in 2 rule sets

    assert layer(torch.ones(3, 1)).flatten().tolist() == [6.0, 6.0, 6.0]
    assert hb.get_rules(layer).tolist() == [0, 0, 0, 8.0]


def test_population_is_not_part_of_the_state_dict():
    layer = HebbianLinear(2, 2)
    hb.set_rules(layer, np.ones((3, 16)))

    assert set(layer.state_dict()) == {"weight", "bias", "A", "B", "C", "D"}


def test_population_rules_are_saved_as_a_2d_array(tmp_path):
    net = model()
    population = np.random.default_rng(0).normal(size=(5, N_RULES)).astype(np.float32)
    hb.set_rules(net, population)

    hb.save_rules(net, tmp_path / "population.npy")
    other = model()
    hb.load_rules(other, tmp_path / "population.npy")

    assert np.array_equal(hb.get_rules(other), population)


def test_loading_a_state_dict_with_rules_replaces_an_active_population():
    # Rules loaded through load_state_dict must be the ones that act, even if
    # a population from an earlier evaluation is still held by the layer.
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    hb.set_rules(layer, [[0, 0, 0, 2.0], [0, 0, 0, 4.0]])
    state = layer.state_dict()
    state["D"] = torch.tensor([[8.0]])

    layer.load_state_dict(state)
    hb.reset_state(layer)
    layer(torch.ones(3, 1))

    assert layer(torch.ones(3, 1)).flatten().tolist() == [6.0, 6.0, 6.0]


def test_loading_a_state_dict_without_rules_keeps_the_population():
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    hb.set_rules(layer, [[0, 0, 0, 2.0], [0, 0, 0, 4.0]])

    layer.load_state_dict({"weight": torch.tensor([[2.0]])}, strict=False)
    layer(torch.ones(2, 1))

    assert layer(torch.ones(2, 1)).flatten().tolist() == [3.0, 4.0]


def test_repr_shows_an_active_population():
    layer = HebbianLinear(2, 2)
    assert "population" not in repr(layer)

    hb.set_rules(layer, np.zeros((3, 16)))

    assert "population=3" in repr(layer)


def test_an_empty_population_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        hb.set_rules(HebbianLinear(2, 2), np.zeros((0, 16)))
