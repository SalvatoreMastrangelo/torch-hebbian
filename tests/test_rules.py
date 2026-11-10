"""Reading and writing the rules of a model as one flat vector / ``.npy`` file."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torch_hebbian as hb
from torch_hebbian import HebbianLinear


def two_layer_model():
    torch.manual_seed(0)
    return nn.Sequential(
        HebbianLinear(3, 2, eta=0.1, decay=0.1),  # 4*2*3 = 24 rules
        nn.Tanh(),
        nn.Linear(2, 2),
        HebbianLinear(2, 1, eta=0.1, decay=0.1),  # 4*1*2 = 8 rules
    )


def run(model, steps=4, rows=5, seed=1):
    """Outputs of a short rollout on fixed inputs, starting from a reset state."""
    hb.reset_state(model)
    generator = torch.Generator().manual_seed(seed)
    in_features = next(model.parameters()).shape[1]
    with torch.no_grad():
        return torch.stack(
            [model(torch.randn(rows, in_features, generator=generator)) for _ in range(steps)]
        )


def test_num_rules_counts_four_values_per_weight():
    assert hb.num_rules(HebbianLinear(3, 2)) == 24
    assert hb.num_rules(two_layer_model()) == 32


def test_num_rules_of_a_model_without_hebbian_layers_is_zero():
    assert hb.num_rules(nn.Linear(3, 2)) == 0


def test_vector_layout_is_A_B_C_D_each_row_major():
    layer = HebbianLinear(3, 2)

    hb.set_rules(layer, np.arange(24.0))

    assert layer.A.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert layer.B.tolist() == [[6, 7, 8], [9, 10, 11]]
    assert layer.C.tolist() == [[12, 13, 14], [15, 16, 17]]
    assert layer.D.tolist() == [[18, 19, 20], [21, 22, 23]]


def test_layers_are_concatenated_in_module_order():
    model = two_layer_model()

    hb.set_rules(model, np.arange(32.0))

    assert model[0].A.flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert model[0].D.flatten().tolist() == [18, 19, 20, 21, 22, 23]
    assert model[3].A.tolist() == [[24, 25]]
    assert model[3].D.tolist() == [[30, 31]]


def test_rule_slices_locate_each_layer_in_the_vector():
    assert hb.rule_slices(two_layer_model()) == {"0": slice(0, 24), "3": slice(24, 32)}


def test_get_rules_returns_what_set_rules_stored():
    model = two_layer_model()
    vector = np.linspace(-1.0, 1.0, 32).astype(np.float32)

    hb.set_rules(model, vector)

    assert np.array_equal(hb.get_rules(model), vector)


def test_get_rules_reads_the_parameters():
    layer = HebbianLinear(1, 1)
    with torch.no_grad():
        layer.A.fill_(1.0)
        layer.B.fill_(2.0)
        layer.C.fill_(3.0)
        layer.D.fill_(4.0)

    assert hb.get_rules(layer).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_set_rules_freezes_the_rules_and_only_the_rules():
    model = two_layer_model()

    hb.set_rules(model, np.zeros(32))

    assert [p.requires_grad for p in hb.rule_parameters(model)] == [False] * 8
    assert all(p.requires_grad for name, p in model.named_parameters() if name[-1] not in "ABCD")


def test_set_rules_can_leave_the_rules_trainable():
    model = two_layer_model()

    hb.set_rules(model, np.zeros(32), freeze=False)

    assert [p.requires_grad for p in hb.rule_parameters(model)] == [True] * 8


def test_rule_parameters_are_the_abcd_tensors_in_vector_order():
    model = two_layer_model()

    params = hb.rule_parameters(model)

    expected = [model[0].A, model[0].B, model[0].C, model[0].D,
                model[3].A, model[3].B, model[3].C, model[3].D]
    assert len(params) == 8
    assert all(p is e for p, e in zip(params, expected))


def test_set_rules_copies_its_argument():
    layer = HebbianLinear(1, 1)
    vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    hb.set_rules(layer, vector)
    vector[:] = 99.0

    assert hb.get_rules(layer).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_set_rules_accepts_tensors_and_lists():
    layer = HebbianLinear(1, 1)

    hb.set_rules(layer, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert hb.get_rules(layer).tolist() == [1.0, 2.0, 3.0, 4.0]

    hb.set_rules(layer, [5.0, 6.0, 7.0, 8.0])
    assert hb.get_rules(layer).tolist() == [5.0, 6.0, 7.0, 8.0]


@pytest.mark.parametrize("shape", [(31,), (33,), (2, 31), (2, 2, 32), ()])
def test_set_rules_rejects_a_vector_of_the_wrong_shape(shape):
    with pytest.raises(ValueError, match="32"):
        hb.set_rules(two_layer_model(), np.zeros(shape))


@pytest.mark.parametrize("call", [
    lambda m: hb.set_rules(m, np.zeros(0)),
    lambda m: hb.get_rules(m),
    lambda m: hb.save_rules(m, "unused.npy"),
    lambda m: hb.load_rules(m, "unused.npy"),
])
def test_rule_functions_refuse_a_model_without_hebbian_layers(call):
    with pytest.raises(ValueError, match="no HebbianLinear"):
        call(nn.Sequential(nn.Linear(3, 2)))


def test_rules_survive_a_round_trip_through_an_npy_file(tmp_path):
    source = two_layer_model()
    hb.set_rules(source, np.random.default_rng(0).normal(size=32))
    path = tmp_path / "rules.npy"

    hb.save_rules(source, path)
    target = two_layer_model()  # same seed -> same weights, zero rules
    hb.load_rules(target, path)

    assert np.load(path).shape == (32,)
    assert torch.equal(run(target), run(source))
    assert not torch.equal(run(target), run(two_layer_model()))  # the rules matter


def test_load_rules_freezes_the_rules(tmp_path):
    model = two_layer_model()
    path = tmp_path / "rules.npy"
    np.save(path, np.zeros(32))

    hb.load_rules(model, path)

    assert not any(p.requires_grad for p in hb.rule_parameters(model))


def test_loaded_rules_receive_no_gradient_but_the_weights_do(tmp_path):
    model = two_layer_model()
    path = tmp_path / "rules.npy"
    np.save(path, np.random.default_rng(0).normal(size=32))
    hb.load_rules(model, path)

    for _ in range(3):
        loss = model(torch.randn(4, 3)).pow(2).sum()
    loss.backward()

    assert all(p.grad is None for p in hb.rule_parameters(model))
    assert model[0].weight.grad.abs().sum() > 0


def test_reset_state_and_detach_state_reach_every_hebbian_layer():
    model = two_layer_model()
    hb.set_rules(model, np.random.default_rng(0).normal(size=32), freeze=False)
    model(torch.randn(4, 3))

    hb.detach_state(model)
    assert not model[0].weight_delta.requires_grad
    assert not model[3].weight_delta.requires_grad

    hb.reset_state(model)
    assert model[0].weight_delta is None
    assert model[3].weight_delta is None


def test_reset_state_passes_rows_to_the_layers():
    model = two_layer_model()
    hb.set_rules(model, np.random.default_rng(0).normal(size=32))
    model(torch.randn(4, 3))

    hb.reset_state(model, rows=[1])

    for layer in (model[0], model[3]):
        moved = layer.weight_delta.flatten(1).abs().sum(1) > 0
        assert moved.tolist() == [True, False, True, True]
