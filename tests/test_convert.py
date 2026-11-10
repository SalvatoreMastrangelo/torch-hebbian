"""Adding plasticity to layers of a network that already exists."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torch_hebbian as hb
from torch_hebbian import HebbianLinear


class Agent(nn.Module):
    """A model with nested, named submodules, like a typical RL policy."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.actor = nn.Sequential(nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 3))

    def forward(self, x):
        return self.actor(torch.tanh(self.encoder(x)))


def agent():
    torch.manual_seed(0)
    return Agent()


def test_from_linear_keeps_the_function_of_the_layer():
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    x = torch.randn(4, 5)

    layer = HebbianLinear.from_linear(linear, eta=0.1, decay=0.2, w_max=4.0)

    assert torch.equal(layer(x), linear(x))
    assert (layer.eta, layer.decay, layer.w_max) == (0.1, 0.2, 4.0)


def test_from_linear_shares_the_parameters_of_the_original():
    # an optimiser built on the original parameters keeps training the new layer
    linear = nn.Linear(5, 3)

    layer = HebbianLinear.from_linear(linear)

    assert layer.weight is linear.weight
    assert layer.bias is linear.bias


def test_from_linear_without_bias():
    layer = HebbianLinear.from_linear(nn.Linear(5, 3, bias=False))

    assert layer.bias is None
    assert layer(torch.zeros(2, 5)).abs().sum() == 0


def test_from_linear_follows_dtype_of_the_original():
    layer = HebbianLinear.from_linear(nn.Linear(5, 3, dtype=torch.float64))

    assert layer.A.dtype == torch.float64
    assert layer(torch.zeros(2, 5, dtype=torch.float64)).dtype == torch.float64


def test_hebbianize_replaces_the_named_layers():
    model = agent()

    hb.hebbianize(model, {"actor.2": dict(eta=0.1, decay=0.2), "encoder": dict(eta=0.3, decay=0.4)})

    assert isinstance(model.actor[2], HebbianLinear)
    assert isinstance(model.encoder, HebbianLinear)
    assert not isinstance(model.actor[0], HebbianLinear)
    assert (model.actor[2].eta, model.actor[2].decay) == (0.1, 0.2)
    assert (model.encoder.eta, model.encoder.decay) == (0.3, 0.4)


def test_hebbianize_leaves_the_outputs_unchanged_until_rules_are_set():
    model, original = agent(), agent()
    xs = [torch.randn(5, 4) for _ in range(4)]

    hb.hebbianize(model, {"actor.2": dict(eta=0.1, decay=0.2)})

    with torch.no_grad():
        for x in xs:
            assert torch.equal(model(x), original(x))


def test_hebbianize_returns_the_same_model():
    model = agent()

    assert hb.hebbianize(model, {"encoder": {}}) is model


def test_hebbianized_model_becomes_plastic_once_rules_are_set():
    model, original = agent(), agent()
    hb.hebbianize(model, {"actor.2": dict(eta=0.1, decay=0.2)})
    hb.set_rules(model, np.random.default_rng(0).normal(size=hb.num_rules(model)))
    x = torch.randn(5, 4)

    with torch.no_grad():
        first, second = model(x), model(x)

    assert torch.equal(first, original(x))  # weights unchanged at the first step
    assert not torch.allclose(second, first)


def test_hebbianize_accepts_names_with_shared_settings():
    model = agent()

    hb.hebbianize(model, ["encoder", "actor.0"], eta=0.3, decay=0.4, w_max=2.0)

    for layer in (model.encoder, model.actor[0]):
        assert (layer.eta, layer.decay, layer.w_max) == (0.3, 0.4, 2.0)


def test_hebbianize_accepts_a_single_name():
    model = agent()

    hb.hebbianize(model, "actor.2", eta=0.3)

    assert model.actor[2].eta == 0.3
    assert [name for name, _ in hb.hebbian_layers(model)] == ["actor.2"]


def test_hebbianize_keeps_an_existing_optimiser_working():
    model = agent()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.actor[2].weight.detach().clone()

    hb.hebbianize(model, "actor.2")
    model(torch.randn(5, 4)).sum().backward()
    optimiser.step()

    assert not torch.equal(model.actor[2].weight, before)


def test_hebbianize_reports_unknown_layer_names():
    with pytest.raises(ValueError, match=r"actor\.7.*encoder"):
        hb.hebbianize(agent(), "actor.7")


def test_hebbianize_refuses_layers_that_are_not_linear():
    with pytest.raises(TypeError, match="actor.1"):
        hb.hebbianize(agent(), "actor.1")


def test_hebbianize_refuses_a_layer_that_is_already_hebbian():
    model = hb.hebbianize(agent(), "encoder")

    with pytest.raises(ValueError, match="already"):
        hb.hebbianize(model, "encoder")


def test_hebbianize_changes_nothing_when_one_name_is_wrong():
    model = agent()

    with pytest.raises(ValueError):
        hb.hebbianize(model, ["encoder", "missing"])

    assert not isinstance(model.encoder, HebbianLinear)


def test_hebbianize_rejects_shared_settings_next_to_a_dict():
    with pytest.raises(TypeError):
        hb.hebbianize(agent(), {"encoder": {}}, eta=0.1)


def test_hebbianize_changes_nothing_when_one_setting_is_invalid():
    model = agent()

    with pytest.raises(ValueError):
        hb.hebbianize(model, {"encoder": dict(eta=0.1), "actor.0": dict(decay=5.0)})

    assert not isinstance(model.encoder, HebbianLinear)
    assert not isinstance(model.actor[0], HebbianLinear)


class Tied(nn.Module):
    """The same layer registered under two names."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.head = self.layers[1]

    def forward(self, x):
        return self.head(self.layers[0](x))


@pytest.mark.parametrize("name", ["head", "layers.1"])
def test_hebbianize_replaces_every_reference_to_a_shared_layer(name):
    model = Tied()

    hb.hebbianize(model, name)

    assert isinstance(model.head, HebbianLinear)
    assert model.head is model.layers[1]


def test_hebbianize_refuses_subclasses_of_linear():
    # a subclass may compute something else in forward (or, like LazyLinear,
    # not have its weights yet); converting it would silently change the model
    model = nn.Sequential(nn.LazyLinear(3))

    with pytest.raises(TypeError, match="LazyLinear"):
        hb.hebbianize(model, "0")


def test_hebbianize_points_to_from_linear_for_the_model_itself():
    with pytest.raises(ValueError, match="from_linear"):
        hb.hebbianize(nn.Linear(3, 3), "")


def test_from_linear_keeps_the_training_mode():
    assert HebbianLinear.from_linear(nn.Linear(5, 3).eval()).training is False
    assert HebbianLinear.from_linear(nn.Linear(5, 3).train()).training is True
