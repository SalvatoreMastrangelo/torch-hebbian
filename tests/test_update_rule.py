"""The plastic update: dW = eta*(A*y*x + B*x + C*y + D), with decay towards
the layer's own ``weight``. Expected numbers are worked out by hand."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from helpers import make_layer, rollout, set_rule
from torch_hebbian import HebbianLinear


def test_zero_rules_reproduce_nn_linear_at_every_step():
    torch.manual_seed(0)
    plain = nn.Linear(5, 3)
    layer = HebbianLinear(5, 3, eta=0.3, decay=0.1)
    layer.load_state_dict(plain.state_dict(), strict=False)

    for _ in range(4):
        x = torch.randn(6, 5)
        assert torch.equal(layer(x), plain(x))


def test_is_an_nn_linear():
    assert isinstance(HebbianLinear(5, 3), nn.Linear)


def test_hebb_term_feeds_back_through_the_output():
    # w0 = 2, A = 1, eta = 0.5, decay = 0.25
    # step 1: y = 2*1 = 2      dW = 0.5*2*1   = 1     W = .75*2    + .25*2 + 1    = 3
    # step 2: y = 3*1 = 3      dW = 0.5*3*1   = 1.5   W = .75*3    + .25*2 + 1.5  = 4.25
    # step 3: y = 4.25*2 = 8.5 dW = 0.5*8.5*2 = 8.5   W = .75*4.25 + .25*2 + 8.5  = 12.1875
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)
    set_rule(layer, "A", [[1.0]])

    outputs = rollout(layer, [[1.0], [1.0], [2.0]])

    assert [y.item() for y in outputs] == [2.0, 3.0, 8.5]
    assert layer.plastic_weight.item() == 12.1875


# weight [[1, -1]], bias 0.5, eta 0.1, no decay; first input [2, 3] gives y = -0.5.
# Second input [1, 1] reads the updated weights: y = W[0] + W[1] + 0.5.
@pytest.mark.parametrize(
    "rule, dw, second_output",
    [
        # A: 0.1*[10*(-0.5)*2, 20*(-0.5)*3] = [-1, -3]   -> W = [0, -4]
        ("A", [-1.0, -3.0], -3.5),
        # B: 0.1*[10*2, 20*3] = [2, 6]                    -> W = [3, 5]
        ("B", [2.0, 6.0], 8.5),
        # C: 0.1*[10*(-0.5), 20*(-0.5)] = [-0.5, -1]      -> W = [0.5, -2]
        ("C", [-0.5, -1.0], -1.0),
        # D: 0.1*[10, 20] = [1, 2]                        -> W = [2, 1]
        ("D", [1.0, 2.0], 3.5),
    ],
)
def test_each_rule_term_alone(rule, dw, second_output):
    layer = make_layer([[1.0, -1.0]], bias=[0.5], eta=0.1, decay=0.0)
    set_rule(layer, rule, [[10.0, 20.0]])

    first = layer(torch.tensor([2.0, 3.0]))
    delta_after_first = layer.weight_delta
    second = layer(torch.tensor([1.0, 1.0]))

    assert first.item() == pytest.approx(-0.5)
    assert delta_after_first.flatten().tolist() == pytest.approx(dw)
    assert second.item() == pytest.approx(second_output)


def test_post_synaptic_term_is_indexed_by_output_neuron():
    # two outputs, one input: weight [[1], [2]], x = 3 -> y = [3, 6]
    # C = [[1], [10]], eta = 0.1: dW = 0.1*[1*3, 10*6] = [0.3, 6] -> W = [1.3, 8]
    layer = make_layer([[1.0], [2.0]], eta=0.1, decay=0.0)
    set_rule(layer, "C", [[1.0], [10.0]])

    _, second = rollout(layer, [[3.0], [1.0]])

    assert second.tolist() == pytest.approx([1.3, 8.0])


# w0 = 2, D = 4, eta = 0.5: the first step moves W to 4. D is then switched
# off, so only the decay acts: W <- (1 - decay)*W + decay*2.
@pytest.mark.parametrize(
    "decay, outputs",
    [
        (0.5, [2.0, 4.0, 3.0, 2.5]),  # halves the distance to 2 each step
        (0.0, [2.0, 4.0, 4.0, 4.0]),  # no decay: the change is permanent
        (1.0, [2.0, 4.0, 2.0, 2.0]),  # full decay: back to 2 in one step
    ],
)
def test_decay_pulls_the_weight_back_to_the_original_value(decay, outputs):
    layer = make_layer([[2.0]], eta=0.5, decay=decay)
    set_rule(layer, "D", [[4.0]])

    seen = [layer(torch.ones(1)).item()]
    set_rule(layer, "D", [[0.0]])
    seen += [layer(torch.ones(1)).item() for _ in range(3)]

    assert seen == outputs


def test_plasticity_never_touches_the_layer_parameters():
    torch.manual_seed(0)
    layer = HebbianLinear(4, 3, eta=0.5, decay=0.1)
    for name in "ABCD":
        set_rule(layer, name, torch.randn(3, 4))
    weight, bias = layer.weight.clone(), layer.bias.clone()

    for _ in range(5):
        layer(torch.randn(2, 4))

    assert torch.equal(layer.weight, weight)
    assert torch.equal(layer.bias, bias)


def test_w_max_clips_the_plastic_weight():
    # w0 = 2, D = 4, eta = 0.5 would give W = 4; w_max = 3.5 clips it.
    layer = make_layer([[2.0]], eta=0.5, decay=0.0, w_max=3.5)
    set_rule(layer, "D", [[4.0]])

    _, second = rollout(layer, [[1.0], [1.0]])

    assert second.item() == 3.5


def test_w_max_is_symmetric():
    layer = make_layer([[2.0]], eta=0.5, decay=0.0, w_max=3.5)
    set_rule(layer, "D", [[-20.0]])

    _, second = rollout(layer, [[1.0], [1.0]])

    assert second.item() == -3.5


@pytest.mark.parametrize(
    "kwargs",
    [{"decay": -0.1}, {"decay": 1.5}, {"w_max": 0.0}, {"w_max": -1.0}],
)
def test_rejects_meaningless_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        HebbianLinear(2, 2, **kwargs)


def test_decay_and_w_max_are_validated_when_assigned_later():
    layer = HebbianLinear(2, 2)

    with pytest.raises(ValueError):
        layer.decay = 3.0
    with pytest.raises(ValueError):
        layer.w_max = -1.0

    assert layer.decay == 0.05 and layer.w_max is None  # unchanged


def test_hyperparameters_can_be_changed_after_construction():
    # w0 = 2, D = 4, eta = 0.5: one step gives W = 4. Then decay = 1 and eta = 0
    # bring the weight straight back.
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    set_rule(layer, "D", [[4.0]])
    layer(torch.ones(1))

    layer.decay, layer.eta = 1.0, 0.0
    outputs = rollout(layer, [[1.0], [1.0]])

    assert [y.item() for y in outputs] == [4.0, 2.0]


def test_plastic_weight_is_a_copy_before_and_after_the_first_step():
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)

    before = layer.plastic_weight
    with torch.no_grad():
        before.fill_(99.0)

    assert layer.weight.item() == 2.0
