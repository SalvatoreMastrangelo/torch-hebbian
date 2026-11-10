"""Rules left trainable are ordinary parameters: gradients flow through the
plastic weights from one forward pass to the next."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torch_hebbian as hb
from helpers import make_layer, set_rule
from torch_hebbian import HebbianLinear


def drifting_layer():
    """w0 = 2, constant drift of +2 per step; fed with ones: 2, 4, 6, ..."""
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    set_rule(layer, "D", [[4.0]])
    return layer


def test_rules_are_trainable_parameters_by_default():
    layer = HebbianLinear(3, 2)

    names = {name for name, p in layer.named_parameters() if p.requires_grad}

    assert names == {"weight", "bias", "A", "B", "C", "D"}


def test_gradient_of_a_later_output_reaches_the_rules():
    # w0 = 2, x = 1, eta = 0.5: y1 = 2, W = 2 + 0.5*(A*2 + B + C*2 + D), y2 = W
    # so dy2/dA = 1, dy2/dB = 0.5, dy2/dC = 1, dy2/dD = 0.5
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)

    layer(torch.ones(1))
    layer(torch.ones(1)).sum().backward()

    assert layer.A.grad.item() == 1.0
    assert layer.B.grad.item() == 0.5
    assert layer.C.grad.item() == 1.0
    assert layer.D.grad.item() == 0.5


@pytest.mark.parametrize("w_max", [None, 0.8], ids=["unclipped", "clipped"])
def test_analytic_gradients_match_finite_differences(w_max):
    torch.manual_seed(0)
    layer = HebbianLinear(3, 2, eta=0.3, decay=0.2, w_max=w_max, dtype=torch.float64)
    params = {
        name: torch.randn_like(p).requires_grad_() for name, p in layer.named_parameters()
    }
    inputs = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)  # 4 steps, 2 rows

    def rollout(inputs, *values):
        layer.reset_state()
        named = dict(zip(params, values))
        return torch.stack([torch.func.functional_call(layer, named, (x,)) for x in inputs])

    assert torch.autograd.gradcheck(rollout, (inputs, *params.values()))


def test_detach_state_stops_the_gradient_at_that_point():
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)

    layer(torch.ones(1))   # the update of this step depends on the rules...
    layer.detach_state()   # ...but its history is cut here
    layer(torch.ones(1)).sum().backward()

    assert layer.A.grad is None
    assert layer.weight.grad.item() == 1.0  # y2 = (w0 + delta) * 1 still depends on w0


def test_no_grad_rollouts_build_no_graph():
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)
    set_rule(layer, "A", [[1.0]])

    with torch.no_grad():
        for _ in range(3):
            layer(torch.ones(1))

    assert not layer.weight_delta.requires_grad


def test_a_step_without_grad_cuts_the_history():
    # Step 1 makes the state depend on the rules. Step 2 runs under no_grad,
    # so the state it leaves is a constant. y3 = (w0 + state) * 1 then depends
    # on w0 only.
    layer = make_layer([[2.0]], eta=0.5, decay=0.5)
    layer(torch.ones(1))
    with torch.no_grad():
        layer(torch.ones(1))

    layer(torch.ones(1)).sum().backward()

    assert layer.A.grad is None
    assert layer.weight.grad.item() == 1.0


def test_rollouts_with_and_without_grad_give_the_same_numbers():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 6), nn.Tanh(), HebbianLinear(6, 3, eta=0.2, decay=0.1, w_max=1.5)
    )
    for name in "ABCD":
        set_rule(model[2], name, torch.randn(3, 6))
    inputs = [torch.randn(5, 4) for _ in range(8)]

    with_grad = torch.stack([model(x) for x in inputs]).detach()
    hb.reset_state(model)
    with torch.no_grad():
        without_grad = torch.stack([model(x) for x in inputs])

    assert torch.allclose(with_grad, without_grad, atol=1e-6)
    assert (with_grad[-1] - with_grad[0]).abs().max() > 0.1


def test_backward_works_after_warm_up_steps_without_grad():
    # warm-up: W goes 2 -> 4 -> 6 without a graph. Then y = (w0 + delta) * x with
    # delta = 4 a constant: dy/dw0 = x = 3.
    layer = drifting_layer()
    with torch.no_grad():
        layer(torch.ones(1))
        layer(torch.ones(1))

    layer(torch.tensor([3.0])).sum().backward()

    assert layer.weight.grad.item() == 3.0


def test_steps_without_grad_do_not_corrupt_an_earlier_graph():
    # y2 = (w0 + delta1) * x2 is recorded, then the state keeps evolving without
    # grad. Backward through y2 must still see the weights y2 was computed with.
    layer = drifting_layer()
    x2 = torch.tensor([3.0], requires_grad=True)
    layer(torch.ones(1))
    y2 = layer(x2)  # W = 4 at this point
    with torch.no_grad():
        for _ in range(3):
            layer(torch.ones(1))

    y2.sum().backward()

    assert x2.grad.item() == 4.0


def test_frozen_state_used_in_a_graph_survives_later_updates():
    # The state is built without grad, read with plasticity off while the input
    # requires grad, then updated again without grad.
    layer = drifting_layer()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        layer(torch.ones(1))  # W = 4
    layer.plastic = False
    x = torch.tensor([3.0], requires_grad=True)
    y = layer(x)
    layer.plastic = True
    with torch.no_grad():
        layer(torch.ones(1))  # W = 6

    y.sum().backward()

    assert x.grad.item() == 4.0


def test_resetting_some_rows_keeps_the_gradient_of_the_others():
    # two rows, A trainable; after one step dy2/dA = 1 per row (see above).
    # Row 0 is reset before the second step, so only row 1 contributes.
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)
    layer(torch.ones(2, 1))

    layer.reset_state(rows=[0])
    layer(torch.ones(2, 1)).sum().backward()

    assert layer.A.grad.item() == 1.0


def test_state_from_inference_mode_can_be_continued_outside_it():
    layer = drifting_layer()
    with torch.inference_mode():
        layer(torch.ones(2, 1))
        layer(torch.ones(2, 1))

    with torch.no_grad():
        third = layer(torch.ones(2, 1))
        fourth = layer(torch.ones(2, 1))

    assert third.flatten().tolist() == [6.0, 6.0]
    assert fourth.flatten().tolist() == [8.0, 8.0]


def test_rows_of_a_state_from_inference_mode_can_be_reset_outside_it():
    layer = drifting_layer()
    with torch.inference_mode():
        layer(torch.ones(2, 1))
        layer(torch.ones(2, 1))

    with torch.no_grad():
        layer.reset_state(rows=[1])
        third = layer(torch.ones(2, 1))

    assert third.flatten().tolist() == [6.0, 2.0]


def test_adam_can_train_the_rules():
    # Target: the output at step 2 should be 5 when fed ones (w0 = 2, so the
    # weight has to move by 3 in one step). Only the rules are optimised.
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    layer.weight.requires_grad_(False)
    optimiser = torch.optim.Adam(hb.rule_parameters(layer), lr=0.1)

    for _ in range(300):
        layer.reset_state()
        layer(torch.ones(1))
        loss = (layer(torch.ones(1)) - 5.0).pow(2).sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss.item() < 1e-4
    assert layer.weight.item() == 2.0


def test_state_from_inference_mode_can_enter_a_graph_afterwards():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), HebbianLinear(4, 2, eta=0.5, decay=0.1))
    set_rule(net[2], "A", torch.randn(2, 4))
    with torch.inference_mode():
        net(torch.randn(3, 4))

    net(torch.randn(3, 4)).sum().backward()

    assert net[0].weight.grad.abs().sum() > 0


def test_population_set_in_inference_mode_can_be_used_with_autograd():
    layer = HebbianLinear(4, 2, eta=0.5, decay=0.1)
    with torch.inference_mode():
        hb.set_rules(layer, torch.ones(3, layer.num_rules))
    x = torch.randn(3, 4, requires_grad=True)

    layer(x)
    layer(x).sum().backward()

    assert x.grad.abs().sum() > 0
