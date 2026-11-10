"""Reading the plastic state out and putting it back: ``get_state`` / ``set_state``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torch_hebbian as hb
from helpers import make_layer, set_rule
from torch_hebbian import HebbianLinear

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def drifting_layer():
    """w0 = 2 and a constant drift of +2 per step (D = 4, eta = 0.5, no decay):
    fed with ones a row answers 2, 4, 6, ... and its state is 0, 2, 4, ..."""
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    set_rule(layer, "D", [[4.0]])
    return layer


def ones(*batch):
    return torch.ones(*batch, 1)


def deltas(*values):
    """A state for a 1x1 layer: one deviation per row, shape (rows, 1, 1)."""
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)


# --------------------------------------------------------------------------
# one layer
# --------------------------------------------------------------------------

def test_get_state_is_the_deviation_from_the_weight():
    layer = drifting_layer()
    with torch.no_grad():
        layer(ones(3))
        layer(ones(3))  # plastic weight 6 = weight 2 + state 4

    state = layer.get_state()

    assert state.shape == (3, 1, 1)
    assert state.flatten().tolist() == [4.0, 4.0, 4.0]


def test_get_state_is_none_when_there_is_no_state():
    layer = drifting_layer()
    assert layer.get_state() is None

    layer(ones(2))
    layer.reset_state()
    assert layer.get_state() is None


def test_set_state_goes_back_to_an_earlier_moment():
    layer = drifting_layer()
    with torch.no_grad():
        first = [layer(ones(1)).item() for _ in range(2)]  # 2, 4
        snapshot = layer.get_state()  # the next answer would be 6
        later = [layer(ones(1)).item() for _ in range(2)]  # 6, 8

        layer.set_state(snapshot)
        again = [layer(ones(1)).item() for _ in range(2)]

    assert first == [2.0, 4.0]
    assert later == [6.0, 8.0]
    assert again == [6.0, 8.0]


def test_a_snapshot_does_not_follow_the_layer():
    layer = drifting_layer()
    with torch.no_grad():
        layer(ones(1))
        snapshot = layer.get_state()
        layer(ones(1))
        layer(ones(1))

    assert snapshot.item() == 2.0


def test_the_layer_does_not_follow_the_tensor_it_was_given():
    layer = drifting_layer()
    given = deltas(1.0, 2.0)

    with torch.no_grad():
        layer.set_state(given)
        first = layer(ones(2))  # 2 + [1, 2]; the state moves on to [3, 4]
        given.fill_(100.0)  # e.g. a rollout buffer being overwritten
        second = layer(ones(2))

    assert first.flatten().tolist() == [3.0, 4.0]
    assert second.flatten().tolist() == [5.0, 6.0]


def test_the_tensor_given_to_set_state_is_left_untouched():
    layer = drifting_layer()
    given = deltas(1.0, 2.0)

    with torch.no_grad():
        layer.set_state(given)
        layer(ones(2))
        layer(ones(2))

    assert given.flatten().tolist() == [1.0, 2.0]


def test_set_state_none_is_a_reset():
    layer = drifting_layer()
    layer(ones(2))
    layer(ones(2))

    layer.set_state(None)

    assert layer.get_state() is None
    assert layer(ones(5)).flatten().tolist() == [2.0] * 5  # any batch shape again


def test_set_state_defines_the_batch():
    # keep only some of the rows, e.g. the environments that are still running
    layer = drifting_layer()
    with torch.no_grad():
        layer.set_state(deltas(1.0, 2.0, 3.0))
        layer(ones(3))  # states become 3, 4, 5

        layer.set_state(layer.get_state()[[0, 2]])
        output = layer(ones(2))

    assert output.flatten().tolist() == [5.0, 7.0]


def test_set_state_accepts_extra_batch_dimensions():
    layer = drifting_layer()
    state = torch.arange(6.0).reshape(2, 3, 1, 1)

    with torch.no_grad():
        layer.set_state(state)
        output = layer(ones(2, 3))

    assert output.squeeze(-1).tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    assert layer.get_state().shape == (2, 3, 1, 1)


def test_set_state_accepts_an_unbatched_state():
    layer = drifting_layer()

    with torch.no_grad():
        layer.set_state(torch.tensor([[5.0]]))
        output = layer(torch.ones(1))

    assert output.tolist() == [7.0]


@pytest.mark.parametrize(
    "shape", [(4, 3, 2), (4, 2), (3,), ()], ids=["transposed", "one matrix too few", "vector", "scalar"]
)
def test_set_state_rejects_a_state_for_another_layer_shape(shape):
    layer = HebbianLinear(3, 2)  # weight is (2, 3)

    with pytest.raises(ValueError, match=r"2, 3"):
        layer.set_state(torch.zeros(shape))

    assert layer.get_state() is None


def test_set_state_converts_to_the_dtype_of_the_layer():
    layer = drifting_layer()

    with torch.no_grad():
        layer.set_state(deltas(1.0, 2.0).double())
        output = layer(ones(2))

    assert output.dtype == torch.float32
    assert output.flatten().tolist() == [3.0, 4.0]


def test_set_state_accepts_a_numpy_array():
    layer = drifting_layer()

    with torch.no_grad():
        layer.set_state(np.array([[[1.0]], [[2.0]]]))
        output = layer(ones(2))

    assert output.flatten().tolist() == [3.0, 4.0]


def test_weight_delta_and_get_state_agree():
    layer = drifting_layer()
    layer(ones(2, 3))

    assert torch.equal(layer.weight_delta, layer.get_state())


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------

def test_gradient_flows_through_a_restored_state():
    # w0 = 2, x = 1, eta = 0.5: after one step the state is 0.5*(2A + B + 2C + D),
    # so the next output has d/dA = 1 (as in test_gradients). Taking the state
    # out and putting it back must not cut that dependence.
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)
    layer(torch.ones(1))

    snapshot = layer.get_state()
    layer.reset_state()
    layer.set_state(snapshot)
    layer(torch.ones(1)).sum().backward()

    assert layer.A.grad.item() == 1.0


def test_a_detached_snapshot_cuts_the_gradient():
    layer = make_layer([[2.0]], eta=0.5, decay=0.25)
    layer(torch.ones(1))

    layer.set_state(layer.get_state().detach())
    layer(torch.ones(1)).sum().backward()

    assert layer.A.grad is None
    assert layer.weight.grad.item() == 1.0


def test_steps_without_grad_after_set_state_do_not_corrupt_an_earlier_graph():
    layer = drifting_layer()
    x = torch.tensor([[3.0]], requires_grad=True)
    layer.set_state(deltas(2.0))  # plastic weight 4
    y = layer(x)
    with torch.no_grad():
        for _ in range(3):
            layer(ones(1))

    y.sum().backward()

    assert x.grad.item() == 4.0


# --------------------------------------------------------------------------
# a whole model
# --------------------------------------------------------------------------

def model():
    torch.manual_seed(0)
    net = nn.Sequential(
        HebbianLinear(3, 4, eta=0.2, decay=0.1),
        nn.Tanh(),
        nn.Linear(4, 4),
        HebbianLinear(4, 2, eta=0.2, decay=0.1, w_max=2.0),
    )
    hb.set_rules(net, np.random.default_rng(0).normal(size=hb.num_rules(net)))
    return net


def inputs(steps, rows=5, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, 3, generator=generator) for _ in range(steps)]


def run(net, xs):
    with torch.no_grad():
        return torch.stack([net(x) for x in xs])


def test_model_snapshot_is_keyed_by_layer_name():
    net = model()
    run(net, inputs(2))

    snapshot = hb.get_state(net)

    assert list(snapshot) == ["0", "3"]
    assert snapshot["0"].shape == (5, 4, 3)
    assert snapshot["3"].shape == (5, 2, 4)


def test_model_snapshot_restores_every_layer():
    net = model()
    warm_up, rest = inputs(3), inputs(4, seed=2)
    run(net, warm_up)

    snapshot = hb.get_state(net)
    expected = run(net, rest)
    hb.set_state(net, snapshot)

    assert torch.equal(run(net, rest), expected)
    assert not torch.equal(expected[0], expected[-1] * 0)  # sanity: something happened


def test_restoring_matters():
    # without set_state the second pass over the same inputs starts elsewhere
    net = model()
    rest = inputs(4, seed=2)
    run(net, inputs(3))

    expected = run(net, rest)

    assert not torch.allclose(run(net, rest), expected)


def test_snapshot_moves_to_another_copy_of_the_model():
    source, target = model(), model()
    rest = inputs(4, seed=2)
    run(source, inputs(3))

    hb.set_state(target, hb.get_state(source))

    assert torch.equal(run(target, rest), run(source, rest))


def test_snapshot_survives_torch_save(tmp_path):
    net = model()
    rest = inputs(4, seed=2)
    run(net, inputs(3))

    torch.save(hb.get_state(net), tmp_path / "state.pt")
    expected = run(net, rest)
    hb.set_state(net, torch.load(tmp_path / "state.pt"))

    assert torch.equal(run(net, rest), expected)


def test_snapshot_of_a_model_without_state_resets_it():
    net = model()
    blank = hb.get_state(net)
    run(net, inputs(2))

    hb.set_state(net, blank)

    assert blank == {"0": None, "3": None}
    assert hb.get_state(net) == {"0": None, "3": None}


def test_snapshot_of_a_single_layer_uses_the_empty_name():
    layer = drifting_layer()
    layer(ones(2))

    assert list(hb.get_state(layer)) == [""]


@pytest.mark.parametrize(
    "broken, complaint",
    [
        (lambda s: {"0": s["0"]}, "3"),  # a layer is missing
        (lambda s: {**s, "7": s["3"]}, "7"),  # a layer too many
    ],
    ids=["missing layer", "unknown layer"],
)
def test_set_state_rejects_a_snapshot_of_another_model(broken, complaint):
    net = model()
    run(net, inputs(2))
    snapshot = hb.get_state(net)

    with pytest.raises(ValueError, match=complaint):
        hb.set_state(net, broken(snapshot))


def test_set_state_installs_nothing_when_one_entry_is_wrong():
    net = model()
    run(net, inputs(2))
    before = hb.get_state(net)
    bad = {"0": torch.zeros(5, 4, 3), "3": torch.zeros(5, 9, 9)}

    with pytest.raises(ValueError, match="3"):
        hb.set_state(net, bad)

    assert torch.equal(hb.get_state(net)["0"], before["0"])


@cuda
def test_snapshot_taken_on_cpu_restores_a_model_on_cuda():
    net = model()
    rest = inputs(4, seed=2)
    run(net, inputs(3))
    snapshot = hb.get_state(net)
    expected = run(net, rest)

    net.cuda()
    hb.set_state(net, snapshot)

    assert torch.allclose(run(net, [x.cuda() for x in rest]).cpu(), expected, atol=1e-5)
