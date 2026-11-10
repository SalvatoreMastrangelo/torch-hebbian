"""The plastic state: one weight matrix per batch row, kept between forward
passes until it is reset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from helpers import make_layer, set_rule
from torch_hebbian import HebbianLinear


def drifting_layer(**kwargs):
    """w0 = 2 and a constant drift of +2 per step (D = 4, eta = 0.5, no decay):
    fed with ones, a row answers 2, 4, 6, ... as long as its state is kept."""
    layer = make_layer([[2.0]], eta=0.5, decay=0.0, **kwargs)
    set_rule(layer, "D", [[4.0]])
    return layer


def ones(*batch):
    return torch.ones(*batch, 1)


def test_state_persists_across_forward_calls():
    layer = drifting_layer()

    assert [layer(ones(1)).item() for _ in range(3)] == [2.0, 4.0, 6.0]


def test_each_batch_row_owns_its_weights():
    # A = 1: the change of a row depends on that row's own input and output.
    # row 0 sees x = 1: y = 2, dW = 0.5*2*1 = 1 -> W = 3
    # row 1 sees x = 3: y = 6, dW = 0.5*6*3 = 9 -> W = 11
    layer = make_layer([[2.0]], eta=0.5, decay=0.0)
    set_rule(layer, "A", [[1.0]])

    layer(torch.tensor([[1.0], [3.0]]))
    second = layer(ones(2))

    assert second.flatten().tolist() == [3.0, 11.0]


def test_reset_state_restores_the_original_weights():
    layer = drifting_layer()
    layer(ones(2))
    layer(ones(2))

    layer.reset_state()

    assert layer(ones(2)).flatten().tolist() == [2.0, 2.0]


def test_reset_state_with_a_mask_resets_only_those_rows():
    layer = drifting_layer()
    layer(ones(3))
    layer(ones(3))  # every row is now at W = 6

    layer.reset_state(rows=torch.tensor([False, True, False]))

    assert layer(ones(3)).flatten().tolist() == [6.0, 2.0, 6.0]


def test_reset_state_with_indices_resets_only_those_rows():
    layer = drifting_layer()
    layer(ones(3))
    layer(ones(3))

    layer.reset_state(rows=[0, 2])

    assert layer(ones(3)).flatten().tolist() == [2.0, 6.0, 2.0]


def test_reset_state_accepts_a_numpy_mask():
    # e.g. ``terminated | truncated`` straight from a vectorised environment
    layer = drifting_layer()
    layer(ones(3))
    layer(ones(3))

    layer.reset_state(rows=np.array([True, False, True]))

    assert layer(ones(3)).flatten().tolist() == [2.0, 6.0, 2.0]


def test_reset_state_with_an_all_false_mask_changes_nothing():
    layer = drifting_layer()
    layer(ones(3))
    layer(ones(3))

    layer.reset_state(rows=torch.zeros(3, dtype=torch.bool))

    assert layer(ones(3)).flatten().tolist() == [6.0, 6.0, 6.0]


def test_reset_state_with_rows_needs_a_batch_dimension():
    layer = drifting_layer()
    layer(torch.ones(1))  # unbatched: there is no row to pick

    with pytest.raises(ValueError, match="batch dimension"):
        layer.reset_state(rows=[0])


def test_reset_state_with_rows_before_any_step_is_harmless():
    layer = drifting_layer()

    layer.reset_state(rows=torch.tensor([True, False]))

    assert layer(ones(2)).flatten().tolist() == [2.0, 2.0]


def test_changing_the_batch_shape_without_reset_is_an_error():
    layer = drifting_layer()
    layer(ones(3))

    with pytest.raises(RuntimeError, match="reset_state"):
        layer(ones(4))


def test_a_new_batch_shape_is_accepted_after_reset():
    layer = drifting_layer()
    layer(ones(3))

    layer.reset_state()

    assert layer(ones(4)).flatten().tolist() == [2.0] * 4


def test_plastic_false_freezes_the_current_weights():
    layer = drifting_layer()
    layer(ones(1))  # W = 4 afterwards

    layer.plastic = False
    frozen = [layer(ones(1)).item() for _ in range(3)]
    layer.plastic = True
    resumed = [layer(ones(1)).item() for _ in range(2)]

    assert frozen == [4.0, 4.0, 4.0]
    assert resumed == [4.0, 6.0]


def test_plastic_false_from_the_start_is_a_plain_linear_layer():
    layer = drifting_layer()
    layer.plastic = False

    assert [layer(ones(1)).item() for _ in range(2)] == [2.0, 2.0]
    assert layer.weight_delta is None


def test_unbatched_input_is_a_single_row():
    layer = drifting_layer()

    outputs = [layer(torch.ones(1)) for _ in range(3)]

    assert [tuple(y.shape) for y in outputs] == [(1,)] * 3
    assert [y.item() for y in outputs] == [2.0, 4.0, 6.0]


def test_every_leading_index_is_its_own_row():
    torch.manual_seed(0)
    nested = HebbianLinear(4, 3, eta=0.2, decay=0.1)
    for name in "ABCD":
        set_rule(nested, name, torch.randn(3, 4))
    flat = HebbianLinear(4, 3, eta=0.2, decay=0.1)
    flat.load_state_dict(nested.state_dict())

    for _ in range(4):
        x = torch.randn(2, 5, 4)
        y = nested(x)
        assert y.shape == (2, 5, 3)
        assert torch.allclose(y, flat(x.reshape(10, 4)).reshape(2, 5, 3))


def test_reset_state_rows_index_the_first_batch_dimension():
    layer = drifting_layer()
    layer(ones(2, 3))
    layer(ones(2, 3))  # all at W = 6

    layer.reset_state(rows=torch.tensor([True, False]))

    assert layer(ones(2, 3)).squeeze(-1).tolist() == [[2.0] * 3, [6.0] * 3]


def test_weight_delta_and_plastic_weight_report_the_state():
    layer = drifting_layer()
    assert layer.weight_delta is None
    assert torch.equal(layer.plastic_weight, layer.weight)

    layer(ones(2, 3))

    assert layer.weight_delta.shape == (2, 3, 1, 1)
    assert layer.weight_delta.flatten().tolist() == [2.0] * 6
    assert layer.plastic_weight.flatten().tolist() == [4.0] * 6


def test_weight_delta_is_a_snapshot_not_a_live_view():
    layer = drifting_layer()
    with torch.no_grad():
        layer(ones(1))
        snapshot = layer.weight_delta
        layer(ones(1))

    assert snapshot.item() == 2.0


# --- what reset_state accepts as ``rows`` ---------------------------------
# Integers are row indices. Environments often report finished episodes as
# 0/1 integers or floats; reading those as indices would reset the wrong rows,
# so they are refused instead of guessed.

def six_everywhere(rows=4):
    layer = drifting_layer()
    layer(ones(rows))
    layer(ones(rows))  # every row at W = 6
    return layer


@pytest.mark.parametrize(
    "flags",
    [
        torch.tensor([0, 0, 1, 0]),
        np.array([0, 0, 1, 0], dtype=np.int32),
        [0, 0, 1, 0],
        torch.tensor([0, 0, 0, 0]),
    ],
    ids=["long tensor", "int32 array", "list", "all zero"],
)
def test_reset_state_refuses_integer_flags_instead_of_guessing(flags):
    layer = six_everywhere()

    with pytest.raises(ValueError, match="bool"):
        layer.reset_state(rows=flags)

    assert layer(ones(4)).flatten().tolist() == [6.0] * 4  # nothing was reset


@pytest.mark.parametrize(
    "flags",
    [torch.tensor([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])],
    ids=["float tensor", "float array"],
)
def test_reset_state_refuses_float_flags(flags):
    layer = six_everywhere()

    with pytest.raises(TypeError, match="bool"):
        layer.reset_state(rows=flags)

    assert layer(ones(4)).flatten().tolist() == [6.0] * 4


def test_reset_state_refuses_indices_that_could_be_flags():
    # with two rows, [1, 0] could mean "both rows" or "the first row only"
    layer = six_everywhere(rows=2)

    with pytest.raises(ValueError, match="bool"):
        layer.reset_state(rows=[1, 0])


def test_reset_state_refuses_repeated_indices():
    layer = six_everywhere()

    with pytest.raises(ValueError, match="repeated"):
        layer.reset_state(rows=[1, 1])


@pytest.mark.parametrize(
    "rows, expected",
    [
        ((0, 2), [2.0, 6.0, 2.0, 6.0]),
        (np.array([3, 1]), [6.0, 2.0, 6.0, 2.0]),
        (torch.tensor([False, True, True, False]).nonzero().squeeze(-1), [6.0, 2.0, 2.0, 6.0]),
        (1, [6.0, 2.0, 6.0, 6.0]),
        (slice(0, 2), [2.0, 2.0, 6.0, 6.0]),
        ([], [6.0] * 4),
        (torch.zeros(0, dtype=torch.long), [6.0] * 4),
    ],
    ids=["tuple", "int array", "nonzero ids", "single int", "slice", "empty list", "empty ids"],
)
def test_reset_state_accepts_row_indices(rows, expected):
    layer = six_everywhere()

    layer.reset_state(rows=rows)

    assert layer(ones(4)).flatten().tolist() == expected
