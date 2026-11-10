"""``HebbianLinear`` has to behave like any other module of a torch network."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

import torch_hebbian as hb
from torch_hebbian import HebbianLinear

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def model(dtype=torch.float32):
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(4, 6, dtype=dtype),
        nn.Tanh(),
        HebbianLinear(6, 5, eta=0.2, decay=0.1, dtype=dtype),
        nn.ReLU(),
        HebbianLinear(5, 2, eta=0.2, decay=0.1, w_max=2.0, dtype=dtype),
    )
    rules = np.random.default_rng(0).normal(size=hb.num_rules(net))
    hb.set_rules(net, rules, freeze=False)
    return net


def inputs(steps=6, rows=3, dtype=torch.float32, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, 4, generator=generator).to(dtype) for _ in range(steps)]


def run(net, xs):
    with torch.no_grad():
        return torch.stack([net(x) for x in xs])


def test_state_dict_holds_parameters_only():
    net = model()
    run(net, inputs())

    assert set(net.state_dict()) == {
        "0.weight", "0.bias",
        "2.weight", "2.bias", "2.A", "2.B", "2.C", "2.D",
        "4.weight", "4.bias", "4.A", "4.B", "4.C", "4.D",
    }


def test_state_dict_round_trip_reproduces_the_behaviour():
    source = model()
    target = nn.Sequential(
        nn.Linear(4, 6), nn.Tanh(), HebbianLinear(6, 5, eta=0.2, decay=0.1),
        nn.ReLU(), HebbianLinear(5, 2, eta=0.2, decay=0.1, w_max=2.0),
    )

    target.load_state_dict(source.state_dict())

    assert torch.equal(run(target, inputs()), run(source, inputs()))


def test_plain_linear_state_dict_loads_non_strictly():
    plain = nn.Linear(6, 5)
    layer = HebbianLinear(6, 5)

    result = layer.load_state_dict(plain.state_dict(), strict=False)

    assert sorted(result.missing_keys) == ["A", "B", "C", "D"]
    assert torch.equal(layer.weight, plain.weight)


def test_parameters_include_the_rules():
    net = model()

    assert sum(p.numel() for p in net.parameters()) == (4 * 6 + 6) + 5 * (6 * 5) + 5 + 5 * (5 * 2) + 2


def test_deepcopy_mid_episode_continues_identically_and_independently():
    net = model()
    xs = inputs(steps=8)
    for x in xs[:4]:
        net(x)  # with grad: the state carries autograd history

    clone = copy.deepcopy(net)
    expected = run(net, xs[4:])
    assert torch.equal(run(clone, xs[4:]), expected)

    # the copy has its own state: running it further does not disturb the original
    assert not torch.equal(net[2].weight_delta, net[2].weight_delta * 0)
    before = net[2].weight_delta
    run(clone, xs[:2])
    assert torch.equal(net[2].weight_delta, before)


def test_deepcopy_keeps_a_loaded_population():
    net = model()
    population = np.random.default_rng(1).normal(size=(3, hb.num_rules(net)))
    hb.set_rules(net, population)
    hb.reset_state(net)

    clone = copy.deepcopy(net)

    assert np.allclose(hb.get_rules(clone), population)
    assert torch.equal(run(clone, inputs()), run(net, inputs()))


def test_whole_module_survives_torch_save_and_load(tmp_path):
    net = model()
    xs = inputs(steps=8)
    for x in xs[:4]:
        net(x)

    torch.save(net, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt", weights_only=False)

    assert torch.equal(run(loaded, xs[4:]), run(net, xs[4:]))


def test_dtype_conversion_carries_the_state_along():
    net = model()
    xs = inputs(steps=8)
    run(net, xs[:4])
    reference = copy.deepcopy(net)

    net.double()
    converted = run(net, [x.double() for x in xs[4:]])

    assert converted.dtype == torch.float64
    assert torch.allclose(converted, run(reference, xs[4:]).double(), atol=1e-5)


def test_dtype_conversion_carries_a_population_along():
    net = model()
    hb.set_rules(net, np.random.default_rng(1).normal(size=(3, hb.num_rules(net))))
    hb.reset_state(net)

    net.double()

    assert run(net, [x.double() for x in inputs()]).dtype == torch.float64


def test_train_and_eval_modes_do_not_change_plasticity():
    net = model()
    xs = inputs()

    net.train()
    hb.reset_state(net)
    training = run(net, xs)
    net.eval()
    hb.reset_state(net)
    evaluating = run(net, xs)

    assert torch.equal(training, evaluating)
    assert (training[-1] - training[0]).abs().max() > 0


def test_repr_shows_the_plasticity_settings():
    text = repr(HebbianLinear(6, 5, eta=0.2, decay=0.1, w_max=2.0))

    assert "in_features=6" in text and "out_features=5" in text
    assert "eta=0.2" in text and "decay=0.1" in text and "w_max=2.0" in text


def test_weight_initialisers_that_look_for_nn_linear_find_it():
    net = model()

    def zero_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)

    net.apply(zero_linear)

    assert net[2].weight.abs().sum() == 0 and net[4].weight.abs().sum() == 0


@cuda
def test_cuda_rollout_matches_cpu():
    xs = inputs(steps=10)
    cpu = run(model(), xs)

    gpu = run(model().cuda(), [x.cuda() for x in xs])

    assert gpu.is_cuda
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-5)


@cuda
def test_moving_to_cuda_mid_episode_carries_the_state_along():
    net = model()
    xs = inputs(steps=8)
    run(net, xs[:4])
    reference = copy.deepcopy(net)

    net.cuda()
    moved = run(net, [x.cuda() for x in xs[4:]])

    assert torch.allclose(moved.cpu(), run(reference, xs[4:]), atol=1e-5)


@cuda
def test_population_on_cuda_matches_cpu():
    population = np.random.default_rng(1).normal(size=(3, hb.num_rules(model())))
    xs = inputs(rows=6)
    cpu_net = model()
    hb.set_rules(cpu_net, population)
    hb.reset_state(cpu_net)
    gpu_net = model().cuda()
    hb.set_rules(gpu_net, population)

    assert torch.allclose(run(gpu_net, [x.cuda() for x in xs]).cpu(), run(cpu_net, xs), atol=1e-5)


@cuda
def test_population_follows_the_module_to_cuda():
    net = model()
    hb.set_rules(net, np.random.default_rng(1).normal(size=(3, hb.num_rules(net))))
    hb.reset_state(net)

    net.cuda()

    assert run(net, [x.cuda() for x in inputs()]).is_cuda


@cuda
def test_reset_rows_with_a_cpu_mask_on_a_cuda_layer():
    net = model().cuda()
    xs = [x.cuda() for x in inputs()]
    run(net, xs)

    hb.reset_state(net, rows=torch.tensor([True, False, False]))

    moved = net[2].weight_delta.flatten(1).abs().sum(1) > 0
    assert moved.tolist() == [False, True, True]


def test_plastic_state_and_population_are_not_buffers():
    # utilities that walk model.buffers() (weight averaging, DDP buffer
    # broadcast) must not see per-episode state
    net = model()
    hb.set_rules(net, np.random.default_rng(1).normal(size=(3, hb.num_rules(net))))
    run(net, inputs())

    assert list(net.buffers()) == []


def test_weight_averaging_ignores_the_plastic_state():
    torch.manual_seed(0)
    net = nn.Sequential(HebbianLinear(4, 4, eta=0.2, decay=0.1), nn.BatchNorm1d(4))
    hb.set_rules(net, np.random.default_rng(0).normal(size=hb.num_rules(net)), freeze=False)
    net(torch.randn(8, 4))

    averaged = torch.optim.swa_utils.AveragedModel(net, use_buffers=True)
    averaged.update_parameters(net)

    assert torch.equal(averaged.module[0].weight, net[0].weight)
