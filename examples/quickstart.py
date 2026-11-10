"""Quickstart: the whole API in one page.

    python examples/quickstart.py

Each numbered block prints what it shows. Read it top to bottom.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

import torch_hebbian as hb

DEFAULT_OUT = Path(__file__).parent / "output"


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="folder for the .npy file")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    facts = {}

    # ---------------------------------------------------------------- 1 ----
    # HebbianLinear is an nn.Linear: use it anywhere in a network. Its rules
    # start at zero, and with zero rules it IS an nn.Linear.
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.Tanh(),
        hb.HebbianLinear(16, 2, eta=0.05, decay=0.1),
    )
    x = torch.randn(3, 4)  # 3 batch rows = 3 independent agents, each with its own weights
    with torch.no_grad():
        first, second = model(x), model(x)
    facts["zero_rules_match_linear"] = torch.equal(first, second)
    print("1. zero rules, same input twice -> same output:", facts["zero_rules_match_linear"])

    # ---------------------------------------------------------------- 2 ----
    # The rules of a model are one flat vector: A|B|C|D for each plastic layer.
    n_rules = hb.num_rules(model)  # 4 rules x 2 outputs x 16 inputs = 128
    rules = rng.normal(size=n_rules)
    hb.set_rules(model, rules)  # also sets requires_grad=False on the rules
    hb.reset_state(model)  # start of an episode: plastic weights = layer weights
    with torch.no_grad():
        outputs = torch.stack([model(x) for _ in range(5)])  # 5 time steps
    facts["outputs_change_with_rules"] = not torch.allclose(outputs[0], outputs[-1])
    print(f"2. {n_rules} rules set; the same input now gives a drifting output (row 0, unit 0):")
    print("  ", [round(v, 3) for v in outputs[:, 0, 0].tolist()])

    # ---------------------------------------------------------------- 3 ----
    # reset_state() puts the weights back; rows=... does it for some rows only,
    # e.g. the environments whose episode has just ended.
    hb.reset_state(model)
    with torch.no_grad():
        after_reset = model(x)
    facts["reset_restores_first_output"] = torch.equal(after_reset, first)
    print("3. after reset_state the first output is back:", facts["reset_restores_first_output"])
    hb.reset_state(model, rows=torch.tensor([True, False, False]))  # only row 0

    # ---------------------------------------------------------------- 4 ----
    # decay pulls the plastic weights back to the layer's own weight. With the
    # learning rate at zero nothing else acts, and the change made so far
    # shrinks by (1 - decay) per step.
    layer = model[2]
    with torch.no_grad():
        for _ in range(5):
            model(x)
        before = layer.weight_delta.norm()  # weight_delta = plastic weights - layer.weight
        layer.eta = 0.0
        for _ in range(10):
            model(x)
        after = layer.weight_delta.norm()
        layer.eta = 0.05
    facts["fraction_left_after_decay"] = (after / before).item()
    print(f"4. 10 steps of pure decay (0.1) leave {after / before:.1%} of the weight change"
          f" (0.9^10 = {0.9 ** 10:.1%})")

    # ---------------------------------------------------------------- 5 ----
    # Rules live in .npy files. Loading freezes them (requires_grad=False), so
    # an evolutionary algorithm can own them while optimisers ignore them.
    path = args.out / "quickstart_rules.npy"
    hb.save_rules(model, path)
    twin = copy.deepcopy(model)
    hb.set_rules(twin, np.zeros(n_rules))  # forget the rules...
    hb.load_rules(twin, path)  # ...and read them back from the file
    facts["reloaded_rules_match"] = np.array_equal(hb.get_rules(twin), hb.get_rules(model))
    print(f"5. rules saved to {path} and loaded back:", facts["reloaded_rules_match"],
          "| trainable rule tensors:", sum(p.requires_grad for p in hb.rule_parameters(twin)))

    # ---------------------------------------------------------------- 6 ----
    # Add plasticity to a network that already exists (e.g. a trained policy).
    # The converted layers reuse the original weights and start with zero
    # rules, so the network computes exactly what it did before.
    trained = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))
    with torch.no_grad():
        reference = trained(x)
    hb.hebbianize(trained, {"2": dict(eta=0.01, decay=0.05), "4": dict(eta=0.05, decay=0.1)})
    with torch.no_grad():
        converted = trained(x)
    facts["conversion_keeps_outputs"] = torch.equal(converted, reference)
    print("6. hebbianize() made layers", [name for name, _ in hb.hebbian_layers(trained)],
          "plastic; outputs unchanged:", facts["conversion_keeps_outputs"])

    # ---------------------------------------------------------------- 7 ----
    # A population: K rule vectors evaluated in ONE batch. The batch is split in
    # K contiguous blocks; block k follows rule vector k.
    K, S = 4, 3  # 4 candidates x 3 rows each
    population = rng.normal(size=(K, hb.num_rules(model)))
    batch = torch.randn(K * S, 4)
    hb.set_rules(model, population)
    hb.reset_state(model)
    with torch.no_grad():
        together = torch.stack([model(batch) for _ in range(5)])
    matches = []
    for k in range(K):
        hb.set_rules(model, population[k])
        hb.reset_state(model)
        with torch.no_grad():
            alone = torch.stack([model(batch[k * S:(k + 1) * S]) for _ in range(5)])
        matches.append(torch.allclose(together[:, k * S:(k + 1) * S], alone, atol=1e-6))
    facts["population_matches_separate_runs"] = all(matches)
    print(f"7. {K} rule sets in one batch of {K * S} rows = {K} separate runs:",
          facts["population_matches_separate_runs"])

    return facts


if __name__ == "__main__":
    main()
