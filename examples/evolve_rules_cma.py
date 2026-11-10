"""Evolve Hebbian rules with CMA-ES on top of a trained, frozen network.

    python examples/evolve_rules_cma.py

Two stages, on the association task of ``association.py``:

1. **Train the weights** the ordinary way, by gradient descent with plasticity
   switched off. The network learns to repeat the teacher while studying, but
   it cannot remember: its recall error stays at chance (1.0).
2. **Freeze the weights and evolve the rules** with pycma. Nothing but the rule
   vector changes, and the recall error drops close to 0.

The second stage shows the three things an evolutionary loop needs from the
library:

* ``hb.num_rules(model)``: the length of the vector to search;
* ``hb.set_rules(model, candidates)`` with the whole list returned by
  ``es.ask()``: all candidates are evaluated in one batch, candidate k owning
  the k-th block of batch rows;
* ``hb.save_rules`` / ``hb.load_rules``: the result lives in a ``.npy`` file
  and is loaded with ``requires_grad=False``.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cma
import numpy as np
import torch
from torch import nn

import torch_hebbian as hb
from association import make_episodes, recall_error, run_episodes

DEFAULT_OUT = Path(__file__).parent / "output" / "association_rules_cma.npy"


def train_weights(model: nn.Module, n_cues: int, generator: torch.Generator, steps: int = 300):
    """Stage 1: supervised training of weight and bias, plasticity off."""
    layer = model[0]
    layer.plastic = False
    optimiser = torch.optim.Adam([layer.weight, layer.bias], lr=0.05)
    for _ in range(steps):
        inputs, targets = make_episodes(64, n_cues, generator)
        loss = (run_episodes(model, inputs) - targets).pow(2).mean()  # every step counts
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    layer.plastic = True


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--population", type=int, default=24, help="candidates per generation")
    parser.add_argument("--episodes", type=int, default=64, help="episodes per candidate")
    parser.add_argument("--cues", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print

    torch.set_num_threads(1)  # tiny tensors: threads only add overhead
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    # One plastic layer: cue and teacher lines in, the answer out.
    model = nn.Sequential(hb.HebbianLinear(args.cues + 1, 1, eta=0.1, decay=0.05))

    # ---- stage 1: weights -------------------------------------------------
    train_weights(model, args.cues, generator)
    model.requires_grad_(False)  # from here on nothing is trained by gradient
    trained_weights = copy.deepcopy(model.state_dict())  # rules are still zero in here

    # ---- stage 2: rules ---------------------------------------------------
    n_rules = hb.num_rules(model)
    K, S = args.population, args.episodes
    say(f"{n_rules} rules to evolve, {K} candidates x {S} episodes per generation")
    es = cma.CMAEvolutionStrategy(
        np.zeros(n_rules), 1.0, {"popsize": K, "seed": args.seed + 1, "verbose": -9}
    )
    with torch.no_grad():
        for generation in range(1, args.generations + 1):
            candidates = es.ask()  # list of K vectors
            hb.set_rules(model, candidates)  # K rule sets side by side
            # Every candidate meets the same S episodes (common random numbers),
            # so candidates are ranked by their rules and not by their luck.
            inputs, targets = make_episodes(S, args.cues, generator)
            inputs, targets = inputs.repeat(1, K, 1), targets.repeat(1, K)
            errors = recall_error(model, inputs, targets).reshape(K, S).mean(dim=1)
            es.tell(candidates, errors.tolist())
            if generation % 25 == 0 or generation == args.generations:
                say(f"generation {generation:4d}   best error {errors.min():.4f}")

    # Keep the mean of the search distribution and store it as a .npy file.
    hb.set_rules(model, es.mean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hb.save_rules(model, args.out)

    # ---- check: a second network gets the weights, then the rules from file
    fresh = nn.Sequential(hb.HebbianLinear(args.cues + 1, 1, eta=0.1, decay=0.05))
    fresh.load_state_dict(trained_weights)
    test_inputs, test_targets = make_episodes(2000, args.cues, generator)
    with torch.no_grad():
        error_without_rules = recall_error(fresh, test_inputs, test_targets).mean().item()
        hb.load_rules(fresh, args.out)
        error_reloaded = recall_error(fresh, test_inputs, test_targets).mean().item()
        error_evolved = recall_error(model, test_inputs, test_targets).mean().item()

    say("\nrecall error on 2000 new episodes (1.0 = chance)")
    say(f"  trained weights, no rules        {error_without_rules:.4f}")
    say(f"  same weights + evolved rules     {error_reloaded:.4f}")
    say(f"rules saved to {args.out}")
    return {
        "error_without_rules": error_without_rules,
        "error_evolved": error_evolved,
        "error_reloaded": error_reloaded,
        "rules_frozen_after_loading": not any(p.requires_grad for p in hb.rule_parameters(fresh)),
    }


if __name__ == "__main__":
    main()
