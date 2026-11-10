"""Train Hebbian rules by gradient descent, together with the network weights.

    python examples/train_rules_adam.py

Same association task as ``evolve_rules_cma.py``, but here nothing is frozen:
the rules ``A, B, C, D`` are ordinary parameters and Adam optimises them along
with the weights of a small MLP. The loss is measured at the end of each
episode, so its gradient travels back through every weight update the plastic
layer made during the episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import torch_hebbian as hb
from association import make_episodes, recall_error

DEFAULT_OUT = Path(__file__).parent / "output" / "association_rules_adam.npy"


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=64, help="episodes per iteration")
    parser.add_argument("--cues", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print

    torch.set_num_threads(1)  # tiny tensors: threads only add overhead
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    model = nn.Sequential(
        nn.Linear(args.cues + 1, args.hidden),
        nn.Tanh(),
        hb.HebbianLinear(args.hidden, 1, eta=0.1, decay=0.02),  # plastic read-out
    )
    # model.parameters() already contains the rules; hb.rule_parameters(model)
    # would select them alone.
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)

    for iteration in range(1, args.iterations + 1):
        inputs, targets = make_episodes(args.episodes, args.cues, generator)
        loss = recall_error(model, inputs, targets).mean()  # resets the plastic state first
        optimiser.zero_grad()
        loss.backward()  # through all the steps of the episode
        optimiser.step()
        if iteration % 100 == 0 or iteration == args.iterations:
            say(f"iteration {iteration:4d}   recall error {loss.item():.4f}")

    test_inputs, test_targets = make_episodes(2000, args.cues, generator)
    plastic_layer = model[2]
    with torch.no_grad():
        error_trained = recall_error(model, test_inputs, test_targets).mean().item()
        plastic_layer.plastic = False  # same weights, but they no longer change
        error_plasticity_off = recall_error(model, test_inputs, test_targets).mean().item()
        plastic_layer.plastic = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hb.save_rules(model, args.out)

    say("\nrecall error on 2000 new episodes (1.0 = chance)")
    say(f"  trained network                    {error_trained:.4f}")
    say(f"  same network with plasticity off   {error_plasticity_off:.4f}")
    say(f"rules saved to {args.out}")
    return {"error_trained": error_trained, "error_plasticity_off": error_plasticity_off}


if __name__ == "__main__":
    main()
