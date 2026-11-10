"""CartPole with random frozen weights and evolved Hebbian rules.

    python examples/cartpole_evolved_rules.py            # about a minute
    python examples/cartpole_evolved_rules.py --no-flip  # plain CartPole, quicker

The policy is a tiny MLP whose weights are random and never trained. CMA-ES
evolves only the Hebbian rules of its two layers (after Najarro & Risi, 2020,
"Meta-learning through Hebbian plasticity in random networks").

By default the task has a hidden twist: in half of the episodes the two actions
are swapped, and the policy is not told. No fixed set of weights can balance
the pole in both cases (the best it can do is 500 steps in one case and about
10 in the other), so a good score in both columns of the final table means the
network adapts its own weights during the episode.

Things to look at:

* one batch row per environment, one rule set per block of rows;
* ``hb.reset_state(policy)`` at the start of every episode;
* everything under ``torch.no_grad()``: evolution needs no gradients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
from torch import nn

import torch_hebbian as hb

DEFAULT_OUT = Path(__file__).parent / "output" / "cartpole_rules.npy"
MAX_STEPS = 500  # CartPole-v1 truncates episodes there


def build_policy(seed: int) -> nn.Module:
    """4 observations -> 8 tanh units -> 2 action scores. Both layers plastic."""
    torch.manual_seed(seed)
    policy = nn.Sequential(
        hb.HebbianLinear(4, 8, eta=0.1, decay=0.01),
        nn.Tanh(),
        hb.HebbianLinear(8, 2, eta=0.1, decay=0.01),
    )
    return policy.requires_grad_(False)


def run_episodes(policy: nn.Module, envs: list, seeds: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    """Play one episode in every environment; batch row i drives ``envs[i]``.

    ``flipped[i]`` is 1 where the two actions are swapped. Returns the episode
    returns (number of steps the pole stayed up).
    """
    observations = np.stack([env.reset(seed=int(s))[0] for env, s in zip(envs, seeds)])
    alive = np.ones(len(envs), dtype=bool)
    returns = np.zeros(len(envs))
    hb.reset_state(policy)  # new episode: plastic weights back to the frozen ones
    with torch.no_grad():
        for _ in range(MAX_STEPS):
            scores = policy(torch.as_tensor(observations, dtype=torch.float32))
            actions = scores.argmax(dim=1).numpy() ^ flipped
            for i in np.flatnonzero(alive):
                observations[i], reward, terminated, truncated, _ = envs[i].step(int(actions[i]))
                returns[i] += reward
                alive[i] = not (terminated or truncated)
            if not alive.any():
                break
    return returns


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generations", type=int, default=None, help="default: 300, or 100 with --no-flip")
    parser.add_argument("--population", type=int, default=16, help="candidates per generation")
    parser.add_argument("--episodes", type=int, default=4, help="episodes per candidate (even)")
    parser.add_argument("--no-flip", action="store_true", help="plain CartPole: never swap the actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print
    flip = not args.no_flip
    generations = args.generations if args.generations is not None else (300 if flip else 100)
    K, S = args.population, args.episodes
    if S % 2:
        parser.error("--episodes must be even (half of them use swapped actions)")

    torch.set_num_threads(1)  # tiny tensors: threads only add overhead
    rng = np.random.default_rng(args.seed)
    policy = build_policy(args.seed)
    n_rules = hb.num_rules(policy)
    envs = [gym.make("CartPole-v1") for _ in range(K * S)]
    # Episodes alternate normal / swapped; every candidate gets the same S episodes.
    case = (np.arange(S) % 2) if flip else np.zeros(S, dtype=np.int64)
    say(f"{n_rules} rules to evolve, {K} candidates x {S} episodes per generation, "
        f"{'hidden action swap in half of the episodes' if flip else 'plain CartPole'}")

    es = cma.CMAEvolutionStrategy(
        np.zeros(n_rules), 0.5, {"popsize": K, "seed": args.seed + 1, "verbose": -9}
    )
    for generation in range(1, generations + 1):
        candidates = es.ask()
        hb.set_rules(policy, candidates)  # candidate k drives rows k*S ... (k+1)*S - 1
        seeds = np.tile(rng.integers(2 ** 30, size=S), K)
        returns = run_episodes(policy, envs, seeds, np.tile(case, K)).reshape(K, S)
        if flip:
            # Score by the worse of the two cases. Averaging them would reward
            # a policy that masters one case and ignores the other.
            fitness = np.minimum(returns[:, 0::2].mean(axis=1), returns[:, 1::2].mean(axis=1))
        else:
            fitness = returns.mean(axis=1)
        es.tell(candidates, (-fitness).tolist())  # CMA-ES minimises
        if generation % 25 == 0 or generation == generations:
            say(f"generation {generation:4d}   best {fitness.max():6.1f}   mean {fitness.mean():6.1f}")

    hb.set_rules(policy, es.mean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hb.save_rules(policy, args.out)

    # ---- check on new episodes: zero rules against the evolved rules --------
    n_test = 40
    test_envs = [gym.make("CartPole-v1") for _ in range(n_test)]
    test_seeds = rng.integers(2 ** 30, size=n_test)
    test_case = (np.arange(n_test) % 2) if flip else np.zeros(n_test, dtype=np.int64)
    evolved = run_episodes(policy, test_envs, test_seeds, test_case)
    blank = build_policy(args.seed)  # same frozen weights, rules at zero
    without_rules = run_episodes(blank, test_envs, test_seeds, test_case)
    for env in envs + test_envs:
        env.close()

    result = {
        "return_without_rules": without_rules.mean(),
        "return_evolved": evolved.mean(),
        "return_evolved_normal": evolved[test_case == 0].mean(),
    }
    say(f"\nmean return over {n_test} new episodes (maximum {MAX_STEPS})")
    if flip:
        result["return_evolved_swapped"] = evolved[test_case == 1].mean()
        say("                          normal   swapped actions")
        say(f"  random frozen weights  {without_rules[test_case == 0].mean():7.1f}   {without_rules[test_case == 1].mean():7.1f}")
        say(f"  + evolved rules        {result['return_evolved_normal']:7.1f}   {result['return_evolved_swapped']:7.1f}")
    else:
        say(f"  random frozen weights  {result['return_without_rules']:7.1f}")
        say(f"  + evolved rules        {result['return_evolved']:7.1f}")
    say(f"rules saved to {args.out}")
    return {key: float(value) for key, value in result.items()}


if __name__ == "__main__":
    main()
