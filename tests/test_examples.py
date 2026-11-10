"""The examples are end-to-end checks of the library: each ``main`` is run with
a small budget and has to reach the outcome it advertises."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torch_hebbian as hb


@pytest.fixture(autouse=True)
def restore_torch_threads():
    # the examples switch torch to one thread; do not leak that into other tests
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


# --------------------------------------------------------------------------
# the association task shared by the CMA-ES and Adam examples
# --------------------------------------------------------------------------

def episodes(n_episodes=50, n_cues=4, seed=0):
    import association

    return association.make_episodes(n_episodes, n_cues, torch.Generator().manual_seed(seed))


def test_association_episode_has_a_study_and_a_recall_phase():
    inputs, targets = episodes(n_episodes=50, n_cues=4)

    assert inputs.shape == (8, 50, 5)  # 4 study + 4 recall steps, 4 cue lines + 1 teacher line
    assert targets.shape == (8, 50)


def test_association_shows_every_cue_once_per_phase():
    inputs, _ = episodes(n_cues=4)
    cues = inputs[..., :4]

    assert (cues.sum(-1) == 1).all()  # exactly one cue per step
    assert (cues[:4].sum(0) == 1).all()  # study: each cue once
    assert (cues[4:].sum(0) == 1).all()  # recall: each cue once


def test_association_teacher_line_carries_the_value_only_while_studying():
    inputs, targets = episodes()
    teacher = inputs[..., 4]

    assert torch.equal(teacher[:4], targets[:4])
    assert (teacher[4:] == 0).all()
    assert set(targets.unique().tolist()) == {-1.0, 1.0}


def test_association_recall_target_is_the_value_studied_for_that_cue():
    inputs, targets = episodes()
    cue = inputs[..., :4].argmax(-1)  # (T, episodes)

    for e in range(inputs.shape[1]):
        studied = {cue[t, e].item(): targets[t, e].item() for t in range(4)}
        assert all(targets[t, e].item() == studied[cue[t, e].item()] for t in range(4, 8))


def test_association_values_change_from_episode_to_episode():
    _, targets = episodes(n_episodes=200)

    assert 0.3 < (targets[0] == 1).float().mean() < 0.7


def test_recall_error_of_a_silent_model_is_one():
    import association

    class Silent(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    inputs, targets = episodes()

    assert association.recall_error(Silent(), inputs, targets).tolist() == [1.0] * 50


def test_recall_error_starts_every_call_from_a_reset_state():
    import association

    model = nn.Sequential(hb.HebbianLinear(5, 1, eta=0.5, decay=0.0))
    hb.set_rules(model, np.random.default_rng(0).normal(size=20))
    inputs, targets = episodes()

    with torch.no_grad():
        first = association.recall_error(model, inputs, targets)
        second = association.recall_error(model, inputs, targets)

    assert torch.equal(first, second)


# --------------------------------------------------------------------------
# the examples themselves
# --------------------------------------------------------------------------

def test_quickstart_runs_and_its_claims_hold(tmp_path, capsys):
    import quickstart

    facts = quickstart.main(["--out", str(tmp_path)])

    assert facts["zero_rules_match_linear"] is True
    assert facts["outputs_change_with_rules"] is True
    assert facts["reset_restores_first_output"] is True
    # with eta = 0 only the decay acts: 10 steps at decay 0.1 leave 0.9**10 = 34.9 %
    assert facts["fraction_left_after_decay"] == pytest.approx(0.9 ** 10, rel=1e-4)
    assert facts["reloaded_rules_match"] is True
    assert facts["conversion_keeps_outputs"] is True
    assert facts["population_matches_separate_runs"] is True
    assert capsys.readouterr().out.strip()  # it explains itself on stdout


def test_cma_example_evolves_rules_that_solve_the_task(tmp_path):
    pytest.importorskip("cma")
    import evolve_rules_cma

    out = tmp_path / "rules.npy"
    result = evolve_rules_cma.main(["--out", str(out), "--quiet"])

    assert result["error_without_rules"] > 0.9  # trained weights alone: chance level
    assert result["error_evolved"] < 0.05
    assert np.load(out).shape == (20,)  # 4 rules x 1 output x (4 cues + teacher)
    assert result["error_reloaded"] == pytest.approx(result["error_evolved"], abs=1e-6)
    assert result["rules_frozen_after_loading"] is True


def test_adam_example_trains_rules_that_solve_the_task(tmp_path):
    import train_rules_adam

    out = tmp_path / "rules.npy"
    result = train_rules_adam.main(["--iterations", "400", "--out", str(out), "--quiet"])

    assert result["error_trained"] < 0.05
    assert result["error_plasticity_off"] > 0.8  # the weights alone are at chance level
    assert np.abs(np.load(out)).max() > 0  # the rules moved away from zero and were saved


def test_cartpole_example_evolves_rules_that_balance_the_pole(tmp_path):
    pytest.importorskip("cma")
    pytest.importorskip("gymnasium")
    import cartpole_evolved_rules

    out = tmp_path / "rules.npy"
    result = cartpole_evolved_rules.main(
        ["--no-flip", "--generations", "30", "--out", str(out), "--quiet"]
    )

    assert result["return_without_rules"] < 50  # random weights drop the pole at once
    assert result["return_evolved"] > 5 * result["return_without_rules"]
    assert np.load(out).shape == (192,)  # 4 * (8*4 + 2*8)


def test_cartpole_example_runs_with_the_hidden_action_swap(tmp_path):
    pytest.importorskip("cma")
    pytest.importorskip("gymnasium")
    import cartpole_evolved_rules

    result = cartpole_evolved_rules.main(
        ["--generations", "3", "--out", str(tmp_path / "rules.npy"), "--quiet"]
    )

    # 3 generations cannot solve it; this only checks the swapped-action path runs
    assert set(result) == {
        "return_without_rules", "return_evolved", "return_evolved_normal", "return_evolved_swapped",
    }
    assert all(1 <= value <= 500 for value in result.values())
