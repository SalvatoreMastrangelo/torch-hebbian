"""Cue -> value association: a toy task that cannot be solved without plasticity.

An episode has two phases of ``n_cues`` steps each.

* **Study.** Every cue is shown once, in random order, together with its value
  (+1 or -1) on a separate "teacher" input line.
* **Recall.** Every cue is shown once more, in a new random order, with the
  teacher line silent. The network has to answer with the value it saw.

The values are drawn anew for every episode, so fixed weights cannot know them:
whatever a network without plasticity answers at recall, its mean squared error
is at least 1. A network that changes its own weights during the study phase
can store the pairs and bring the error close to 0.

Input at each step: ``n_cues`` one-hot cue lines followed by the teacher line.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

import torch_hebbian as hb


def make_episodes(
    n_episodes: int, n_cues: int = 4, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """Draw a batch of episodes.

    Returns:
        inputs: ``(2 * n_cues, n_episodes, n_cues + 1)``, time first.
        targets: ``(2 * n_cues, n_episodes)``, the value of the cue shown at
            each step (the error is measured on the recall half only).
    """
    values = torch.randint(0, 2, (n_episodes, n_cues), generator=generator) * 2.0 - 1.0
    study_order = torch.rand(n_episodes, n_cues, generator=generator).argsort(dim=1)
    recall_order = torch.rand(n_episodes, n_cues, generator=generator).argsort(dim=1)
    cues = torch.cat([study_order, recall_order], dim=1)  # (episodes, time)
    targets = values.gather(1, cues).T  # (time, episodes)

    inputs = torch.zeros(2 * n_cues, n_episodes, n_cues + 1)
    inputs.scatter_(2, cues.T.unsqueeze(-1), 1.0)
    inputs[:n_cues, :, -1] = targets[:n_cues]  # the teacher speaks only while studying
    return inputs, targets


def run_episodes(model: nn.Module, inputs: Tensor) -> Tensor:
    """Answers of the model at every step, ``(time, n_episodes)``.

    Batch row ``i`` lives through episode ``i``: the plastic weights are reset
    once, then the steps are fed one after the other.
    """
    hb.reset_state(model)
    return torch.stack([model(step).squeeze(-1) for step in inputs])


def recall_error(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error of each episode over its recall phase, ``(n_episodes,)``."""
    recall = slice(inputs.shape[0] // 2, None)
    return (run_episodes(model, inputs)[recall] - targets[recall]).pow(2).mean(dim=0)
