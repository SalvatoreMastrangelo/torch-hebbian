"""``HebbianLinear``: an ``nn.Linear`` whose weights change while it is used."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RULE_NAMES = ("A", "B", "C", "D")


class HebbianLinear(nn.Linear):
    r"""Linear layer with Hebbian plasticity (ABCD rule) on its weights.

    After every forward pass the weights used by that pass are updated from the
    layer's input ``x`` and its output ``y = W x + b``:

    .. math::
        \Delta W = \eta\,(A \odot y x^\top + B \odot x + C \odot y + D)

        W \leftarrow (1 - \text{decay})\,W + \text{decay}\,W_0 + \Delta W

    ``W_0`` is the layer's ordinary ``weight`` parameter, so the decay pulls the
    plastic weights back to the value the layer would have without plasticity.
    The bias is never plastic.

    **Rules.** ``A``, ``B``, ``C``, ``D`` are parameters of shape
    ``(out_features, in_features)``, one value per weight, initialised to zero:
    a new layer behaves exactly like ``nn.Linear`` until its rules are set
    (:func:`torch_hebbian.set_rules`, :func:`torch_hebbian.load_rules`) or
    trained. Left trainable, they receive gradients through the whole sequence
    of forward passes.

    **State.** Every batch row owns its plastic weights, like the hidden state
    of an RNN cell: one forward pass is one time step, and the weights of a row
    persist from one call to the next until :meth:`reset_state`. With input of
    shape ``(*batch, in_features)`` every index of ``batch`` is a row. The
    batch shape must stay the same between resets. The state is not part of
    the ``state_dict``.

    Args:
        in_features, out_features, bias: as in ``nn.Linear``.
        eta: Hebbian learning rate.
        decay: fraction of the distance to ``weight`` recovered at every step,
            in ``[0, 1]``. ``0`` makes every change permanent.
        w_max: if given, the plastic weights are clipped to ``[-w_max, w_max]``.

    Attributes:
        plastic (bool): set to ``False`` to stop the weights from changing;
            the current plastic weights keep being used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        eta: float = 0.005,
        decay: float = 0.05,
        w_max: Optional[float] = None,
        device=None,
        dtype=None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if w_max is not None and w_max <= 0.0:
            raise ValueError(f"w_max must be positive (or None for no clipping), got {w_max}")
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.eta = float(eta)
        self.decay = float(decay)
        self.w_max = None if w_max is None else float(w_max)
        self.plastic = True
        for name in RULE_NAMES:
            rule = torch.zeros(out_features, in_features, device=device, dtype=dtype)
            self.register_parameter(name, nn.Parameter(rule))
        # Both tensors below are non-persistent buffers: they follow the module
        # across devices and dtypes but stay out of the state_dict.
        # Plastic state: deviation of every row's weights from ``weight``,
        # shape (rows, out, in). ``None`` means "no deviation yet".
        self.register_buffer("_delta", None, persistent=False)
        self._batch_shape: Optional[torch.Size] = None
        self._delta_in_graph = False
        # K rule sets evaluated side by side, shape (K, 4, out, in); see set_rules.
        self.register_buffer("_population", None, persistent=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **settings) -> "HebbianLinear":
        """A plastic layer that starts out computing exactly what ``linear`` does.

        The new layer *shares* the weight and bias parameters of ``linear``
        (as ``nn.SyncBatchNorm.convert_sync_batchnorm`` does), so optimisers
        and references that hold them stay valid.

        Args:
            linear: the layer to take the weights from.
            **settings: ``eta``, ``decay``, ``w_max`` as in the constructor.
        """
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **settings,
        )
        layer.weight = linear.weight
        layer.bias = linear.bias
        layer.train(linear.training)
        return layer

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, eta={self.eta}, decay={self.decay}, w_max={self.w_max}"

    def __getstate__(self):
        # Used by copy.deepcopy and pickle. A state that carries autograd
        # history cannot be copied, so the copy receives its values only.
        parent = getattr(super(), "__getstate__", None)
        state = dict(parent() if parent is not None else self.__dict__)
        delta = state["_buffers"].get("_delta")
        if delta is not None and delta.requires_grad:
            state["_buffers"] = state["_buffers"].copy()
            state["_buffers"]["_delta"] = delta.detach()
        state["_delta_in_graph"] = False
        return state

    @property
    def weight_delta(self) -> Optional[Tensor]:
        """Deviation of the plastic weights from ``weight``, ``(*batch, out, in)``.

        ``None`` before the first forward pass after a reset.
        """
        if self._delta is None:
            return None
        return self._delta.reshape(*self._batch_shape, self.out_features, self.in_features).clone()

    @property
    def plastic_weight(self) -> Tensor:
        """Weights the next forward pass will use, ``(*batch, out, in)``.

        Before the first forward pass after a reset there is no batch yet and
        this is ``weight`` itself, ``(out, in)``.
        """
        delta = self.weight_delta
        return self.weight if delta is None else self.weight + delta

    @property
    def num_rules(self) -> int:
        """Length of this layer's rule vector: ``A``, ``B``, ``C``, ``D`` per weight."""
        return len(RULE_NAMES) * self.out_features * self.in_features

    def _read_rules(self) -> Tensor:
        """The rules as one vector ``A|B|C|D``, each flattened row-major.

        Shape ``(num_rules,)``, or ``(K, num_rules)`` while a population of K
        rule sets is loaded.
        """
        if self._population is not None:
            return self._population.reshape(self._population.shape[0], -1)
        return torch.cat([getattr(self, name).detach().reshape(-1) for name in RULE_NAMES])

    def _write_rules(self, rules: Tensor) -> None:
        """Inverse of :meth:`_read_rules`.

        A vector goes into the ``A, B, C, D`` parameters. A ``(K, num_rules)``
        matrix becomes a population, which takes precedence over the
        parameters until a vector is written again.
        """
        rules = rules.to(device=self.weight.device, dtype=self.weight.dtype)
        shape = (len(RULE_NAMES), self.out_features, self.in_features)
        if rules.dim() == 2:
            self._population = rules.reshape(rules.shape[0], *shape).clone()
            return
        self._population = None
        with torch.no_grad():
            for name, block in zip(RULE_NAMES, rules.reshape(shape)):
                getattr(self, name).copy_(block)

    def detach_state(self) -> None:
        """Cut the autograd history of the plastic weights (truncated BPTT)."""
        if self._delta is not None:
            self._delta = self._delta.detach()

    def reset_state(self, rows=None) -> None:
        """Put the plastic weights back to ``weight``.

        Args:
            rows: which entries of the first batch dimension to reset: a boolean
                mask, a list or tensor of indices, a slice. ``None`` resets
                everything and also forgets the batch shape, so the next
                forward pass may use a different one.
        """
        if rows is None:
            self._delta = None
            self._batch_shape = None
            self._delta_in_graph = False
            return
        delta = self._delta
        if delta is None:
            return
        if len(self._batch_shape) == 0:
            raise ValueError("reset_state(rows=...) needs a batch dimension to pick rows from")
        if isinstance(rows, Tensor):
            rows = rows.to(delta.device)
        row_ids = torch.arange(delta.shape[0], device=delta.device).reshape(self._batch_shape)
        row_ids = row_ids[rows].reshape(-1)
        if self._state_is_private():
            delta.index_fill_(0, row_ids, 0.0)
        else:
            self._delta = delta.index_fill(0, row_ids, 0.0)
            self._delta_in_graph = False

    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.bias)
        batch_shape = input.shape[:-1]
        x = input.reshape(-1, self.in_features)
        if self._delta is not None:
            if batch_shape != self._batch_shape:
                raise RuntimeError(
                    f"HebbianLinear holds plastic weights for a batch of shape "
                    f"{tuple(self._batch_shape)} but received a batch of shape "
                    f"{tuple(batch_shape)}. Call reset_state() before changing the batch."
                )
            plastic = torch.bmm(self._delta, x.unsqueeze(2)).squeeze(2)
            output = output + plastic.reshape(output.shape)
            if output.requires_grad:
                # the graph of this output may hold a reference to the state
                self._delta_in_graph = True
        if self.plastic:
            self._update(x, output.reshape(-1, self.out_features), batch_shape)
        return output

    def _rules(self, batch_shape: torch.Size):
        """``A, B, C, D`` broadcastable to ``(K, rows // K, out, in)``, and K."""
        population = self._population
        if population is None:
            return self.A, self.B, self.C, self.D, 1
        sets = population.shape[0]
        first = batch_shape[0] if len(batch_shape) else 1
        if first % sets:
            raise RuntimeError(
                f"HebbianLinear holds {sets} rule sets, so the first batch dimension "
                f"must be a multiple of {sets}; got {first}."
            )
        A, B, C, D = population.unsqueeze(2).unbind(1)  # each (K, 1, out, in)
        return A, B, C, D, sets

    def _update(self, x: Tensor, y: Tensor, batch_shape: torch.Size) -> None:
        A, B, C, D, sets = self._rules(batch_shape)
        # Rule set k owns the k-th contiguous block of rows.
        x = x.reshape(sets, -1, 1, self.in_features)
        y = y.reshape(sets, -1, self.out_features, 1)
        shape = (-1, self.out_features, self.in_features)
        old = self._delta
        trainable_rules = self._population is None and any(r.requires_grad for r in (A, B, C, D))
        if torch.is_grad_enabled() and (y.requires_grad or trainable_rules):
            # Differentiable path: every step adds to the autograd graph.
            delta = self.eta * (A * (y * x) + B * x + C * y + D).reshape(shape)
            if old is not None:
                delta = delta + (1.0 - self.decay) * old
            if self.w_max is not None:
                delta = torch.clamp(delta, -self.w_max - self.weight, self.w_max - self.weight)
        else:
            # Nothing to differentiate: same arithmetic with a single temporary.
            with torch.no_grad():
                change = (y * x).mul_(A).addcmul_(B, x).addcmul_(C, y).add_(D).reshape(shape)
                if old is None:
                    delta = change.mul_(self.eta)
                else:
                    # overwrite the state when nobody else can be reading it
                    decayed = old.mul_ if self._state_is_private() else old.mul
                    delta = decayed(1.0 - self.decay).add_(change, alpha=self.eta)
                if self.w_max is not None:
                    delta.clamp_(-self.w_max - self.weight, self.w_max - self.weight)
        if delta is not old:
            self._delta_in_graph = False
        self._delta = delta
        self._batch_shape = batch_shape

    def _state_is_private(self) -> bool:
        """Whether ``_delta`` may be overwritten in place.

        Not if autograd may still need its current values, and not if it was
        created under ``torch.inference_mode()`` and we have since left it.
        """
        delta = self._delta
        return not (
            delta.requires_grad
            or self._delta_in_graph
            or (delta.is_inference() and not torch.is_inference_mode_enabled())
        )
