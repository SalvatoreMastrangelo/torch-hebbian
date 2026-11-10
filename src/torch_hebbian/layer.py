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
    batch shape must stay the same between resets. The state is neither a
    parameter nor a buffer, and it is not part of the ``state_dict``.

    Args:
        in_features, out_features, bias: as in ``nn.Linear``.
        eta: Hebbian learning rate.
        decay: fraction of the distance to ``weight`` recovered at every step,
            in ``[0, 1]``. ``0`` makes every change permanent.
        w_max: if given, the plastic weights are clipped to ``[-w_max, w_max]``.

    ``eta``, ``decay`` and ``w_max`` are plain attributes and can be changed at
    any time. They are saved neither in the ``state_dict`` nor in rule files.

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
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.eta = float(eta)
        self.decay = decay
        self.w_max = w_max
        self.plastic = True
        for name in RULE_NAMES:
            rule = torch.zeros(out_features, in_features, device=device, dtype=dtype)
            self.register_parameter(name, nn.Parameter(rule))
        # Per-episode tensors. They are plain attributes rather than buffers, so
        # that code walking ``model.buffers()`` (weight averaging, distributed
        # buffer broadcast) never sees them; ``_apply`` moves them with the module.
        # Plastic state: deviation of every row's weights from ``weight``,
        # shape (rows, out, in). ``None`` means "no deviation yet".
        self._delta: Optional[Tensor] = None
        self._batch_shape: Optional[torch.Size] = None
        self._delta_in_graph = False
        # K rule sets evaluated side by side, shape (K, 4, out, in); see set_rules.
        self._population: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # hyper-parameters
    # ------------------------------------------------------------------

    @property
    def decay(self) -> float:
        return self._decay

    @decay.setter
    def decay(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {value}")
        self._decay = value

    @property
    def w_max(self) -> Optional[float]:
        return self._w_max

    @w_max.setter
    def w_max(self, value: Optional[float]) -> None:
        if value is not None:
            value = float(value)
            if not value > 0.0:
                raise ValueError(f"w_max must be positive (or None for no clipping), got {value}")
        self._w_max = value

    # ------------------------------------------------------------------
    # construction and torch plumbing
    # ------------------------------------------------------------------

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
        text = f"{super().extra_repr()}, eta={self.eta}, decay={self.decay}, w_max={self.w_max}"
        if self._population is not None:
            text += f", population={self._population.shape[0]}"
        return text

    def _apply(self, fn, *args, **kwargs):
        # .to(), .cuda(), .double(), ...: carry the per-episode tensors along
        super()._apply(fn, *args, **kwargs)
        if self._delta is not None:
            self._delta = fn(self._delta)
        if self._population is not None:
            self._population = fn(self._population)
        return self

    def __getstate__(self):
        # Used by copy.deepcopy and pickle. A state that carries autograd
        # history cannot be copied, so the copy receives its values only.
        parent = getattr(super(), "__getstate__", None)
        state = dict(parent() if parent is not None else self.__dict__)
        delta = state.get("_delta")
        if delta is not None and delta.requires_grad:
            state["_delta"] = delta.detach()
        state["_delta_in_graph"] = False
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if any(prefix + name in state_dict for name in RULE_NAMES):
            # Rules are being loaded: they must be the ones that act, so a
            # population left over from an earlier evaluation is dropped.
            self._population = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def get_state(self) -> Optional[Tensor]:
        """The plastic state: deviation of every row's weights from ``weight``.

        Shape ``(*batch, out_features, in_features)``, or ``None`` when the
        layer holds no state (before the first forward pass after a reset).

        The result is a copy: later steps do not change it. It keeps the
        autograd history of the state; call ``.detach()`` on it to store it
        without one.
        """
        if self._delta is None:
            return None
        return self._delta.reshape(*self._batch_shape, self.out_features, self.in_features).clone()

    def set_state(self, state) -> None:
        """Install a plastic state, typically one returned by :meth:`get_state`.

        Args:
            state: tensor or array of shape ``(*batch, out_features,
                in_features)``. Its leading dimensions become the batch shape
                the layer expects from now on, so this can also change which
                rows are in the batch. ``None`` is the same as
                :meth:`reset_state`.

        The state is copied (to the device and dtype of the layer), so the
        tensor passed in is neither modified nor followed afterwards. If it
        carries autograd history, gradients flow back through it.
        """
        self._install_state(self._prepare_state(state))

    def _prepare_state(self, state) -> Optional[Tensor]:
        """Check a state given to :meth:`set_state`; change nothing yet."""
        if state is None:
            return None
        state = torch.as_tensor(state)
        shape = (self.out_features, self.in_features)
        if state.dim() < 2 or tuple(state.shape[-2:]) != shape:
            raise ValueError(
                f"expected a state of shape (*batch, {shape[0]}, {shape[1]}), "
                f"got {tuple(state.shape)}"
            )
        return state.to(device=self.weight.device, dtype=self.weight.dtype)

    def _install_state(self, state: Optional[Tensor]) -> None:
        if state is None:
            self.reset_state()
            return
        self._batch_shape = state.shape[:-2]
        # an own copy, which the plastic update is then free to overwrite
        self._delta = state.reshape(-1, self.out_features, self.in_features).clone()
        self._delta_in_graph = False

    @property
    def weight_delta(self) -> Optional[Tensor]:
        """Same as :meth:`get_state`: the deviation of the plastic weights from
        ``weight``, ``(*batch, out, in)``, or ``None`` when there is no state."""
        return self.get_state()

    @property
    def plastic_weight(self) -> Tensor:
        """Weights the next forward pass will use, ``(*batch, out, in)``.

        A copy. Before the first forward pass after a reset there is no batch
        yet and the result is a copy of ``weight``, ``(out, in)``.
        """
        delta = self.weight_delta
        return self.weight.clone() if delta is None else self.weight + delta

    def detach_state(self) -> None:
        """Cut the autograd history of the plastic weights (truncated BPTT)."""
        if self._delta is not None:
            self._delta = self._delta.detach()

    def reset_state(self, rows=None) -> None:
        """Put the plastic weights back to ``weight``.

        Args:
            rows: which entries of the first batch dimension to reset. Either a
                **boolean** mask (tensor, array or list), or integer row
                indices (an int, a slice, or a tensor, array, list or tuple of
                ints). ``None`` resets everything and also forgets the batch
                shape, so the next forward pass may use a different one.

        Integers are always read as indices. Environments often report
        finished episodes as 0/1 integers or floats: those are refused rather
        than guessed at, convert them with ``.bool()``.
        """
        if rows is None:
            self._delta = None
            self._batch_shape = None
            self._delta_in_graph = False
            return
        if self._delta is None:
            return
        if len(self._batch_shape) == 0:
            raise ValueError("reset_state(rows=...) needs a batch dimension to pick rows from")
        self._leave_inference_mode()
        row_ids = self._row_ids(rows)
        if row_ids.numel() == 0:
            return
        if self._state_is_private():
            self._delta.index_fill_(0, row_ids, 0.0)
        else:
            self._delta = self._delta.index_fill(0, row_ids, 0.0)
            self._delta_in_graph = False

    def _row_ids(self, rows) -> Tensor:
        """Indices along the first axis of ``_delta`` selected by ``rows``."""
        device = self._delta.device
        if not isinstance(rows, (int, slice)):
            rows = torch.as_tensor(rows, device=device)
            if rows.dtype != torch.bool:
                if rows.numel() == 0:
                    return torch.zeros(0, dtype=torch.long, device=device)
                if rows.is_floating_point() or rows.is_complex():
                    raise TypeError(
                        f"reset_state(rows=...) takes a boolean mask or integer row indices, "
                        f"got dtype {rows.dtype}. For 0/1 flags pass rows.bool()."
                    )
                flat = rows.reshape(-1).long()
                one_per_row = rows.dim() == 1 and flat.numel() == self._batch_shape[0]
                if one_per_row and bool(((flat == 0) | (flat == 1)).all()):
                    raise ValueError(
                        "reset_state(rows=...) received one 0/1 integer per row. Integers are "
                        "read as row indices, so this is ambiguous: pass rows.bool() for a mask."
                    )
                if flat.unique().numel() != flat.numel():
                    raise ValueError(
                        "reset_state(rows=...) received repeated row indices. If these are "
                        "0/1 flags, pass rows.bool() for a mask."
                    )
                rows = rows.long()
        row_ids = torch.arange(self._delta.shape[0], device=device).reshape(self._batch_shape)
        return row_ids[rows].reshape(-1)

    def _leave_inference_mode(self) -> None:
        """Replace tensors created under ``torch.inference_mode()`` by ordinary
        copies once we are outside it: inference tensors can neither be updated
        in place nor take part in a graph there."""
        if torch.is_inference_mode_enabled():
            return
        if self._delta is not None and self._delta.is_inference():
            self._delta = self._delta.clone()
            self._delta_in_graph = False
        if self._population is not None and self._population.is_inference():
            self._population = self._population.clone()

    def _state_is_private(self) -> bool:
        """Whether ``_delta`` may be overwritten in place, i.e. autograd cannot
        need its current values any more."""
        return not (self._delta.requires_grad or self._delta_in_graph)

    # ------------------------------------------------------------------
    # rules
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # forward pass and plastic update
    # ------------------------------------------------------------------

    def forward(self, input: Tensor) -> Tensor:
        self._leave_inference_mode()
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
