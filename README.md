# torch-hebbian

Hebbian plasticity on top of `torch.nn.Linear`.

`HebbianLinear` is a drop-in `nn.Linear` whose weights change while the network
is used, following per-weight ABCD rules. The rules can be

- **evolved**: read from and written to a flat vector or a `.npy` file, frozen
  with `requires_grad=False`, so that CMA-ES or any other black-box optimiser
  owns them;
- **trained**: left as ordinary parameters, optimised by any torch optimiser
  with gradients flowing through the whole sequence of weight updates.

It depends only on `torch` and `numpy`.

## Install

```bash
pip install git+https://github.com/SalvatoreMastrangelo/torch-hebbian
```

or, from a local clone, `pip install -e path/to/clone`.

Requires Python ≥ 3.10, torch ≥ 2.0 and numpy ≥ 1.24. The test-suite passes
on Python 3.10 with torch 2.12 and numpy 2.2 (CPU and CUDA), and on Python 3.11
with torch 2.0.1 and numpy 1.24 (CPU).

## The rule

After every forward pass, each weight of the layer is updated from the layer's
input `x` and its output `y = W x + b`:

```
ΔW = eta · (A · y xᵀ + B · x + C · y + D)
W  ← (1 − decay) · W + decay · W₀ + ΔW
```

- `A, B, C, D` have one value per weight, shape `(out_features, in_features)`.
- `W₀` is the layer's own `weight` parameter: `decay` pulls the plastic weights
  back to the value the layer has without plasticity. `decay = 0` makes every
  change permanent, `decay = 1` forgets it at the next step.
- `eta` and `decay` are set per layer. The bias is never plastic.
- An optional `w_max` clips the plastic weights to `[−w_max, w_max]`.

## Quickstart

```python
import torch
from torch import nn
import torch_hebbian as hb

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.Tanh(),
    hb.HebbianLinear(32, 4, eta=0.005, decay=0.05),   # plastic layer
)

hb.load_rules(model, "rules.npy")       # rules from a file, requires_grad=False

hb.reset_state(model)                   # start of an episode
with torch.no_grad():
    for obs in episode:                 # obs: (batch, 8), one forward pass per time step
        action = model(obs)
```

Rules start at zero, and with zero rules `HebbianLinear` gives the same outputs
as `nn.Linear`. Plasticity appears once rules are set or trained.

To add plasticity to a network that already exists:

```python
hb.hebbianize(policy, {"actor.4": dict(eta=0.005, decay=0.05)})
```

The named `nn.Linear` layers are replaced in place. The new layers reuse the
original weight and bias parameters, so the network behaves as before and
optimisers that hold those parameters keep working. Only plain `nn.Linear`
layers are accepted, not subclasses.

## How the state works

Think of `HebbianLinear` as an RNN cell whose hidden state is its weight
matrix.

- **One forward pass is one time step.** The output is computed with the
  current plastic weights, then the weights are updated.
- **Every batch row has its own weights.** With input of shape
  `(*batch, in_features)` each index of `batch` is an independent agent. This
  is what vectorised environments need.
- **The state lasts until you reset it.** Call `hb.reset_state(model)` at the
  start of every episode. `hb.reset_state(model, rows=done)` resets only some
  rows of the first batch dimension: `done` is a **boolean** mask, or a list of
  row indices. Integers are always read as indices, so 0/1 flags stored as
  integers or floats are refused with an error: pass `done.bool()`.
- **The batch shape is fixed between resets.** Changing it raises an error
  that tells you to reset.
- **The state is not saved with the model.** It is not part of the
  `state_dict` and it is not a buffer; it follows the module across `.to()`,
  `.cuda()`, `.double()` and `copy.deepcopy`. To keep it, take a snapshot (see
  below).
- A non-finite input (`inf`, `nan`) turns the plastic weights of that row into
  `nan` until the row is reset. Other rows are not affected.
- Plasticity is active in both `train()` and `eval()` mode. Set
  `layer.plastic = False` to pause it (the current plastic weights stay in use).
- `layer.plastic_weight` gives the weights a row currently uses, shape
  `(*batch, out, in)`.

A typical loop with a vectorised environment:

```python
hb.reset_state(policy)
obs = envs.reset()
with torch.no_grad():
    while running:
        obs, reward, done = envs.step(policy(obs))    # done: bool, one per environment
        hb.reset_state(policy, rows=done)             # finished episodes start from W₀ again
```

### Saving and restoring the state

The state of a layer is the deviation of every row's plastic weights from the
layer's `weight`, a tensor of shape `(*batch, out, in)`.

```python
state = hb.get_state(model)     # {"layer name": tensor}, one entry per plastic layer
...                             # more steps
hb.set_state(model, state)      # the model is back at that moment
```

- `get_state` returns copies, and `set_state` copies what it is given: a
  snapshot and the model never share memory.
- A snapshot is an ordinary dict of tensors: `torch.save` stores it, and it can
  be loaded into another copy of the model, on another device.
- The batch shape comes from the snapshot, so indexing it changes which rows
  are in the batch:
  `hb.set_state(model, {name: s[keep] for name, s in state.items()})`.
- Snapshots keep the autograd history of the state, so gradients flow through
  a restored state. Call `.detach()` on the tensors to store plain values.
- A layer that has no state yet (just after a reset) is `None` in the
  snapshot, and restoring `None` resets it.

This is what replaying stored rollouts needs (start each chunk from the state it
had when it was collected), and what resuming or branching in the middle of an
episode needs. `layer.get_state()` and `layer.set_state(tensor)` do the same for
a single layer.

## Evolving the rules

The rules of a model form one flat vector: the plastic layers in
`model.modules()` order, each contributing `A | B | C | D`, each of them
flattened row by row.

```python
n = hb.num_rules(model)                 # length of the vector
hb.set_rules(model, vector)             # shape (n,); copies it, freezes the rules
vector = hb.get_rules(model)            # back as a numpy array
hb.save_rules(model, "rules.npy")
hb.load_rules(model, "rules.npy")       # = set_rules(np.load(...))
hb.rule_slices(model)                   # {"layer name": slice} into the vector
```

`set_rules` and `load_rules` set `requires_grad=False` on `A, B, C, D` and on
nothing else. Pass `freeze=False` to keep them trainable.

A rule file holds the rules and nothing else. The weights are in the
`state_dict`, and `eta`, `decay`, `w_max` are constructor settings: rules are
meaningful only for the settings they were optimised with.

**A whole population in one batch.** Give `set_rules` a `(K, n)` array, or the
list of vectors returned by `es.ask()`, and the layer holds K rule sets at
once. The first batch dimension is then split in K contiguous blocks, and
block k uses rule set k:

```python
es = cma.CMAEvolutionStrategy(np.zeros(hb.num_rules(model)), 1.0, {"popsize": K})
with torch.no_grad():
    while not es.stop():
        candidates = es.ask()
        hb.set_rules(model, candidates)       # K rule sets side by side
        hb.reset_state(model)
        cost = rollout(model, K * S)          # rows k*S ... (k+1)*S-1 belong to candidate k
        es.tell(candidates, cost.reshape(K, S).mean(1).tolist())   # CMA-ES minimises
hb.set_rules(model, es.mean)                  # back to a single rule set
hb.save_rules(model, "rules.npy")
```

While a population is loaded it replaces the `A, B, C, D` parameters, and
`repr(layer)` shows `population=K`. It stays active until a single vector is
set, or until rules are loaded with `load_state_dict`.

Run evolution under `torch.no_grad()`: no graph is built and the weights are
updated in place. Without it, as long as any parameter requires gradients,
autograd records every step until the next full `reset_state`.

## Training the rules by gradient descent

Leave the rules trainable and they are parameters like any other:

```python
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)   # weights and rules
# or only the rules: torch.optim.Adam(hb.rule_parameters(model), lr=1e-3)

hb.reset_state(model)
loss = 0.0
for x, target in sequence:
    loss = loss + criterion(model(x), target)
optimiser.zero_grad()
loss.backward()            # through every weight update of the sequence
optimiser.step()
```

The graph grows with every step until `hb.reset_state(model)`. For long
sequences call `hb.detach_state(model)` every few steps to truncate
backpropagation, as you would detach the hidden state of an RNN.

## API

| | |
|---|---|
| `HebbianLinear(in_features, out_features, bias=True, *, eta=0.005, decay=0.05, w_max=None)` | the plastic layer, a subclass of `nn.Linear` |
| `HebbianLinear.from_linear(linear, **settings)` | plastic layer sharing the parameters of an existing `nn.Linear` |
| `hebbianize(model, layers, **settings)` | replace named `nn.Linear` layers of a model, in place |
| `num_rules(model)`, `rule_slices(model)` | size and layout of the rule vector |
| `get_rules(model)`, `set_rules(model, rules, freeze=True)` | rule vector as numpy array; `(K, n)` for a population |
| `save_rules(model, path)`, `load_rules(model, path, freeze=True)` | the same through a `.npy` file |
| `rule_parameters(model)` | the `A, B, C, D` parameters, for an optimiser |
| `reset_state(model, rows=None)` | plastic weights back to `weight` |
| `get_state(model)`, `set_state(model, state)` | snapshot of the plastic weights as `{layer name: tensor}`, and its restoration |
| `detach_state(model)` | cut the autograd history of the plastic weights |
| `hebbian_layers(model)` | `(name, layer)` for every plastic layer |

Every function takes any `nn.Module`, including a single `HebbianLinear`.

## Examples

The examples also need pycma and Gymnasium (`pip install cma gymnasium`). Run
them from the repository root.

| Script | What it shows | Result |
|---|---|---|
| `examples/quickstart.py` | the whole API, one printed fact per block | a few seconds |
| `examples/evolve_rules_cma.py` | weights trained without plasticity and frozen, rules evolved by CMA-ES, saved and reloaded as `.npy` | recall error 1.00 → 0.005 |
| `examples/train_rules_adam.py` | weights and rules trained together by Adam | recall error 0.0006, against 1.04 with plasticity switched off |
| `examples/cartpole_evolved_rules.py` | CartPole with random frozen weights and evolved rules; in half of the episodes the two actions are swapped without telling the policy | see below, about a minute |

The CMA-ES and Adam examples use a cue → value association task
(`examples/association.py`) in which the values change at every episode, so a
network without plasticity cannot do better than chance (error 1.0).

In the CartPole example no fixed set of weights can handle both the normal and
the swapped case: the best it can reach is 500 steps in one and about 10 in the
other. The evolved rules make the network adapt during the episode:

```
mean return over 40 new episodes (maximum 500)
                          normal   swapped actions
  random frozen weights      9.3       9.6
  + evolved rules          500.0     474.9
```

(Seed 0; seed 1 gives 459 and 500.) Candidates are scored on the worse of
their two cases: scored on the average, the search often settles for mastering
one case only. `--no-flip` runs plain CartPole, where the same network goes
from 9 to 500 steps in 100 generations.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite checks the update against hand-computed numbers, against finite
differences (`gradcheck`), and step by step against the implementation this
library was extracted from. It also runs every example.
