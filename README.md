# randpick

Weighted random sampling helpers in pure Python — Vose's alias method, cumulative-bisect, Efraimidis-Spirakis A-Res. Zero dependencies.

`randpick` covers the four common shapes of weighted-random work:

- One item, one shot — `weighted_choice`.
- Many items, one shot — `weighted_choices` (with replacement) and `weighted_sample` (without replacement).
- Many items over and over — `WeightedPicker`, which builds an alias table once in O(n) and draws each item in O(1).
- Manual cumulative path — `cumulative_pick` for callers who already maintain a cumulative-weight prefix.

## Features

- Vose's alias method (`WeightedPicker`) — O(n) build, O(1) per pick. Stable under floating-point noise.
- Cumulative-bisect helpers (`weighted_choice`, `weighted_choices`) — O(n) build, O(log n) per pick. No allocation.
- Efraimidis-Spirakis A-Res sampling without replacement (`weighted_sample`) — O(n log k) total. Items with zero weight are excluded.
- Injectable `random.Random` for deterministic tests; bare module default for one-off use.
- Strict input validation via a precise error hierarchy rooted at `ValueError`.
- Type-hinted everywhere; `mypy --strict` clean.
- Pure Python; works on CPython 3.10+.

## Install

```bash
pip install randpick
```

## Usage

```python
import randpick

# One-off draws
favourite = randpick.weighted_choice(["red", "green", "blue"], [1, 3, 6])

# k draws with replacement
spins = randpick.weighted_choices(
    ["common", "rare", "epic"],
    [80, 18, 2],
    k=5,
)

# k distinct draws without replacement
hand = randpick.weighted_sample(
    ["A", "K", "Q", "J", "10"],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    k=3,
)

# Reusable picker — pay O(n) once, draw O(1) per call
picker = randpick.WeightedPicker(["a", "b", "c"], [0.1, 0.7, 0.2])
for _ in range(1_000_000):
    item = picker.pick()
```

### Deterministic draws

Pass any `random.Random` (or anything that implements `random()` and `randrange(n)`):

```python
import random
import randpick

rng = random.Random(0xC0FFEE)
picker = randpick.WeightedPicker(["a", "b", "c"], [1, 2, 3], rng=rng)
sequence = [picker.pick() for _ in range(20)]
```

### Cumulative pick

When the caller already maintains a cumulative-weight prefix:

```python
items = ("apple", "banana", "cherry")
cum = [1.0, 3.0, 6.0]  # weights = 1, 2, 3
choice = randpick.cumulative_pick(cum, items)
```

## API

### `class WeightedPicker(items, weights, *, rng=None)`

Builds an alias table once in O(n). Subsequent draws are O(1).

| Method / property | Description |
| --- | --- |
| `pick()` | Draw a single item. |
| `sample(k, *, replace=True)` | Draw `k` items. With `replace=False`, falls back to `weighted_sample`. |
| `items` | The population, in registration order, as a tuple. |
| `weights` | The original weights as floats, in registration order. |
| `size` / `len(picker)` | Population size. |

### Top-level helpers

- `weighted_choice(items, weights, *, rng=None) -> item`
- `weighted_choices(items, weights, k, *, rng=None) -> list[item]`
- `weighted_sample(items, weights, k, *, rng=None) -> list[item]`
- `cumulative_pick(cum_weights, items, *, rng=None) -> item`

### Errors

All errors inherit from `RandPickError`, which inherits from `ValueError`:

- `EmptyPopulationError` — empty `items`.
- `WeightLengthMismatchError` — `len(items) != len(weights)`.
- `NegativeWeightError` — negative or non-finite weight.
- `ZeroTotalWeightError` — sum of weights is zero.
- `InvalidSampleSizeError` — invalid `k` (negative, non-int, or > population for sampling without replacement).

## Running tests

```bash
pip install pytest pytest-cov mypy
pytest tests --cov=randpick --cov-branch
mypy --strict src/randpick
```

The suite has 66 tests across five files. Coverage: 100% line / 100% branch on every module.

## Non-goals

- Streaming reservoir sampling (no fixed-`k`-from-stream helper).
- Dirichlet, posterior, or other parametric samplers.
- Continuous distributions.

## License

MIT — see [LICENSE](LICENSE).
