# PyEvo — Correctness and Packaging Fixes

This document records defects found in PyEvo and what was done about them.
Everything listed as fixed has a regression test in `tests/test_regressions.py`.

## Correctness bugs

### DE never optimized anything
`DE.tell()` built trial vectors, stored them on `self.trial_vectors`, and then
returned. `_selection()` — the method that would have replaced parents with
better offspring — was never called from anywhere, and `self.population` was
never reassigned after initialization. `ask()` returned the same initial random
population on every generation, so DE's fitness was flat forever.

Fixed by giving DE a real ask/tell cycle: the first `ask()` returns the initial
population to be scored, every later `ask()` returns fresh trial vectors, and
`tell()` performs greedy one-to-one selection against the stored parent
fitnesses.

### CMA-ES produced NaN in low dimensions
`cc` and `cs` were both set to `4/n`, which exceeds 1 for `n < 4`. That inverts
the sign of the evolution-path cumulation and puts a negative value under the
`sqrt(cc * (2 - cc))` term, yielding NaN at `n = 1` and a collapsed step size at
`n = 2`. Replaced with the canonical expressions, which stay in `(0, 1)` for
every dimensionality.

### CMA-ES was not really CMA-ES
Two of the algorithm's defining mechanisms were missing:

- `cmu` was hardcoded to `0.0`, disabling the rank-mu covariance update.
- The recombination weights covered all `lambda` samples with positive weight,
  so the *worst* solutions still pulled the mean toward them. Canonical CMA-ES
  truncates at `mu = lambda / 2`.

Both are now implemented, along with the `hsig` variance correction and an
eigendecomposition cadence that scales with how fast `C` actually changes
(previously a fixed "every 10 generations", which let `B` and `D` drift
arbitrarily far from `C`). On a rotated ellipsoid with condition number 1e6,
this reaches a target of 1e-10 in a median of 538 generations versus 977 before.

### Simulated Annealing reported zero improvement
`improvement` was computed *after* `self.best_fitness` had already been
reassigned to the new value, so the expression was identically zero whenever a
new best was found. Early stopping therefore triggered precisely when the search
was making progress. `min_temp` was also stored but never enforced, letting the
temperature decay toward zero and divide into the acceptance probability. The
default `initial_temp` of 100 made the search a near-pure random walk on
typically-scaled objectives; it is now 1.0.

### SNES silently shrank sigma on every resume
`__init__` multiplied any caller-supplied `sigma` by `alpha`. Because
`load_state()` passes the saved sigma back through `__init__`, every
save/load round trip shrank sigma by a factor of `alpha` (0.05 by default) —
a silent 20x change in search scale per resume. An explicit `sigma` is now used
verbatim; `alpha` only scales the default.

### Checkpoints restored the wrong optimizer
`InteractiveOptimizer.load_session()` inferred the optimizer type from state
keys and knew only SNES, CMA-ES, and PSO — everything else silently came back as
SNES, changing the algorithm mid-run. Checkpoints now record the class that
wrote them. The RNG state was also dropped, so resumed runs did not reproduce
the original random stream; it is now saved and restored.

### PSO reported its best fitness under a different key
Every other optimizer reports `best_fitness` from `get_stats()`; PSO reported
only `global_best_fitness`. `optimize_with_acceleration` reads `best_fitness`
with a default of 0, so PSO runs recorded a best fitness of 0 throughout.

### Early stopping fired on the first flat generation
`optimize_with_acceleration` broke out of the loop the first time `improvement`
fell below a hardcoded `1e-8`. For a stochastic optimizer a single flat or
negative generation is entirely normal, so runs terminated after a handful of
iterations. Tolerance and patience are now parameters, and stopping requires
`patience` *consecutive* stalled generations.

## Interface consistency

- `SimulatedAnnealing` rejected `population_count`, so it could not be used
  interchangeably with the other optimizers. It now accepts it and proposes that
  many neighbours per step.
- `PSO` had no `load_state`, so it could not be resumed at all. `load_state` is
  now part of the `Optimizer` contract and all seven implement it.
- `save_state` wrote `None` into `np.savez` for absent optional fields, creating
  object arrays that `np.load` refuses to read without `allow_pickle=True`.
  Saving a fresh optimizer therefore produced an unloadable file. Optional
  fields now use `pack_optional`/`unpack_optional` in `pyevo.optimizers.base`.
- `DE`, `GA`, and `CEM` could not round-trip at all with the default
  `bounds=None`, for the same reason.

## Packaging

- `install_requires` listed only NumPy, but `import pyevo` reached
  `pyevo/utils/image.py`, which imported Pillow unconditionally. A clean
  `pip install pyevo` followed by `import pyevo` raised `ModuleNotFoundError`.
  Pillow is now imported lazily at its single use site.
- `python_requires` claimed 3.6+, but the codebase uses PEP 585 builtin generics
  (`dict[str, Any]`) in evaluated annotations, which need 3.9+.
- The `all` extra pulled in a CUDA-specific CuPy wheel, so it failed to install
  on any machine without CUDA. GPU support is now only in the `gpu` extra.
- `scikit-image` was allowed from 0.18, but `scipy_ssim` passed `multichannel=`,
  which was deprecated in 0.19 and removed in 0.23. Now uses `channel_axis`.

## Performance and memory

- `SNES.tell()` was labelled "vectorized" but looped over `solution_length` in
  Python. At 5,000 dimensions this took 46 ms per generation; the vectorized
  form takes under 1 ms for bit-identical results. `ask()` and the utility
  weights were vectorized too.
- `parallel_evaluate` passed a locally-defined closure to
  `ProcessPoolExecutor`, which cannot pickle local objects — so the parallel
  path raised `AttributeError: Can't pickle local object` every time it ran. It
  now uses a module-level worker with `functools.partial`, and falls back to
  serial evaluation with a warning if the fitness function itself is
  unpicklable.
- `CEM`'s `diagonal_cov=True` still allocated a full `n x n` covariance matrix,
  defeating the option's entire purpose (400 MB at n=10,000). It now stores a
  length-`n` variance vector.
- `CEM`'s default `population_count` of `10 * solution_length` was uncapped,
  asking for 50,000 evaluations per generation on a 5,000-parameter problem. Now
  capped at 200, matching DE's existing treatment.
- `CEM.get_stats()` reported `cov_determinant`, which underflows to exactly 0 in
  even moderate dimensions. Replaced with `cov_logdet`.

## Housekeeping

- Deleted `pyevo/utils.py`, a dead module shadowed by the `pyevo/utils/`
  package. (The previous version of this document claimed this had been done.)
- `DEFAULT_OUTPUT_DIR` was `"examples/output"` — a repo-relative path baked into
  library code, which created an `examples/` tree in the working directory of
  any program using it. Now `"output"`.
- `image_approximation.py` joined `output_dir` into the default output paths and
  then joined it again at every save site, producing
  `examples/output/examples/output/...`. The artifacts of that bug were
  committed to the repository; they have been moved to the correct path.
- All eight examples had a `sys.path` fallback pointing one directory too
  shallow (`..` instead of `../..`), so running any example from a checkout
  without installing the package failed with `ModuleNotFoundError`.
- Added GitHub Actions CI across Python 3.9–3.12. It installs with core
  dependencies only and asserts `import pyevo` works before running the suite,
  then repeats with the optional extras installed. The test suite was red at the
  previous commit; the absence of CI is why that went unnoticed.
