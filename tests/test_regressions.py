"""Regression tests for previously-fixed defects.

Each test here corresponds to a specific bug. They are deliberately written
against observable behaviour rather than implementation details, so they keep
holding if the internals are rewritten again.
"""

import os
import tempfile
import unittest

import numpy as np

from pyevo import (
    SNES, CMA_ES, PSO, DE, SimulatedAnnealing, GeneticAlgorithm,
    CrossEntropyMethod, parallel_evaluate, save_checkpoint, load_checkpoint,
    apply_checkpoint,
)
from pyevo.utils import create_optimizer


ALL_OPTIMIZERS = [
    SNES, CMA_ES, PSO, DE, SimulatedAnnealing, GeneticAlgorithm, CrossEntropyMethod
]

OPTIMIZER_NAMES = ["snes", "cmaes", "pso", "de", "sa", "ga", "cem"]


def sphere(x):
    """Maximization form of the sphere function (optimum 0 at the origin)."""
    return -float(np.sum(np.asarray(x) ** 2))


def _module_level_fitness(x):
    """Top-level so it stays picklable for the parallel-evaluation test."""
    return float(np.sum(np.asarray(x) ** 2))


class TestOptimizersActuallyOptimize(unittest.TestCase):
    """Every optimizer must measurably improve on a trivial problem.

    DE used to build trial vectors and then throw them away, so its population
    never changed and it returned the same fitness forever.
    """

    def test_all_optimizers_improve_on_sphere(self):
        for cls in ALL_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                opt = cls(solution_length=10, population_count=20, random_seed=0)
                first = None
                best = -np.inf
                for _ in range(80):
                    solutions = opt.ask()
                    fitnesses = [sphere(s) for s in solutions]
                    opt.tell(fitnesses)
                    if first is None:
                        first = max(fitnesses)
                    best = max(best, max(fitnesses))
                self.assertGreater(
                    best, first,
                    f"{cls.__name__} did not improve over 80 generations",
                )

    def test_de_population_changes_between_generations(self):
        opt = DE(solution_length=8, population_count=16, random_seed=0)
        initial = opt.ask().copy()
        for _ in range(10):
            solutions = opt.ask()
            opt.tell([sphere(s) for s in solutions])
        self.assertFalse(
            np.array_equal(initial, opt.ask()),
            "DE population is identical after 10 generations",
        )


class TestOptimizerInterface(unittest.TestCase):
    """All optimizers share one constructor and state contract."""

    def test_create_optimizer_accepts_population_count(self):
        # SimulatedAnnealing used to reject population_count outright.
        for name in OPTIMIZER_NAMES:
            with self.subTest(optimizer=name):
                opt = create_optimizer(
                    name, solution_length=10, population_count=12, random_seed=0
                )
                solutions = opt.ask()
                self.assertEqual(len(solutions), 12)
                opt.tell([sphere(s) for s in solutions])

    def test_every_optimizer_implements_load_state(self):
        for cls in ALL_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                self.assertTrue(
                    hasattr(cls, "load_state"),
                    f"{cls.__name__} cannot be resumed from disk",
                )

    def test_save_load_roundtrip_preserves_best_solution(self):
        for cls in ALL_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                opt = cls(solution_length=6, population_count=10, random_seed=1)
                for _ in range(3):
                    solutions = opt.ask()
                    opt.tell([sphere(s) for s in solutions])
                before = opt.get_best_solution().copy()

                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "state.npz")
                    opt.save_state(path)
                    restored = cls.load_state(path)

                np.testing.assert_allclose(
                    restored.get_best_solution(), before, rtol=1e-5,
                    err_msg=f"{cls.__name__} lost state across save/load",
                )

    def test_save_load_before_first_tell(self):
        # Saving a fresh optimizer used to write None into the npz, which then
        # failed to load with "Object arrays cannot be loaded".
        for cls in ALL_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                opt = cls(solution_length=5, population_count=8, random_seed=1)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "fresh.npz")
                    opt.save_state(path)
                    cls.load_state(path)  # must not raise


class TestSNES(unittest.TestCase):
    def test_explicit_sigma_is_not_rescaled_by_alpha(self):
        sigma = np.full(4, 2.0)
        opt = SNES(solution_length=4, alpha=0.05, sigma=sigma, random_seed=0)
        np.testing.assert_allclose(opt.sigma, sigma, rtol=1e-6)

    def test_save_load_preserves_sigma_exactly(self):
        # load_state used to re-apply alpha, shrinking sigma by 20x per resume.
        opt = SNES(solution_length=4, random_seed=0)
        opt.tell(np.random.RandomState(0).rand(opt.population_count))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snes.npz")
            opt.save_state(path)
            restored = SNES.load_state(path)
        np.testing.assert_allclose(restored.sigma, opt.sigma, rtol=1e-6)

    def test_tell_matches_reference_implementation(self):
        """The vectorized update must equal the original per-dimension loop."""
        opt = SNES(solution_length=12, population_count=10, random_seed=4)
        opt.ask()
        fitnesses = np.random.RandomState(1).rand(opt.population_count)

        center, sigma = opt.center.copy(), opt.sigma.copy()
        gaussians, utilities = opt.gaussians.copy(), opt.utility_weights.copy()
        indices = np.argsort(-fitnesses)

        for j in range(opt.solution_length):
            noises = gaussians[indices, j]
            center[j] += opt.eta_center * sigma[j] * np.sum(utilities * noises)
            sigma[j] *= np.exp(0.5 * opt.eta_sigma * np.sum(utilities * (noises ** 2 - 1)))

        opt.tell(fitnesses)
        np.testing.assert_allclose(opt.center, center, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(opt.sigma, sigma, rtol=1e-4, atol=1e-6)


class TestCMAES(unittest.TestCase):
    def test_no_nan_in_low_dimensions(self):
        # cc = cs = 4/n exceeded 1 below n=4 and produced NaN at n=1.
        for n in (1, 2, 3, 4, 5):
            with self.subTest(solution_length=n):
                opt = CMA_ES(solution_length=n, population_count=8, random_seed=0)
                for _ in range(30):
                    solutions = opt.ask()
                    opt.tell([sphere(s) for s in solutions])
                best = opt.get_best_solution()
                self.assertFalse(np.isnan(best).any(), f"NaN solution at n={n}")
                self.assertFalse(np.isnan(opt.sigma), f"NaN sigma at n={n}")

    def test_learning_rates_are_in_valid_range(self):
        for n in (1, 2, 3, 10, 100):
            with self.subTest(solution_length=n):
                opt = CMA_ES(solution_length=n, population_count=8, random_seed=0)
                for name in ("cc", "cs", "c1", "cmu"):
                    rate = getattr(opt, name)
                    self.assertTrue(
                        0 <= rate <= 1, f"{name}={rate} out of range at n={n}"
                    )
                self.assertLessEqual(opt.c1 + opt.cmu, 1.0)

    def test_rank_mu_update_is_enabled(self):
        # cmu was hardcoded to 0, removing the rank-mu update entirely.
        opt = CMA_ES(solution_length=10, population_count=12, random_seed=0)
        self.assertGreater(opt.cmu, 0.0)

    def test_recombination_uses_only_the_better_half(self):
        opt = CMA_ES(solution_length=10, population_count=12, random_seed=0)
        self.assertEqual(opt.mu, 6)
        self.assertEqual(len(opt.weights), opt.mu)
        self.assertTrue(np.all(opt.weights > 0))
        np.testing.assert_allclose(np.sum(opt.weights), 1.0, rtol=1e-6)

    def test_solves_rotated_ill_conditioned_problem(self):
        """Full covariance adaptation should handle a rotated ellipsoid.

        This is what distinguishes CMA-ES from a separable method, and it only
        works with the rank-mu update in place.
        """
        n = 8
        rs = np.random.RandomState(0)
        rotation, _ = np.linalg.qr(rs.randn(n, n))
        scale = 10 ** (5 * np.arange(n) / (n - 1))

        def rotated_ellipsoid(x):
            y = rotation @ np.asarray(x)
            return -float(np.sum(scale * y ** 2))

        opt = CMA_ES(solution_length=n, population_count=12, random_seed=1,
                     center=np.ones(n), sigma=1.0)
        best = -np.inf
        for _ in range(1500):
            solutions = opt.ask()
            fitnesses = [rotated_ellipsoid(s) for s in solutions]
            opt.tell(fitnesses)
            best = max(best, max(fitnesses))
        self.assertGreater(best, -1e-8)

    def test_covariance_stays_symmetric(self):
        opt = CMA_ES(solution_length=6, population_count=10, random_seed=0)
        for _ in range(40):
            solutions = opt.ask()
            opt.tell([sphere(s) for s in solutions])
        np.testing.assert_allclose(opt.C, opt.C.T, rtol=1e-10, atol=1e-12)


class TestSimulatedAnnealing(unittest.TestCase):
    def test_improvement_is_nonzero_when_best_improves(self):
        # improvement was computed after best_fitness was reassigned, so it was
        # always 0 and tripped early stopping exactly when progress was made.
        opt = SimulatedAnnealing(solution_length=5, random_seed=0)
        improvements = []
        for _ in range(200):
            solutions = opt.ask()
            improvements.append(opt.tell([sphere(s) for s in solutions]))
        finite = [i for i in improvements[1:] if np.isfinite(i)]
        self.assertTrue(any(i > 0 for i in finite),
                        "SA never reported a positive improvement")

    def test_temperature_never_falls_below_min_temp(self):
        opt = SimulatedAnnealing(solution_length=4, min_temp=1e-3,
                                 cooling_factor=0.5, random_seed=0)
        for _ in range(200):
            solutions = opt.ask()
            opt.tell([sphere(s) for s in solutions])
        self.assertGreaterEqual(opt.temperature, 1e-3)


class TestCEM(unittest.TestCase):
    def test_diagonal_cov_uses_linear_memory(self):
        # diagonal_cov stored a full n x n matrix, defeating its own purpose.
        n = 64
        opt = CrossEntropyMethod(solution_length=n, population_count=20,
                                 diagonal_cov=True, random_seed=0)
        self.assertEqual(opt.cov.shape, (n,))
        for _ in range(10):
            solutions = opt.ask()
            opt.tell([sphere(s) for s in solutions])
        self.assertEqual(opt.cov.shape, (n,))


class TestParallelEvaluation(unittest.TestCase):
    def test_parallel_evaluate_runs_in_multiple_processes(self):
        # The worker used to be a local closure, which ProcessPoolExecutor
        # could not pickle, so this path raised every time it was used.
        solutions = [np.arange(4, dtype=float) + i for i in range(6)]
        result = parallel_evaluate(_module_level_fitness, solutions, max_workers=2)
        expected = [_module_level_fitness(s) for s in solutions]
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_falls_back_when_fitness_function_is_unpicklable(self):
        solutions = [np.arange(3, dtype=float) + i for i in range(4)]
        unpicklable = lambda x: float(np.sum(x))  # noqa: E731
        result = parallel_evaluate(unpicklable, solutions, max_workers=2)
        np.testing.assert_allclose(result, [float(np.sum(s)) for s in solutions])

    def test_empty_input(self):
        self.assertEqual(parallel_evaluate(_module_level_fitness, [], max_workers=2), [])


class TestCheckpointing(unittest.TestCase):
    def test_checkpoint_restores_rng_stream(self):
        opt = DE(solution_length=6, population_count=10, random_seed=3)
        for _ in range(4):
            solutions = opt.ask()
            opt.tell([sphere(s) for s in solutions])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            self.assertTrue(save_checkpoint(opt, {"iterations": [1, 2]}, path))
            state, info = load_checkpoint(path)

        self.assertIsNotNone(state)
        self.assertEqual(state["_class_name"], "DE")

        restored = apply_checkpoint(DE(solution_length=6, population_count=10), state)
        np.testing.assert_allclose(opt.ask(), restored.ask(), rtol=1e-6)

    def test_failed_save_reports_false(self):
        opt = SNES(solution_length=4, random_seed=0)
        result = save_checkpoint(opt, {}, "/nonexistent-directory/ck.npz")
        self.assertFalse(result)


class TestImports(unittest.TestCase):
    def test_image_module_imports_without_pillow(self):
        """pyevo.utils.image must not require Pillow at import time."""
        import pyevo.utils.image as image_module
        self.assertTrue(hasattr(image_module, "calculate_ssim"))
        # Pillow is only pulled in when a resize actually happens.
        self.assertTrue(hasattr(image_module, "_require_pil"))

    def test_no_shadowed_utils_module(self):
        """pyevo/utils.py used to sit dead beside the pyevo/utils/ package."""
        import pyevo.utils as utils
        self.assertTrue(hasattr(utils, "__path__"), "pyevo.utils must be a package")


if __name__ == "__main__":
    unittest.main()


class TestStatsContract(unittest.TestCase):
    """Every optimizer reports best_fitness under the same key."""

    def test_all_optimizers_report_best_fitness(self):
        for cls in ALL_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                opt = cls(solution_length=5, population_count=8, random_seed=0)
                solutions = opt.ask()
                opt.tell([sphere(s) for s in solutions])
                stats = opt.get_stats()
                self.assertIn("best_fitness", stats)
                self.assertIsNotNone(stats["best_fitness"])
