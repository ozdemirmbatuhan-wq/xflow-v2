from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution

from .airfoil import alpha_for_cl, cl_max, foil_name, polar_point, profile_cd
from .models import AirfoilDesign, AirfoilLike, Fluid, WingGeometry
from .wing import WingResult, geometry_for_target_lift


@dataclass(frozen=True)
class OptimizerEffort:
    airfoil_iterations: int
    wing_iterations: int
    population: int
    polish: bool


EFFORTS: dict[str, OptimizerEffort] = {
    "quick": OptimizerEffort(10, 14, 5, False),
    "balanced": OptimizerEffort(24, 32, 7, True),
    "thorough": OptimizerEffort(55, 70, 10, True),
}


@dataclass
class OptimizationTrace:
    evaluations: int = 0
    best_objective: float = float("inf")
    lock: Lock = field(default_factory=Lock, repr=False)

    def wrap(self, fn: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        def measured(x: np.ndarray) -> float:
            value = float(fn(x))
            with self.lock:
                self.evaluations += 1
                self.best_objective = min(self.best_objective, value)
            return value

        return measured


def optimize_airfoil(
    *,
    fluid: Fluid,
    speed: float,
    reference_chord: float,
    target_cl: float,
    camber_bounds: tuple[float, float],
    camber_position_bounds: tuple[float, float],
    thickness_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    effort: OptimizerEffort,
    seed: int,
    parallel_workers: int = 1,
) -> tuple[AirfoilDesign, dict[str, Any]]:
    reynolds = fluid.reynolds(speed, reference_chord)
    mach = fluid.mach(speed)
    trace = OptimizationTrace()

    def objective(x: np.ndarray) -> float:
        foil = AirfoilDesign(float(x[0]), float(x[1]), float(x[2]))
        limit = cl_max(foil, reynolds)
        alpha = alpha_for_cl(foil, target_cl, reynolds, mach)
        cd = profile_cd(foil, target_cl, reynolds, mach)
        stall_violation = max(0.0, abs(target_cl) / max(0.90 * limit, 1e-6) - 1.0)
        alpha_violation = max(alpha_bounds[0] - alpha, 0.0, alpha - alpha_bounds[1])
        # Gentle regularization avoids fragile extremes when drag differences are negligible.
        location_regularization = 0.00008 * ((foil.camber_position - 0.42) / 0.25) ** 2
        return cd + 0.45 * stall_violation**2 + 0.006 * alpha_violation**2 + location_regularization

    worker_count = max(1, min(int(parallel_workers), os.cpu_count() or 1))
    common = dict(
        bounds=[camber_bounds, camber_position_bounds, thickness_bounds],
        seed=seed,
        maxiter=effort.airfoil_iterations,
        popsize=effort.population,
        polish=effort.polish,
        tol=1e-7,
    )
    measured = trace.wrap(objective)
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="aeropt-2d") as pool:
            result = differential_evolution(
                measured, **common, updating="deferred", workers=pool.map
            )
    else:
        result = differential_evolution(measured, **common, updating="immediate", workers=1)
    raw = AirfoilDesign(float(result.x[0]), float(result.x[1]), float(result.x[2]))
    foil = AirfoilDesign(raw.max_camber, raw.camber_position, raw.thickness, foil_name(raw))
    alpha = alpha_for_cl(foil, target_cl, reynolds, mach)
    point = polar_point(foil, alpha, reynolds, mach)
    metadata = {
        "success": bool(result.success or np.isfinite(result.fun)),
        "message": str(result.message),
        "evaluations": trace.evaluations,
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "reference_chord_m": float(reference_chord),
        "reynolds": float(reynolds),
        "mach": float(mach),
        "target_cl": float(target_cl),
        "design_alpha_deg": float(alpha),
        "design_point": point.to_dict(),
        "cl_max_estimate": float(cl_max(foil, reynolds)),
        "parallel_workers": worker_count,
    }
    return foil, metadata


def optimize_wing(
    *,
    foil: AirfoilLike,
    fluid: Fluid,
    speed: float,
    target_lift: float,
    span_bounds: tuple[float, float],
    root_chord_bounds: tuple[float, float],
    taper_bounds: tuple[float, float],
    sweep_bounds: tuple[float, float],
    twist_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    effort: OptimizerEffort,
    seed: int,
    modes: int,
    polar_mesh: list[dict[str, Any]] | None = None,
    parallel_workers: int = 1,
) -> tuple[WingResult, WingResult, dict[str, Any]]:
    trace = OptimizationTrace()
    target_scale = max(abs(target_lift), 1.0)

    def evaluate_vector(x: np.ndarray, distribution_points: int = 31) -> WingResult:
        geometry = WingGeometry(
            span=float(x[0]),
            root_chord=float(x[1]),
            taper=float(x[2]),
            sweep_deg=float(x[3]),
            tip_twist_deg=float(x[4]),
            alpha_deg=0.0,
        )
        return geometry_for_target_lift(
            foil,
            geometry,
            fluid,
            speed,
            target_lift,
            alpha_bounds[0],
            alpha_bounds[1],
            modes=modes,
            distribution_points=distribution_points,
            polar_mesh=polar_mesh,
        )

    def objective(x: np.ndarray) -> float:
        try:
            wing = evaluate_vector(x)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e6
        lift_error = (wing.lift_n - target_lift) / target_scale
        stall_violation = max(0.0, wing.stall_ratio - 0.92)
        # D/L is a useful scale-free drag objective; constraints dominate infeasible cases.
        return (
            wing.drag_n / target_scale
            + 260.0 * lift_error**2
            + 18.0 * stall_violation**2
        )

    bounds = [span_bounds, root_chord_bounds, taper_bounds, sweep_bounds, twist_bounds]
    worker_count = max(1, min(int(parallel_workers), os.cpu_count() or 1))
    common = dict(
        bounds=bounds,
        seed=seed + 17,
        maxiter=effort.wing_iterations,
        popsize=effort.population,
        polish=effort.polish,
        tol=2e-6,
    )
    measured = trace.wrap(objective)
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="aeropt-3d") as pool:
            result = differential_evolution(
                measured, **common, updating="deferred", workers=pool.map
            )
    else:
        result = differential_evolution(measured, **common, updating="immediate", workers=1)
    optimum = evaluate_vector(result.x, distribution_points=101)

    # A fair rectangular baseline: same span and area, no sweep/twist.
    baseline_chord = float(np.clip(optimum.geometry.area / optimum.geometry.span, *root_chord_bounds))
    baseline_geometry = WingGeometry(
        optimum.geometry.span, baseline_chord, 1.0, 0.0, 0.0, 0.0
    )
    baseline = geometry_for_target_lift(
        foil,
        baseline_geometry,
        fluid,
        speed,
        target_lift,
        alpha_bounds[0],
        alpha_bounds[1],
        modes=modes,
        distribution_points=101,
        polar_mesh=polar_mesh,
    )
    feasible = (
        abs(optimum.lift_n - target_lift) / target_scale <= 0.02
        and optimum.stall_ratio <= 1.0
    )
    metadata = {
        "success": bool(result.success or np.isfinite(result.fun)),
        "feasible": bool(feasible),
        "message": str(result.message),
        "evaluations": trace.evaluations,
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "lift_error_percent": float(100.0 * (optimum.lift_n - target_lift) / target_scale),
        "drag_reduction_vs_rectangular_percent": float(
            100.0 * (baseline.drag_n - optimum.drag_n) / max(baseline.drag_n, 1e-9)
        ),
        "parallel_workers": worker_count,
        "section_polar_source": optimum.section_polar_source,
    }
    return optimum, baseline, metadata
