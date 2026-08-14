from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .airfoil import cst_geometry_is_valid, fit_naca_to_cst, make_cst_airfoil
from .baselines import BaselineProfile
from .checkpoint import OptimizerCheckpointStore
from .convergence import BudgetEscalationController, BudgetEscalationSettings
from .exporters import airfoil_dat
from .flow5 import Flow5CancelledError, Flow5Mesh, Flow5Runner
from .hydro import HydroSettings, analyze_hydro
from .models import AirfoilDesign, AirfoilLike, CSTAirfoilDesign, Fluid, WingGeometry
from .structures import StructuralSettings, analyze_structure
from .surrogate import RBFSurrogateAdvisor, SurrogateSettings


@dataclass
class FoilCandidate:
    foil: CSTAirfoilDesign
    score: float = math.inf
    response: dict[str, Any] | None = None
    conditions: list[dict[str, Any]] | None = None
    error: str | None = None


@dataclass
class WingCandidate:
    geometry: WingGeometry
    score: float = math.inf
    response: dict[str, Any] | None = None
    conditions: list[dict[str, Any]] | None = None
    error: str | None = None
    structural: dict[str, Any] | None = None
    hydro: dict[str, Any] | None = None


def wing_objective_specs(
    *, structural_enabled: bool = False, hydro_enabled: bool = False
) -> list[dict[str, str]]:
    """Return the fixed minimization objectives used by a wing search."""
    specs = [
        {"key": "mean_drag_n", "label": "Ortalama sürükleme", "unit": "N", "direction": "min"},
        {"key": "worst_stall_ratio", "label": "En kötü stall kullanımı", "unit": "-", "direction": "min"},
    ]
    if structural_enabled:
        specs.extend(
            [
                {"key": "estimated_material_mass_kg", "label": "Tahmini malzeme kütlesi", "unit": "kg", "direction": "min"},
                {"key": "maximum_structural_utilization", "label": "Yapısal kullanım", "unit": "-", "direction": "min"},
            ]
        )
    if hydro_enabled:
        specs.append(
            {"key": "cavitation_utilization", "label": "Kavitasyon kullanımı", "unit": "-", "direction": "min"}
        )
    return specs


def _wing_candidate_metrics(candidate: WingCandidate) -> dict[str, float]:
    if not candidate.conditions or not math.isfinite(candidate.score):
        return {}
    structure = candidate.structural or {}
    hydro = candidate.hydro or {}
    structural_utilization = max(
        float(structure.get("stress_utilization", 0.0)),
        float(structure.get("deflection_utilization", 0.0)),
        float(structure.get("twist_utilization", 0.0)),
    )
    return {
        "mean_drag_n": float(np.mean([float(item["drag_n"]) for item in candidate.conditions])),
        "worst_stall_ratio": float(
            np.max([float(item.get("stall_ratio", math.inf)) for item in candidate.conditions])
        ),
        "estimated_material_mass_kg": (
            float(structure["estimated_wing_material_mass_kg"])
            if structure.get("estimated_wing_material_mass_kg") is not None
            else math.inf
        ),
        "maximum_structural_utilization": structural_utilization,
        "cavitation_utilization": (
            float(hydro["cavitation_utilization"])
            if hydro.get("cavitation_utilization") is not None
            else math.inf
        ),
    }


def wing_candidate_objectives(
    candidate: WingCandidate, objective_keys: Sequence[str]
) -> np.ndarray:
    metrics = _wing_candidate_metrics(candidate)
    if not metrics:
        return np.full(len(objective_keys), math.inf, dtype=float)
    return np.asarray([metrics.get(key, math.inf) for key in objective_keys], dtype=float)


def wing_constraint_violation(candidate: WingCandidate) -> float:
    """Aggregate normalized hard-constraint violation for NSGA-II constraint domination."""
    if not candidate.conditions or not math.isfinite(candidate.score) or not candidate.response:
        return math.inf
    violation = 0.0
    for item in candidate.conditions:
        point = item.get("point", {})
        violation += max(0.0, float(item.get("stall_ratio", math.inf)) - 1.0)
        if point.get("out_of_mesh", False):
            violation += 1.0
        if not point.get("viscous_converged", True):
            violation += 1.0
    structure = candidate.structural or {}
    if structure.get("enabled"):
        if not structure.get("performed"):
            violation += 2.0
        else:
            violation += sum(
                max(0.0, float(structure.get(key, 0.0)) - 1.0)
                for key in ("stress_utilization", "deflection_utilization", "twist_utilization")
            )
    hydro = candidate.hydro or {}
    if hydro.get("enabled") and hydro.get("constraint_mode", "hard") == "hard":
        if not hydro.get("performed"):
            violation += 2.0
        else:
            violation += max(0.0, float(hydro.get("cavitation_utilization", 0.0)) - 1.0)
            if hydro.get("free_surface_risk"):
                violation += 1.0
    return float(violation)


def wing_candidate_selection_key(candidate: WingCandidate) -> tuple[float, float]:
    """Rank a deliverable candidate by hard constraints before scalar quality."""
    violation = wing_constraint_violation(candidate)
    score = float(candidate.score)
    return (
        violation if math.isfinite(violation) else math.inf,
        score if math.isfinite(score) else math.inf,
    )


def select_wing_finalist_indices(
    candidates: Sequence[WingCandidate],
    finalist_count: int,
    objective_keys: Sequence[str],
    *,
    optimizer: str,
) -> tuple[list[int], dict[str, Any]]:
    """Keep feasibility-first finalists and append the scalar-only winner."""
    if not candidates:
        raise ValueError("Finalist seçimi için en az bir kanat adayı gerekli")
    requested = max(1, min(int(finalist_count), len(candidates)))
    feasibility_index = min(
        range(len(candidates)),
        key=lambda index: (*wing_candidate_selection_key(candidates[index]), index),
    )
    scalar_index = min(
        range(len(candidates)),
        key=lambda index: (
            float(candidates[index].score)
            if math.isfinite(float(candidates[index].score))
            else math.inf,
            index,
        ),
    )
    selected = [feasibility_index]
    if optimizer == "nsga2":
        fronts, _, crowding = nsga2_rank_and_crowding(candidates, objective_keys)
        for front in fronts:
            for index in sorted(
                front,
                key=lambda item_index: (
                    crowding.get(item_index, 0.0),
                    -float(candidates[item_index].score),
                    -item_index,
                ),
                reverse=True,
            ):
                if index not in selected:
                    selected.append(index)
                if len(selected) >= requested:
                    break
            if len(selected) >= requested:
                break
    else:
        for index in sorted(
            range(len(candidates)),
            key=lambda item_index: (
                *wing_candidate_selection_key(candidates[item_index]),
                item_index,
            ),
        ):
            if index not in selected:
                selected.append(index)
            if len(selected) >= requested:
                break

    scalar_added = scalar_index not in selected
    if scalar_added:
        # This diagnostic candidate does not consume the user's finalist quota.
        selected.append(scalar_index)
    return selected, {
        "requested_finalists": requested,
        "evaluated_finalists": len(selected),
        "feasibility_first_search_index": int(feasibility_index),
        "scalar_only_search_index": int(scalar_index),
        "scalar_only_diagnostic_added": bool(scalar_added),
    }


def constrained_dominates(
    left: WingCandidate,
    right: WingCandidate,
    objective_keys: Sequence[str],
) -> bool:
    """Deb-style constraint domination followed by Pareto domination."""
    left_violation = wing_constraint_violation(left)
    right_violation = wing_constraint_violation(right)
    tolerance = 1e-12
    if left_violation < right_violation - tolerance:
        return True
    if left_violation > right_violation + tolerance:
        return False
    left_values = wing_candidate_objectives(left, objective_keys)
    right_values = wing_candidate_objectives(right, objective_keys)
    return bool(
        np.all(left_values <= right_values + tolerance)
        and np.any(left_values < right_values - tolerance)
    )


def fast_non_dominated_sort(
    candidates: Sequence[WingCandidate],
    objective_keys: Sequence[str],
) -> list[list[int]]:
    """Return NSGA-II fronts as candidate indices."""
    count = len(candidates)
    dominated_sets: list[list[int]] = [[] for _ in range(count)]
    domination_counts = [0] * count
    first_front: list[int] = []
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left_dominates = constrained_dominates(
                candidates[left_index],
                candidates[right_index],
                objective_keys,
            )
            right_dominates = constrained_dominates(
                candidates[right_index],
                candidates[left_index],
                objective_keys,
            )
            if left_dominates:
                dominated_sets[left_index].append(right_index)
                domination_counts[right_index] += 1
            elif right_dominates:
                dominated_sets[right_index].append(left_index)
                domination_counts[left_index] += 1
        if domination_counts[left_index] == 0:
            first_front.append(left_index)
    fronts = [first_front] if first_front else []
    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        next_front: list[int] = []
        for left_index in fronts[cursor]:
            for right_index in dominated_sets[left_index]:
                domination_counts[right_index] -= 1
                if domination_counts[right_index] == 0:
                    next_front.append(right_index)
        if next_front:
            fronts.append(next_front)
        cursor += 1
    return fronts


def crowding_distances(
    candidates: Sequence[WingCandidate],
    front: Sequence[int],
    objective_keys: Sequence[str],
) -> dict[int, float]:
    distances = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: math.inf for index in front}
    for objective_index in range(len(objective_keys)):
        ordered = sorted(
            front,
            key=lambda index: wing_candidate_objectives(
                candidates[index], objective_keys
            )[objective_index],
        )
        values = [
            float(wing_candidate_objectives(candidates[index], objective_keys)[objective_index])
            for index in ordered
        ]
        if not all(math.isfinite(value) for value in values):
            continue
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        scale = values[-1] - values[0]
        if scale <= 1e-15:
            continue
        for position in range(1, len(ordered) - 1):
            if not math.isinf(distances[ordered[position]]):
                distances[ordered[position]] += (
                    values[position + 1] - values[position - 1]
                ) / scale
    return distances


def nsga2_rank_and_crowding(
    candidates: Sequence[WingCandidate],
    objective_keys: Sequence[str],
) -> tuple[list[list[int]], dict[int, int], dict[int, float]]:
    fronts = fast_non_dominated_sort(candidates, objective_keys)
    ranks: dict[int, int] = {}
    crowding: dict[int, float] = {}
    for rank, front in enumerate(fronts):
        ranks.update({index: rank for index in front})
        crowding.update(crowding_distances(candidates, front, objective_keys))
    return fronts, ranks, crowding


def nsga2_environmental_selection(
    candidates: Sequence[WingCandidate],
    population_size: int,
    objective_keys: Sequence[str],
) -> list[WingCandidate]:
    fronts, _, crowding = nsga2_rank_and_crowding(candidates, objective_keys)
    selected: list[WingCandidate] = []
    for front in fronts:
        remaining = population_size - len(selected)
        if remaining <= 0:
            break
        ordered = sorted(
            front,
            key=lambda index: (crowding.get(index, 0.0), -candidates[index].score),
            reverse=True,
        )
        selected.extend(candidates[index] for index in ordered[:remaining])
    return selected


def _wing_tradeoff_summary(
    candidate: WingCandidate,
    *,
    identifier: str,
    selected: bool,
    fidelity: str,
) -> dict[str, Any] | None:
    if not candidate.conditions or not math.isfinite(candidate.score):
        return None
    drag_values = [float(item["drag_n"]) for item in candidate.conditions]
    ld_values = [float(item["ld"]) for item in candidate.conditions]
    stall_values = [float(item.get("stall_ratio", math.inf)) for item in candidate.conditions]
    structure = candidate.structural or {}
    hydro = candidate.hydro or {}
    structural_utilization = max(
        (
            float(structure.get("stress_utilization", 0.0)),
            float(structure.get("deflection_utilization", 0.0)),
            float(structure.get("twist_utilization", 0.0)),
        )
    ) if structure.get("enabled") and structure.get("performed") else None
    cavitation_utilization = (
        float(hydro.get("cavitation_utilization", 0.0))
        if hydro.get("enabled") and hydro.get("performed")
        else None
    )
    feasible = all(
        item.get("stall_ratio", math.inf) <= 1.0
        and not item.get("point", {}).get("out_of_mesh", False)
        and item.get("point", {}).get("viscous_converged", True)
        for item in candidate.conditions
    ) and bool(structure.get("passed", True)) and bool(
        hydro.get("constraint_passed", hydro.get("passed", True))
    )
    return {
        "id": identifier,
        "selected": bool(selected),
        "fidelity": fidelity,
        "feasible": bool(feasible),
        "score": float(candidate.score),
        "mean_drag_n": float(np.mean(drag_values)),
        "maximum_drag_n": float(np.max(drag_values)),
        "mean_ld": float(np.mean(ld_values)),
        "minimum_ld": float(np.min(ld_values)),
        "worst_stall_ratio": float(np.max(stall_values)),
        "estimated_material_mass_kg": (
            float(structure["estimated_wing_material_mass_kg"])
            if structure.get("estimated_wing_material_mass_kg") is not None
            else None
        ),
        "maximum_structural_utilization": structural_utilization,
        "cavitation_utilization": cavitation_utilization,
        "cavitation_constraint_active": bool(
            hydro.get("enabled") and hydro.get("constraint_mode", "hard") == "hard"
        ),
        "geometry": candidate.geometry.to_dict(),
    }


def build_pareto_analysis(
    search_candidates: list[WingCandidate],
    selected_candidate: WingCandidate,
    *,
    optimizer: str = "post_hoc",
) -> dict[str, Any]:
    """Build a non-dominated trade-space from real solver-evaluated candidates."""
    summaries: list[dict[str, Any]] = []
    for index, candidate in enumerate(search_candidates):
        summary = _wing_tradeoff_summary(
            candidate,
            identifier=f"search-{index + 1}",
            selected=False,
            fidelity="search",
        )
        if summary is not None:
            summaries.append(summary)
    selected_summary = _wing_tradeoff_summary(
        selected_candidate,
        identifier="selected-final",
        selected=True,
        fidelity="final-fine-mesh",
    )
    if selected_summary is not None:
        summaries.append(selected_summary)

    objective_specs = wing_objective_specs(
        structural_enabled=any(
            item.get("estimated_material_mass_kg") is not None for item in summaries
        ),
        hydro_enabled=any(
            item.get("cavitation_constraint_active") for item in summaries
        ),
    )

    objective_keys = [item["key"] for item in objective_specs]

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if bool(left.get("feasible")) != bool(right.get("feasible")):
            return bool(left.get("feasible"))
        left_values = [float(left.get(key, math.inf) if left.get(key) is not None else math.inf) for key in objective_keys]
        right_values = [float(right.get(key, math.inf) if right.get(key) is not None else math.inf) for key in objective_keys]
        return all(a <= b + 1e-12 for a, b in zip(left_values, right_values)) and any(
            a < b - 1e-12 for a, b in zip(left_values, right_values)
        )

    for candidate in summaries:
        candidate["on_pareto_front"] = not any(
            other is not candidate and dominates(other, candidate) for other in summaries
        )
    frontier = [item for item in summaries if item["on_pareto_front"]]
    frontier.sort(key=lambda item: (item["mean_drag_n"], item["id"]))
    if len(frontier) > 48:
        indices = np.linspace(0, len(frontier) - 1, 48, dtype=int)
        frontier = [frontier[int(index)] for index in indices]
    retained_ids = {item["id"] for item in frontier}
    ranked = sorted(summaries, key=lambda item: item["score"])
    retained = list(frontier)
    for item in ranked:
        if len(retained) >= 80:
            break
        if item["id"] not in retained_ids:
            retained.append(item)
            retained_ids.add(item["id"])
    selected_on_front = bool(
        selected_summary and selected_summary.get("on_pareto_front", False)
    )
    return {
        "enabled": True,
        "definition": "non-dominated real flow5 candidates; all listed objectives minimized",
        "optimizer": optimizer,
        "provenance": (
            "optimizer_generated"
            if optimizer == "nsga2"
            else "post_hoc_from_scalar_search"
        ),
        "optimizer_generated_frontier": optimizer == "nsga2",
        "objective_specs": objective_specs,
        "candidate_count": len(summaries),
        "frontier_count": len(frontier),
        "selected_on_front": selected_on_front,
        "selected_id": "selected-final",
        "frontier": frontier,
        "candidates": retained,
        "fidelity_note": "Search points use the search method/mesh; selected-final uses the final method and output mesh.",
    }


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(
            {
                "stage": stage,
                "current": int(current),
                "total": int(max(total, 1)),
                "fraction": float(np.clip(current / max(total, 1), 0.0, 1.0)),
                "message": message,
            }
        )


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise Flow5CancelledError("Optimizasyon kullanıcı tarafından durduruldu")


def interpolate_at_cl(points: list[dict[str, Any]], target_cl: float) -> dict[str, Any] | None:
    """Interpolate between adjacent-alpha solver points; never extrapolate."""
    usable = sorted(
        (point for point in points if math.isfinite(float(point.get("cl", math.nan)))),
        key=lambda point: float(point["alpha_deg"]),
    )
    if len(usable) < 2:
        return None
    candidates: list[dict[str, Any]] = []
    for left, right in zip(usable, usable[1:]):
        cl0, cl1 = float(left["cl"]), float(right["cl"])
        if (target_cl - cl0) * (target_cl - cl1) > 0.0 or abs(cl1 - cl0) < 1e-12:
            continue
        fraction = (target_cl - cl0) / (cl1 - cl0)
        result: dict[str, Any] = {"cl": float(target_cl)}
        numeric_keys = set(left).intersection(right)
        for key in numeric_keys:
            if key in {"cl", "out_of_mesh", "viscous_converged"}:
                continue
            try:
                a, b = float(left[key]), float(right[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(a) and math.isfinite(b):
                result[key] = a + fraction * (b - a)
        result["out_of_mesh"] = bool(left.get("out_of_mesh") or right.get("out_of_mesh"))
        result["viscous_converged"] = bool(
            left.get("viscous_converged", True)
            and right.get("viscous_converged", True)
        )
        left_dist, right_dist = left.get("distribution"), right.get("distribution")
        if isinstance(left_dist, list) and isinstance(right_dist, list) and len(left_dist) == len(right_dist):
            distribution: list[dict[str, float]] = []
            for station0, station1 in zip(left_dist, right_dist):
                if not isinstance(station0, dict) or not isinstance(station1, dict):
                    continue
                station: dict[str, float] = {}
                for key in (
                    "y_m",
                    "chord_m",
                    "local_cl",
                    "lift_n_per_m",
                    "reynolds",
                    "induced_angle_deg",
                    "cdi",
                    "cdv",
                    "bending_moment_nm",
                    "twist_deg",
                ):
                    if key in station0 and key in station1:
                        station[key] = float(station0[key]) + fraction * (
                            float(station1[key]) - float(station0[key])
                        )
                station["converged"] = bool(
                    station0.get("converged", True) and station1.get("converged", True)
                )
                if all(key in station for key in ("y_m", "chord_m", "local_cl", "lift_n_per_m")):
                    distribution.append(station)
            result["distribution"] = distribution
        if "cd" in result and result["cd"] > 0.0:
            result["ld"] = target_cl / result["cd"]
            candidates.append(result)
    return min(candidates, key=lambda point: float(point["cd"])) if candidates else None


def _flow5_foil_name(foil: CSTAirfoilDesign, suffix: str = "") -> str:
    tail = f"-{suffix}" if suffix else ""
    return (
        f"Flow5-{foil.family}-m{100.0 * foil.max_camber:.1f}"
        f"-p{100.0 * foil.camber_position:.0f}-t{100.0 * foil.thickness:.1f}{tail}"
    )


def _rename_foil(foil: CSTAirfoilDesign, name: str) -> CSTAirfoilDesign:
    return CSTAirfoilDesign(
        foil.upper_weights,
        foil.lower_weights,
        foil.max_camber,
        foil.camber_position,
        foil.thickness,
        name,
        foil.trailing_edge_gap,
    )


def _foil_score(
    response: dict[str, Any],
    target_cls: list[float],
    alpha_bounds: tuple[float, float],
) -> tuple[float, list[dict[str, Any]]]:
    polars = response["polars"]
    if len(polars) != len(target_cls):
        return math.inf, []
    drag_to_lift: list[float] = []
    conditions: list[dict[str, Any]] = []
    penalty = 0.0
    for polar, target_cl in zip(polars, target_cls):
        point = interpolate_at_cl(polar["points"], target_cl)
        if point is None or point.get("cd", 0.0) <= 0.0:
            return math.inf, []
        alpha = float(point["alpha_deg"])
        alpha_violation = max(alpha_bounds[0] - alpha, 0.0, alpha - alpha_bounds[1])
        cl_peak = max(float(row["cl"]) for row in polar["points"])
        stall_ratio = target_cl / max(cl_peak, 1e-8)
        penalty += 0.012 * alpha_violation**2 + 0.35 * max(0.0, stall_ratio - 0.90) ** 2
        drag_to_lift.append(float(point["cd"]) / max(abs(target_cl), 0.08))
        conditions.append(
            {
                "speed_m_s": float(polar["speed_m_s"]),
                "reynolds": float(polar["reynolds"]),
                "mach": float(polar["mach"]),
                "target_cl": float(target_cl),
                "cl_max_converged": cl_peak,
                "stall_ratio": float(stall_ratio),
                "point": point,
            }
        )
    robust_drag = float(np.mean(drag_to_lift) + 0.40 * np.max(drag_to_lift))
    return robust_drag + penalty / max(len(conditions), 1), conditions


def evaluate_fixed_airfoil_with_flow5(
    *,
    runner: Flow5Runner,
    baseline_profile: BaselineProfile,
    fluid: Fluid,
    speeds_m_s: list[float],
    target_cls: list[float],
    reference_chord_m: float,
    alpha_bounds: tuple[float, float],
    total_threads: int,
    coordinate_points: int,
    alpha_step_final_deg: float,
    ncrit: float,
    xtr_top: float,
    xtr_bottom: float,
) -> tuple[CSTAirfoilDesign, dict[str, Any], dict[str, Any], str]:
    """Analyze an imported DAT once without changing its geometry."""
    foil = baseline_profile.foil
    threads = max(1, min(int(total_threads), len(speeds_m_s)))
    response = runner.analyze_foil(
        foil=foil,
        fluid=fluid,
        speeds_m_s=speeds_m_s,
        reference_chord_m=reference_chord_m,
        alpha_min_deg=alpha_bounds[0],
        alpha_max_deg=alpha_bounds[1],
        alpha_step_deg=alpha_step_final_deg,
        max_threads=threads,
        coordinate_points=coordinate_points,
        foil_dat_text=baseline_profile.solver_dat_text,
        ncrit=ncrit,
        xtr_top=xtr_top,
        xtr_bottom=xtr_bottom,
    )
    score, conditions = _foil_score(response, target_cls, alpha_bounds)
    if not math.isfinite(score) or not conditions:
        raise RuntimeError(
            "Seçilen profil flow5/XFoil ile hedef CL aralığında doğrulanamadı"
        )
    budget_report = {
        "enabled": False,
        "converged": None,
        "status": "not_applicable",
        "base_budget": 0,
        "maximum_budget": 0,
        "evaluations_completed": 1,
        "milestones": [],
        "checkpoints": [],
    }
    metadata = {
        "success": True,
        "source": "flow5 embedded XFoil only",
        "optimizer": "skipped_fixed_airfoil",
        "objective": float(score),
        "candidate_budget": 0,
        "maximum_candidate_budget": 0,
        "candidates_evaluated": 1,
        "budget_convergence": budget_report,
        "valid_candidates": 1,
        "outer_parallel_runners": 1,
        "threads_per_runner": threads,
        "total_threads_requested": int(total_threads),
        "detected_logical_cores": os.cpu_count() or 1,
        "surrogate": {"enabled": False, "reason": "sabit profil"},
        "checkpoint": {"enabled": False, "resumed": False},
        "reference_chord_m": float(reference_chord_m),
        "cst_order": int(baseline_profile.cst_order),
        "solver_coordinate_points": int(coordinate_points),
        "conditions": conditions,
        "baseline": {
            **baseline_profile.to_dict(),
            "within_design_envelope": True,
            "solver_geometry": "source DAT cosine-resampled to 100 points",
            "search_converged": True,
            "search_score": float(score),
            "final_score": float(score),
            "selected": True,
            "error": None,
        },
        "selection": {
            "mode": "fixed_import",
            "selected_name": foil.name,
            "selected_family": foil.family,
            "selected_baseline": True,
            "minimum_improvement_percent": 0.0,
            "raw_best_improvement_vs_baseline_percent": 0.0,
            "selected_improvement_vs_baseline_percent": 0.0,
            "baseline_retained_by_threshold": False,
            "finalists_evaluated": 1,
        },
        "finalists": [
            {
                "name": foil.name,
                "score": float(score),
                "is_baseline": True,
                "within_design_envelope": True,
                "selected": True,
                "error": None,
            }
        ],
        "top_candidates": [
            {
                "rank": 1,
                "name": foil.name,
                "family": foil.family,
                "is_baseline": True,
                "score": float(score),
                "thickness": float(foil.thickness),
                "max_camber": float(foil.max_camber),
                "error": None,
            }
        ],
        "solver": response.get("solver", {}),
    }
    return foil, response, metadata, baseline_profile.solver_dat_text


def optimize_airfoil_with_flow5(
    *,
    runner: Flow5Runner,
    baseline_profile: BaselineProfile,
    fluid: Fluid,
    speeds_m_s: list[float],
    target_cls: list[float],
    reference_chord_m: float,
    camber_bounds: tuple[float, float],
    camber_position_bounds: tuple[float, float],
    thickness_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    candidate_budget: int,
    seed: int,
    total_threads: int,
    cst_order: int,
    coordinate_points: int,
    minimum_improvement_percent: float,
    alpha_step_search_deg: float,
    alpha_step_final_deg: float,
    ncrit: float,
    xtr_top: float,
    xtr_bottom: float,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    optimizer: str = "differential_evolution",
    surrogate_settings: SurrogateSettings = SurrogateSettings(),
    budget_escalation_settings: BudgetEscalationSettings = BudgetEscalationSettings(),
    checkpoint_store: OptimizerCheckpointStore | None = None,
    checkpoint_key: str = "",
) -> tuple[CSTAirfoilDesign, dict[str, Any], dict[str, Any], str]:
    """Optimize around a real DAT baseline using flow5 embedded XFoil only."""
    budget = max(8, int(candidate_budget))
    budget_controller = BudgetEscalationController(
        budget, budget_escalation_settings, hard_limit=4096
    )
    rng = np.random.default_rng(seed + 5705)
    threads_per_runner = max(1, min(int(total_threads), len(speeds_m_s)))
    detected = os.cpu_count() or 1
    outer_workers = max(1, min(budget, detected, int(total_threads) // threads_per_runner))
    family_prefix = f"Flow5-CST{cst_order}"

    def random_naca_seed(index: int) -> CSTAirfoilDesign:
        naca = AirfoilDesign(
            float(rng.uniform(*camber_bounds)),
            float(rng.uniform(*camber_position_bounds)),
            float(rng.uniform(*thickness_bounds)),
            f"geometry-seed-{index}",
        )
        foil = fit_naca_to_cst(naca, order=cst_order, name=f"{family_prefix}-c{index:04d}")
        return _rename_foil(foil, f"{family_prefix}-c{index:04d}")

    midpoint = AirfoilDesign(
        float(np.mean(camber_bounds)),
        float(np.mean(camber_position_bounds)),
        float(np.mean(thickness_bounds)),
        "geometry-midpoint",
    )
    seed_foil = fit_naca_to_cst(midpoint, order=cst_order, name=f"{family_prefix}-c0000")
    seed_foil = _rename_foil(seed_foil, f"{family_prefix}-c0000")
    baseline_foil = baseline_profile.foil
    baseline_key = (baseline_foil.upper_weights, baseline_foil.lower_weights)
    queue: list[CSTAirfoilDesign] = [seed_foil]
    candidates: list[FoilCandidate] = []
    candidate_index = 1

    def valid(foil: CSTAirfoilDesign) -> bool:
        return cst_geometry_is_valid(
            foil,
            camber_bounds=camber_bounds,
            camber_position_bounds=camber_position_bounds,
            thickness_bounds=thickness_bounds,
        )

    def mutate(parent: CSTAirfoilDesign, index: int, sigma: float) -> CSTAirfoilDesign | None:
        vector = np.asarray((*parent.upper_weights, *parent.lower_weights), dtype=float)
        for _ in range(24):
            proposal = np.clip(vector + rng.normal(0.0, sigma, size=vector.size), -0.58, 0.58)
            half = proposal.size // 2
            foil = make_cst_airfoil(
                proposal[:half], proposal[half:], name=f"{family_prefix}-c{index:04d}"
            )
            if valid(foil):
                return foil
        return None

    def analyze(foil: CSTAirfoilDesign, alpha_step: float) -> FoilCandidate:
        _check_cancelled(cancel_event)
        try:
            uses_source_baseline = (
                foil.upper_weights,
                foil.lower_weights,
            ) == baseline_key
            response = runner.analyze_foil(
                foil=foil,
                fluid=fluid,
                speeds_m_s=speeds_m_s,
                reference_chord_m=reference_chord_m,
                alpha_min_deg=alpha_bounds[0],
                alpha_max_deg=alpha_bounds[1],
                alpha_step_deg=alpha_step,
                max_threads=threads_per_runner,
                coordinate_points=coordinate_points,
                foil_dat_text=(
                    baseline_profile.solver_dat_text if uses_source_baseline else None
                ),
                ncrit=ncrit,
                xtr_top=xtr_top,
                xtr_bottom=xtr_bottom,
            )
            score, conditions = _foil_score(response, target_cls, alpha_bounds)
            return FoilCandidate(foil, score, response, conditions)
        except Flow5CancelledError:
            raise
        except Exception as exc:
            return FoilCandidate(foil, error=str(exc)[-300:])

    def evaluate_search(foil: CSTAirfoilDesign) -> FoilCandidate:
        return analyze(foil, alpha_step_search_deg)

    baseline_within_envelope = valid(baseline_foil)
    baseline_search = evaluate_search(baseline_foil)
    if baseline_within_envelope:
        candidates.append(baseline_search)
    _emit_progress(
        progress_callback,
        "foil_search",
        len(candidates),
        budget_controller.progress_total,
        "E818 başlangıç profili flow5/XFoil ile değerlendirildi",
    )

    optimizer_key = optimizer.strip().lower()
    if optimizer_key not in {"differential_evolution", "adaptive_elite"}:
        raise ValueError("Profil optimizeri differential_evolution veya adaptive_elite olmalı")

    def evaluate_batch(batch: list[CSTAirfoilDesign]) -> list[FoilCandidate]:
        if len(batch) > 1:
            with ThreadPoolExecutor(
                max_workers=min(outer_workers, len(batch)), thread_name_prefix="flow5-foil"
            ) as pool:
                return list(pool.map(evaluate_search, batch))
        return [evaluate_search(batch[0])] if batch else []

    search_evaluations = len(candidates)
    score_history = [float(item.score) for item in candidates]

    def observe_foil_budget() -> dict[str, Any] | None:
        checkpoint = budget_controller.observe(scores=score_history)
        if checkpoint is not None:
            decision = checkpoint["decision"]
            if decision == "escalated":
                message = (
                    f"Profil yakınsaması sürüyor; bütçe "
                    f"{checkpoint['next_budget']} adaya yükseltildi"
                )
            elif decision == "converged":
                message = (
                    f"Profil bütçesi yeterli: değişim "
                    f"%{checkpoint['controlling_change_percent']:.3f}"
                )
            elif decision == "maximum_budget_reached":
                message = "Profil azami bütçeye ulaştı; sonuç bütçe-sınırlı işaretlendi"
            else:
                message = "Sabit profil aday bütçesi tamamlandı"
            _emit_progress(
                progress_callback,
                "foil_budget",
                search_evaluations,
                budget_controller.progress_total,
                message,
            )
        return checkpoint
    surrogate_report: dict[str, Any] = {
        "enabled": bool(surrogate_settings.enabled),
        "trained": False,
        "reason": "yalnız differential_evolution ile etkin",
    }
    checkpoint_report: dict[str, Any] = {
        "enabled": bool(checkpoint_store and checkpoint_store.enabled and checkpoint_key),
        "resumed": False,
        "evaluations_restored": 0,
        "generation_restored": 0,
    }
    completed_generation = 0
    if optimizer_key == "differential_evolution":
        dimension = 2 * (cst_order + 1)
        weight_bounds = np.tile(np.asarray([[-0.58, 0.58]], dtype=float), (dimension, 1))
        advisor = RBFSurrogateAdvisor(weight_bounds, surrogate_settings)
        population_size = min(budget, max(8, dimension + 2))
        population: list[FoilCandidate] = []
        generation = 0
        restored = (
            checkpoint_store.load(checkpoint_key)
            if checkpoint_store is not None and checkpoint_key
            else None
        )
        if restored and restored.get("optimizer") == "differential_evolution":
            try:
                rng.bit_generator.state = restored["rng_state"]
                candidate_index = int(restored.get("candidate_index", 1))
                generation = int(restored.get("generation", 0))
                for index, raw_vector in enumerate(restored["population_vectors"]):
                    vector = np.asarray(raw_vector, dtype=float)
                    half = vector.size // 2
                    restored_foil = make_cst_airfoil(
                        vector[:half], vector[half:], name=f"{family_prefix}-resume{index:04d}"
                    )
                    population.append(evaluate_search(restored_foil))
                candidates = list(population)
                restored_count = max(
                    int(restored.get("evaluations_completed", len(population))),
                    len(population),
                )
                restored_scores = [
                    float(value) for value in restored.get("score_history", [])
                ]
                if len(restored_scores) < restored_count:
                    fallback = min(
                        (item.score for item in population), default=math.inf
                    )
                    restored_scores.extend(
                        [float(fallback)] * (restored_count - len(restored_scores))
                    )
                score_history = restored_scores[:restored_count]
                search_evaluations = len(score_history)
                budget_controller.restore(restored.get("budget_controller"))
                surrogate_state = restored.get("surrogate", {})
                advisor.restore(
                    list(surrogate_state.get("vectors", [])),
                    list(surrogate_state.get("scores", [])),
                )
                checkpoint_report.update(
                    resumed=True,
                    evaluations_restored=search_evaluations,
                    generation_restored=generation,
                )
                _emit_progress(
                    progress_callback,
                    "foil_search",
                    search_evaluations,
                    budget_controller.progress_total,
                    f"DE profil checkpoint'i yüklendi: nesil {generation}, {search_evaluations}/{budget_controller.progress_total}",
                )
            except (KeyError, TypeError, ValueError):
                population = []
                candidates = [baseline_search] if baseline_within_envelope else []
                search_evaluations = len(candidates)
                score_history = [float(item.score) for item in candidates]

        def save_de_checkpoint() -> None:
            if checkpoint_store is None or not checkpoint_key or not population:
                return
            checkpoint_store.save(
                checkpoint_key,
                {
                    "optimizer": "differential_evolution",
                    "generation": generation,
                    "evaluations_completed": search_evaluations,
                    "candidate_index": candidate_index,
                    "population_vectors": [
                        [*item.foil.upper_weights, *item.foil.lower_weights]
                        for item in population
                    ],
                    "rng_state": rng.bit_generator.state,
                    "surrogate": advisor.state(),
                    "score_history": score_history,
                    "budget_controller": budget_controller.state(),
                },
            )

        if not population:
            population = list(candidates)
            initial_batch: list[CSTAirfoilDesign] = []
            if len(population) < population_size:
                initial_batch.append(queue.pop(0))
            while len(population) + len(initial_batch) < population_size:
                parent = baseline_foil if rng.random() < 0.65 else seed_foil
                foil = mutate(parent, candidate_index, float(rng.uniform(0.012, 0.045)))
                if foil is None:
                    foil = random_naca_seed(candidate_index)
                candidate_index += 1
                if valid(foil):
                    initial_batch.append(foil)
            for start in range(0, len(initial_batch), outer_workers):
                evaluated = evaluate_batch(initial_batch[start : start + outer_workers])
                population.extend(evaluated)
                candidates.extend(evaluated)
                search_evaluations += len(evaluated)
                score_history.extend(float(item.score) for item in evaluated)
                for item in evaluated:
                    advisor.record((*item.foil.upper_weights, *item.foil.lower_weights), item.score)
                _emit_progress(
                    progress_callback,
                    "foil_search",
                    search_evaluations,
                    budget_controller.progress_total,
                    f"DE profil popülasyonu çözüldü: {search_evaluations}/{budget_controller.progress_total}",
                )
            for item in population:
                advisor.record((*item.foil.upper_weights, *item.foil.lower_weights), item.score)
            save_de_checkpoint()
            observe_foil_budget()
            save_de_checkpoint()

        while not budget_controller.stopped:
            _check_cancelled(cancel_event)
            if not budget_controller.should_evaluate(search_evaluations):
                observe_foil_budget()
                save_de_checkpoint()
                continue
            generation += 1
            previous_best = min((item.score for item in population), default=math.inf)
            trial_foils: list[CSTAirfoilDesign] = []
            trial_vectors: list[np.ndarray] = []
            target_indices: list[int] = []
            remaining = budget_controller.current_target - search_evaluations
            for target_index in range(min(len(population), remaining)):
                target = population[target_index].foil
                target_vector = np.asarray(
                    (*target.upper_weights, *target.lower_weights), dtype=float
                )
                proposal_foils: list[CSTAirfoilDesign] = []
                proposal_vectors: list[np.ndarray] = []
                proposal_count = (
                    surrogate_settings.proposals_per_real_evaluation if advisor.ready else 1
                )
                for _proposal_index in range(proposal_count):
                    proposal_foil: CSTAirfoilDesign | None = None
                    proposal: np.ndarray | None = None
                    choices = [index for index in range(len(population)) if index != target_index]
                    for _ in range(30):
                        a_index, b_index, c_index = rng.choice(choices, size=3, replace=False)
                        vectors = []
                        for source_index in (a_index, b_index, c_index):
                            source = population[int(source_index)].foil
                            vectors.append(
                                np.asarray(
                                    (*source.upper_weights, *source.lower_weights), dtype=float
                                )
                            )
                        mutation = vectors[0] + float(rng.uniform(0.50, 0.92)) * (
                            vectors[1] - vectors[2]
                        )
                        crossover = rng.random(dimension) < 0.78
                        crossover[int(rng.integers(0, dimension))] = True
                        proposal = np.clip(
                            np.where(crossover, mutation, target_vector), -0.58, 0.58
                        )
                        half = proposal.size // 2
                        proposal_foil = make_cst_airfoil(
                            proposal[:half],
                            proposal[half:],
                            name=f"{family_prefix}-de{candidate_index:04d}",
                        )
                        candidate_index += 1
                        if valid(proposal_foil):
                            break
                        proposal_foil = None
                    if proposal_foil is None:
                        proposal_foil = random_naca_seed(candidate_index)
                        candidate_index += 1
                        proposal = np.asarray(
                            (*proposal_foil.upper_weights, *proposal_foil.lower_weights), dtype=float
                        )
                    proposal_foils.append(proposal_foil)
                    proposal_vectors.append(np.asarray(proposal, dtype=float))
                chosen = advisor.choose(proposal_vectors)
                trial_foils.append(proposal_foils[chosen])
                trial_vectors.append(proposal_vectors[chosen])
                target_indices.append(target_index)

            for start in range(0, len(trial_foils), outer_workers):
                batch = trial_foils[start : start + outer_workers]
                evaluated = evaluate_batch(batch)
                candidates.extend(evaluated)
                search_evaluations += len(evaluated)
                score_history.extend(float(item.score) for item in evaluated)
                for local_index, trial in enumerate(evaluated):
                    population_index = target_indices[start + local_index]
                    advisor.record(trial_vectors[start + local_index], trial.score)
                    if trial.score <= population[population_index].score:
                        population[population_index] = trial
                _emit_progress(
                    progress_callback,
                    "foil_search",
                    search_evaluations,
                    budget_controller.progress_total,
                    f"DE profil nesli {generation}: {search_evaluations}/{budget_controller.progress_total}",
                )
            observe_foil_budget()
            save_de_checkpoint()
            current_best = min((item.score for item in population), default=math.inf)
            if not budget_escalation_settings.enabled and advisor.may_stop_early(
                evaluations=search_evaluations,
                budget=budget,
                previous_best=previous_best,
                current_best=current_best,
            ):
                budget_controller.mark_converged("validated_surrogate_early_stop")
                break
        surrogate_report = advisor.report(
            real_evaluations=search_evaluations,
            budget=budget_controller.maximum_budget,
        )
        completed_generation = generation
    else:
        while not budget_controller.stopped:
            _check_cancelled(cancel_event)
            if not budget_controller.should_evaluate(search_evaluations):
                observe_foil_budget()
                continue
            remaining = budget_controller.current_target - search_evaluations
            batch_size = min(outer_workers, remaining)
            finite = sorted(
                (item for item in candidates if math.isfinite(item.score)),
                key=lambda item: item.score,
            )
            elites = finite[: min(8, len(finite))]
            batch: list[CSTAirfoilDesign] = []
            progress = search_evaluations / max(budget_controller.maximum_budget - 1, 1)
            sigma = 0.022 * (1.0 - progress) + 0.0035 * progress
            while len(batch) < batch_size:
                if queue:
                    foil = queue.pop(0)
                elif elites and rng.random() < 0.82:
                    foil = mutate(
                        elites[int(rng.integers(0, len(elites)))].foil,
                        candidate_index,
                        sigma,
                    )
                    if foil is None:
                        foil = random_naca_seed(candidate_index)
                elif rng.random() < 0.62:
                    parent = baseline_foil if rng.random() < 0.72 else seed_foil
                    foil = mutate(parent, candidate_index, sigma)
                    if foil is None:
                        foil = random_naca_seed(candidate_index)
                else:
                    foil = random_naca_seed(candidate_index)
                candidate_index += 1
                if valid(foil):
                    batch.append(foil)
            evaluated = evaluate_batch(batch)
            candidates.extend(evaluated)
            score_history.extend(float(item.score) for item in evaluated)
            search_evaluations += len(evaluated)
            _emit_progress(
                progress_callback,
                "foil_search",
                search_evaluations,
                budget_controller.progress_total,
                f"Profil adayları çözüldü: {search_evaluations}/{budget_controller.progress_total}",
            )
            observe_foil_budget()

    valid_candidates = [item for item in candidates if math.isfinite(item.score) and item.response]
    if not valid_candidates:
        errors = "; ".join(item.error for item in candidates if item.error)[:900]
        raise RuntimeError(f"flow5/XFoil profil aramasında geçerli aday bulunamadı: {errors}")

    ranked_search = sorted(valid_candidates, key=lambda item: item.score)
    fine_pool = ranked_search[: min(5, len(ranked_search))]
    if math.isfinite(baseline_search.score) and baseline_search.response:
        if not any(
            (item.foil.upper_weights, item.foil.lower_weights) == baseline_key
            for item in fine_pool
        ):
            fine_pool.append(baseline_search)

    fine_results: list[tuple[FoilCandidate, bool, float, bool]] = []
    for index, search_item in enumerate(fine_pool):
        _check_cancelled(cancel_event)
        is_baseline = (
            search_item.foil.upper_weights,
            search_item.foil.lower_weights,
        ) == baseline_key
        fine_name = (
            baseline_profile.foil.name
            if is_baseline
            else _flow5_foil_name(search_item.foil, f"f{index + 1}")
        )
        fine = analyze(_rename_foil(search_item.foil, fine_name), alpha_step_final_deg)
        fine_results.append((fine, is_baseline, float(search_item.score), valid(search_item.foil)))
        _emit_progress(
            progress_callback,
            "foil_final",
            index + 1,
            len(fine_pool),
            f"İnce profil doğrulaması: {index + 1}/{len(fine_pool)}",
        )

    valid_fine = [
        item
        for item in fine_results
        if math.isfinite(item[0].score) and item[0].response and (item[3] or not item[1])
    ]
    selectable_fine = [item for item in valid_fine if not item[1] or baseline_within_envelope]
    if not selectable_fine:
        errors = "; ".join(item[0].error for item in fine_results if item[0].error)[:900]
        raise RuntimeError(f"flow5 ince profil doğrulamasında geçerli finalist yok: {errors}")

    winner = min(selectable_fine, key=lambda item: item[0].score)
    baseline_final = next(
        (
            item
            for item in fine_results
            if item[1] and math.isfinite(item[0].score) and item[0].response
        ),
        None,
    )
    raw_improvement_percent: float | None = None
    threshold_applied = False
    if baseline_final is not None and baseline_within_envelope:
        raw_improvement_percent = float(
            100.0
            * (baseline_final[0].score - winner[0].score)
            / max(abs(baseline_final[0].score), 1e-12)
        )
        if not winner[1] and raw_improvement_percent < minimum_improvement_percent:
            winner = baseline_final
            threshold_applied = True

    winner_candidate, winner_is_baseline, _, _ = winner
    final_foil = (
        _rename_foil(winner_candidate.foil, baseline_profile.foil.name)
        if winner_is_baseline
        else _rename_foil(winner_candidate.foil, _flow5_foil_name(winner_candidate.foil, "opt"))
    )
    final_response = winner_candidate.response
    final_score = float(winner_candidate.score)
    final_conditions = winner_candidate.conditions or []
    if final_response is None or not final_conditions:
        raise RuntimeError("flow5 ince profil doğrulamasında hedef CL aralığı yakınsamadı")

    ranked = sorted(candidates, key=lambda item: item.score)
    selected_key = (final_foil.upper_weights, final_foil.lower_weights)
    selected_dat_text = (
        baseline_profile.solver_dat_text
        if winner_is_baseline
        else airfoil_dat(final_foil, total_points=coordinate_points)
    )
    baseline_score = float(baseline_final[0].score) if baseline_final else None
    selected_improvement = (
        0.0
        if winner_is_baseline and baseline_score is not None
        else (
            float(100.0 * (baseline_score - final_score) / max(abs(baseline_score), 1e-12))
            if baseline_score is not None
            else None
        )
    )
    if checkpoint_store is not None and checkpoint_key:
        checkpoint_store.clear(checkpoint_key)
        checkpoint_report["generations_completed"] = completed_generation
        checkpoint_report["store"] = checkpoint_store.stats()
    metadata = {
        "success": True,
        "source": "flow5 embedded XFoil only",
        "optimizer": optimizer_key,
        "objective": float(final_score),
        "candidate_budget": budget,
        "maximum_candidate_budget": budget_controller.maximum_budget,
        "candidates_evaluated": int(search_evaluations),
        "budget_convergence": budget_controller.report(
            evaluations=search_evaluations
        ),
        "valid_candidates": len(valid_candidates),
        "outer_parallel_runners": outer_workers,
        "threads_per_runner": threads_per_runner,
        "total_threads_requested": int(total_threads),
        "detected_logical_cores": detected,
        "surrogate": surrogate_report,
        "checkpoint": checkpoint_report,
        "reference_chord_m": float(reference_chord_m),
        "cst_order": int(cst_order),
        "solver_coordinate_points": int(coordinate_points),
        "conditions": final_conditions,
        "baseline": {
            **baseline_profile.to_dict(),
            "within_design_envelope": bool(baseline_within_envelope),
            "solver_geometry": "source DAT cosine-resampled to 100 points",
            "search_converged": bool(math.isfinite(baseline_search.score) and baseline_search.response),
            "search_score": (
                float(baseline_search.score) if math.isfinite(baseline_search.score) else None
            ),
            "final_score": baseline_score,
            "selected": bool(winner_is_baseline),
            "error": baseline_search.error,
        },
        "selection": {
            "mode": "automatic",
            "selected_name": final_foil.name,
            "selected_family": final_foil.family,
            "selected_baseline": bool(winner_is_baseline),
            "minimum_improvement_percent": float(minimum_improvement_percent),
            "raw_best_improvement_vs_baseline_percent": raw_improvement_percent,
            "selected_improvement_vs_baseline_percent": selected_improvement,
            "baseline_retained_by_threshold": bool(threshold_applied),
            "finalists_evaluated": len(fine_results),
        },
        "finalists": [
            {
                "name": item[0].foil.name,
                "score": None if not math.isfinite(item[0].score) else float(item[0].score),
                "is_baseline": bool(item[1]),
                "within_design_envelope": bool(item[3]),
                "selected": (
                    item[0].foil.upper_weights,
                    item[0].foil.lower_weights,
                )
                == selected_key,
                "error": item[0].error,
            }
            for item in fine_results
        ],
        "top_candidates": [
            {
                "rank": rank + 1,
                "name": item.foil.name,
                "family": item.foil.family,
                "is_baseline": (
                    item.foil.upper_weights,
                    item.foil.lower_weights,
                )
                == baseline_key,
                "score": None if not math.isfinite(item.score) else float(item.score),
                "thickness": float(item.foil.thickness),
                "max_camber": float(item.foil.max_camber),
                "error": item.error,
            }
            for rank, item in enumerate(ranked[: min(20, len(ranked))])
        ],
        "solver": final_response.get("solver", {}),
    }
    return final_foil, final_response, metadata, selected_dat_text


def _wing_conditions(
    response: dict[str, Any],
    geometry: WingGeometry,
    fluid: Fluid,
    target_lift_n: float,
    alpha_bounds: tuple[float, float],
) -> tuple[float, list[dict[str, Any]], str | None]:
    drag_ratios: list[float] = []
    conditions: list[dict[str, Any]] = []
    penalty = 0.0
    for case in response["cases"]:
        speed = float(case["speed_m_s"])
        target_cl = target_lift_n / max(fluid.dynamic_pressure(speed) * geometry.area, 1e-12)
        point = interpolate_at_cl(case["points"], target_cl)
        if point is None:
            converged_cls = [
                float(row["cl"])
                for row in case.get("points", [])
                if row.get("cl") is not None and math.isfinite(float(row["cl"]))
            ]
            if converged_cls:
                coverage = f"{min(converged_cls):.4f}…{max(converged_cls):.4f}"
                reason = (
                    f"{speed:g} m/s noktasında hedef CL={target_cl:.4f}, "
                    f"yakınsayan flow5 aralığı={coverage}; hedef CL ince ağ "
                    "polar/yakınsama aralığının dışında kaldı"
                )
            else:
                reason = (
                    f"{speed:g} m/s noktasında hedef CL={target_cl:.4f}; "
                    "flow5 yakınsayan kanat polar noktası döndürmedi"
                )
            return math.inf, [], reason
        if point.get("cd", 0.0) <= 0.0:
            return (
                math.inf,
                [],
                f"{speed:g} m/s hedef CL={target_cl:.4f} noktasında flow5 pozitif CD döndürmedi",
            )
        alpha = float(point["alpha_deg"])
        alpha_violation = max(alpha_bounds[0] - alpha, 0.0, alpha - alpha_bounds[1])
        cl_peak = max(float(row["cl"]) for row in case["points"])
        stall_ratio = target_cl / max(cl_peak, 1e-8)
        q_area = fluid.dynamic_pressure(speed) * geometry.area
        drag_n = q_area * float(point["cd"])
        penalty += (
            0.018 * alpha_violation**2
            + 0.65 * max(0.0, stall_ratio - 0.92) ** 2
            + (0.8 if point.get("out_of_mesh") else 0.0)
        )
        drag_ratios.append(drag_n / max(target_lift_n, 1e-9))
        conditions.append(
            {
                "speed_m_s": speed,
                "target_cl": float(target_cl),
                "cl_max_converged": cl_peak,
                "stall_ratio": float(stall_ratio),
                "lift_n": float(target_lift_n),
                "drag_n": float(drag_n),
                "ld": float(target_lift_n / drag_n),
                "point": point,
                "method": case["method"],
                **(
                    {"panel_telemetry": case["panel_telemetry"]}
                    if isinstance(case.get("panel_telemetry"), dict)
                    else {}
                ),
            }
        )
    robust_drag = float(np.mean(drag_ratios) + 0.45 * np.max(drag_ratios))
    return robust_drag + penalty / max(len(conditions), 1), conditions, None


def _vector_to_geometry(
    vector: Iterable[float],
    *,
    multi_section: bool = False,
    winglet_enabled: bool = False,
) -> WingGeometry:
    values = list(vector)
    cursor = 5
    mid_chord_factor = 1.0
    mid_twist_deg: float | None = None
    if multi_section:
        mid_chord_factor = float(values[cursor])
        mid_twist_deg = float(values[cursor + 1])
        cursor += 2
    winglet_height = 0.0
    winglet_cant_deg = 90.0
    winglet_toe_deg = 0.0
    winglet_taper = 1.0
    if winglet_enabled:
        winglet_height = float(values[cursor])
        winglet_cant_deg = float(values[cursor + 1])
        winglet_toe_deg = float(values[cursor + 2])
        winglet_taper = float(values[cursor + 3])
    return WingGeometry(
        span=float(values[0]),
        root_chord=float(values[1]),
        taper=float(values[2]),
        sweep_deg=float(values[3]),
        tip_twist_deg=float(values[4]),
        alpha_deg=0.0,
        mid_chord_factor=mid_chord_factor,
        mid_twist_deg=mid_twist_deg,
        winglet_enabled=bool(winglet_enabled),
        winglet_height=winglet_height,
        winglet_cant_deg=winglet_cant_deg,
        winglet_toe_deg=winglet_toe_deg,
        winglet_taper=winglet_taper,
    )


def _geometry_to_vector(
    geometry: WingGeometry, multi_section: bool, winglet_enabled: bool = False
) -> np.ndarray:
    values = [
        geometry.span,
        geometry.root_chord,
        geometry.taper,
        geometry.sweep_deg,
        geometry.tip_twist_deg,
    ]
    if multi_section:
        values.extend([geometry.mid_chord_factor, geometry.effective_mid_twist_deg])
    if winglet_enabled:
        values.extend(
            [
                geometry.winglet_height,
                geometry.winglet_cant_deg,
                geometry.winglet_toe_deg,
                geometry.winglet_taper,
            ]
        )
    return np.asarray(values, dtype=float)


def mesh_convergence_report(
    coarse: WingCandidate,
    fine: WingCandidate,
    *,
    cd_tolerance_percent: float,
    alpha_tolerance_deg: float,
) -> dict[str, Any]:
    """Compare two meshes at the same requested lift without extrapolation."""
    coarse_conditions = coarse.conditions or []
    fine_conditions = fine.conditions or []
    rows: list[dict[str, Any]] = []
    for fine_item in fine_conditions:
        speed = float(fine_item["speed_m_s"])
        coarse_item = min(
            coarse_conditions,
            key=lambda item: abs(float(item["speed_m_s"]) - speed),
            default=None,
        )
        if coarse_item is None or abs(float(coarse_item["speed_m_s"]) - speed) > 1e-7:
            continue
        coarse_point = coarse_item["point"]
        fine_point = fine_item["point"]
        cd_coarse = float(coarse_point["cd"])
        cd_fine = float(fine_point["cd"])
        alpha_coarse = float(coarse_point["alpha_deg"])
        alpha_fine = float(fine_point["alpha_deg"])
        rows.append(
            {
                "speed_m_s": speed,
                "target_cl": float(fine_item["target_cl"]),
                "cd_coarse": cd_coarse,
                "cd_fine": cd_fine,
                "cd_change_percent": float(
                    100.0 * abs(cd_fine - cd_coarse) / max(abs(cd_fine), 1e-12)
                ),
                "alpha_coarse_deg": alpha_coarse,
                "alpha_fine_deg": alpha_fine,
                "alpha_change_deg": float(abs(alpha_fine - alpha_coarse)),
            }
        )
    max_cd = max((row["cd_change_percent"] for row in rows), default=math.inf)
    max_alpha = max((row["alpha_change_deg"] for row in rows), default=math.inf)
    passed = bool(
        len(rows) == len(fine_conditions) == len(coarse_conditions)
        and max_cd <= cd_tolerance_percent
        and max_alpha <= alpha_tolerance_deg
    )
    return {
        "enabled": True,
        "passed": passed,
        "cd_tolerance_percent": float(cd_tolerance_percent),
        "alpha_tolerance_deg": float(alpha_tolerance_deg),
        "max_cd_change_percent": float(max_cd),
        "max_alpha_change_deg": float(max_alpha),
        "conditions": rows,
        "coarse_mesh": (coarse.response or {}).get("mesh", {}),
        "fine_mesh": (fine.response or {}).get("mesh", {}),
    }


def optimize_wing_with_flow5(
    *,
    runner: Flow5Runner,
    foil: CSTAirfoilDesign,
    foil_dat_text: str,
    fluid: Fluid,
    speeds_m_s: list[float],
    reference_speed_m_s: float,
    target_lift_n: float,
    span_bounds: tuple[float, float],
    root_chord_bounds: tuple[float, float],
    taper_bounds: tuple[float, float],
    sweep_bounds: tuple[float, float],
    twist_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    candidate_budget: int,
    finalists: int,
    seed: int,
    total_threads: int,
    coordinate_points: int,
    search_method: str,
    final_method: str,
    alpha_step_search_deg: float,
    alpha_step_final_deg: float,
    ncrit: float,
    xtr_top: float,
    xtr_bottom: float,
    search_mesh: Flow5Mesh = Flow5Mesh(10, 14),
    final_mesh: Flow5Mesh = Flow5Mesh(14, 22),
    convergence_mesh: Flow5Mesh = Flow5Mesh(20, 32),
    mesh_convergence_enabled: bool = True,
    mesh_cd_tolerance_percent: float = 2.0,
    mesh_alpha_tolerance_deg: float = 0.25,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    optimizer: str = "differential_evolution",
    multi_section_geometry_enabled: bool = False,
    mid_chord_factor_bounds: tuple[float, float] = (0.8, 1.2),
    mid_twist_bounds: tuple[float, float] | None = None,
    winglet_optimization_enabled: bool = False,
    winglet_height_bounds: tuple[float, float] = (0.05, 0.40),
    winglet_cant_bounds: tuple[float, float] = (60.0, 90.0),
    winglet_toe_bounds: tuple[float, float] = (-3.0, 3.0),
    winglet_taper_bounds: tuple[float, float] = (0.30, 1.0),
    section_foils: tuple[AirfoilLike, ...] | None = None,
    section_foil_dat_texts: tuple[str, ...] | None = None,
    structural_settings: StructuralSettings = StructuralSettings(),
    hydro_settings: HydroSettings = HydroSettings(),
    initial_geometry: WingGeometry | None = None,
    surrogate_settings: SurrogateSettings = SurrogateSettings(),
    budget_escalation_settings: BudgetEscalationSettings = BudgetEscalationSettings(),
    checkpoint_store: OptimizerCheckpointStore | None = None,
    checkpoint_key: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Search planforms and validate finalists entirely with flow5 3D analyses."""
    budget = max(8, int(candidate_budget))
    budget_controller = BudgetEscalationController(
        budget, budget_escalation_settings, hard_limit=2048
    )
    rng = np.random.default_rng(seed + 757)
    bounds_list: list[tuple[float, float]] = [
        span_bounds,
        root_chord_bounds,
        taper_bounds,
        sweep_bounds,
        twist_bounds,
    ]
    if multi_section_geometry_enabled:
        bounds_list.extend([mid_chord_factor_bounds, mid_twist_bounds or twist_bounds])
    if winglet_optimization_enabled:
        bounds_list.extend(
            [
                winglet_height_bounds,
                winglet_cant_bounds,
                winglet_toe_bounds,
                winglet_taper_bounds,
            ]
        )
        if min(
            search_mesh.half_span_panels,
            final_mesh.half_span_panels,
            convergence_mesh.half_span_panels,
        ) < 6:
            raise ValueError("Winglet çözümü için yarı açıklık panel sayısı en az 6 olmalı")
    bounds = np.asarray(bounds_list, dtype=float)
    candidates: list[WingCandidate] = []

    def geometry_from_vector(vector: Iterable[float]) -> WingGeometry:
        return _vector_to_geometry(
            vector,
            multi_section=multi_section_geometry_enabled,
            winglet_enabled=winglet_optimization_enabled,
        )

    def vector_from_geometry(geometry: WingGeometry) -> np.ndarray:
        return _geometry_to_vector(
            geometry,
            multi_section_geometry_enabled,
            winglet_optimization_enabled,
        )

    def evaluate(
        geometry: WingGeometry,
        method: str,
        alpha_step: float,
        save: bool,
        mesh: Flow5Mesh,
    ) -> WingCandidate:
        _check_cancelled(cancel_event)
        if not geometry.winglet_geometry_valid:
            return WingCandidate(
                geometry,
                error="Winglet yatay izdüşümü ana yarı açıklık için yer bırakmıyor",
            )
        try:
            response = runner.analyze_wing(
                foil=foil,
                geometry=geometry,
                fluid=fluid,
                speeds_m_s=speeds_m_s,
                method=method,
                alpha_min_deg=alpha_bounds[0],
                alpha_max_deg=alpha_bounds[1],
                alpha_step_deg=alpha_step,
                max_threads=total_threads,
                coordinate_points=coordinate_points,
                foil_dat_text=foil_dat_text,
                ncrit=ncrit,
                xtr_top=xtr_top,
                xtr_bottom=xtr_bottom,
                save_project=save,
                panel_telemetry=bool(save and hydro_settings.enabled),
                panel_telemetry_target_lift_n=(
                    target_lift_n if save and hydro_settings.enabled else None
                ),
                mesh=mesh,
                section_foils=section_foils,
                section_foil_dat_texts=section_foil_dat_texts,
            )
            score, conditions, condition_error = _wing_conditions(
                response,
                geometry,
                fluid,
                target_lift_n,
                alpha_bounds,
            )
            structural = analyze_structure(
                geometry=geometry,
                foil_thickness_ratio=foil.thickness,
                fluid=fluid,
                conditions=conditions,
                settings=structural_settings,
            )
            if structural_settings.enabled:
                score += 0.80 * float(structural.get("penalty", 0.0))
            hydro = analyze_hydro(
                geometry=geometry,
                fluid=fluid,
                conditions=conditions,
                settings=hydro_settings,
            )
            if hydro_settings.enabled:
                score += 0.80 * float(hydro.get("penalty", 0.0))
            return WingCandidate(
                geometry,
                score,
                response,
                conditions,
                error=condition_error,
                structural=structural,
                hydro=hydro,
            )
        except Flow5CancelledError:
            raise
        except Exception as exc:
            return WingCandidate(geometry, error=str(exc)[-300:])

    optimizer_key = optimizer.strip().lower()
    if optimizer_key not in {"nsga2", "differential_evolution", "adaptive_elite"}:
        raise ValueError(
            "Kanat optimizeri nsga2, differential_evolution veya adaptive_elite olmalı"
        )
    midpoint = np.mean(bounds, axis=1)
    dimension = bounds.shape[0]
    objective_specs = wing_objective_specs(
        structural_enabled=structural_settings.enabled,
        hydro_enabled=hydro_settings.optimization_active,
    )
    objective_keys = [item["key"] for item in objective_specs]
    search_evaluations = 0
    completed_generation = 0
    score_history: list[float] = []
    objective_history: list[list[float]] = []
    nsga2_report: dict[str, Any] = {
        "enabled": optimizer_key == "nsga2",
        "algorithm": "NSGA-II" if optimizer_key == "nsga2" else None,
        "objective_specs": objective_specs,
        "constraint_handling": "Deb constraint-domination",
        "selection": "Pareto rank + crowding distance",
    }
    surrogate_report: dict[str, Any] = {
        "enabled": bool(surrogate_settings.enabled),
        "trained": False,
        "reason": "yalnız differential_evolution ile etkin",
    }
    if optimizer_key == "nsga2":
        surrogate_report = {
            "enabled": False,
            "requested": bool(surrogate_settings.enabled),
            "trained": False,
            "reason": (
                "NSGA-II çok-amaçlı çeşitliliğini skaler surrogate ile bozmamak için "
                "kanat surrogate ön elemesi kullanılmadı"
            ),
            "real_solver_evaluations": 0,
            "proposals_screened": 0,
            "finalists_always_solver_verified": True,
        }
    checkpoint_report: dict[str, Any] = {
        "enabled": bool(checkpoint_store and checkpoint_store.enabled and checkpoint_key),
        "resumed": False,
        "evaluations_restored": 0,
        "generation_restored": 0,
    }

    def record_candidate(candidate: WingCandidate) -> None:
        nonlocal search_evaluations
        candidates.append(candidate)
        score_history.append(float(candidate.score))
        objective_history.append(
            wing_candidate_objectives(candidate, objective_keys).tolist()
        )
        search_evaluations += 1

    def frontier_size_now() -> int:
        usable = [
            item
            for item in candidates
            if math.isfinite(item.score) and item.response and item.conditions
        ]
        if not usable:
            return 0
        fronts = fast_non_dominated_sort(
            usable,
            objective_keys,
        )
        return len(fronts[0]) if fronts else 0

    def observe_wing_budget() -> dict[str, Any] | None:
        checkpoint = budget_controller.observe(
            scores=score_history,
            objectives=objective_history,
            frontier_size=frontier_size_now(),
        )
        if checkpoint is not None:
            decision = checkpoint["decision"]
            if decision == "escalated":
                message = (
                    f"Kanat Pareto cephesi hareket ediyor; bütçe "
                    f"{checkpoint['next_budget']} adaya yükseltildi"
                )
            elif decision == "converged":
                message = (
                    f"Kanat bütçesi yeterli: cephe değişimi "
                    f"%{checkpoint['controlling_change_percent']:.3f}"
                )
            elif decision == "maximum_budget_reached":
                message = "Kanat azami bütçeye ulaştı; sonuç bütçe-sınırlı işaretlendi"
            else:
                message = "Sabit kanat aday bütçesi tamamlandı"
            _emit_progress(
                progress_callback,
                "wing_budget",
                search_evaluations,
                budget_controller.progress_total,
                message,
            )
        return checkpoint

    def latin_hypercube(size: int) -> np.ndarray:
        unit_population = np.empty((size, dimension), dtype=float)
        for variable in range(dimension):
            unit_population[:, variable] = (
                rng.permutation(size) + rng.random(size)
            ) / size
        vectors = bounds[:, 0] + unit_population * (bounds[:, 1] - bounds[:, 0])
        vectors[0] = (
            np.clip(
                vector_from_geometry(initial_geometry),
                bounds[:, 0],
                bounds[:, 1],
            )
            if initial_geometry is not None
            else midpoint
        )
        return vectors

    def restore_histories(restored: dict[str, Any], population: list[WingCandidate]) -> None:
        nonlocal search_evaluations, score_history, objective_history
        restored_count = max(
            int(restored.get("evaluations_completed", len(population))), len(population)
        )
        score_history = [float(value) for value in restored.get("score_history", [])]
        objective_history = [
            [float(value) for value in row]
            for row in restored.get("objective_history", [])
        ]
        fallback_score = min((item.score for item in population), default=math.inf)
        fallback_objectives = (
            wing_candidate_objectives(
                min(population, key=lambda item: item.score), objective_keys
            ).tolist()
            if population
            else [math.inf] * len(objective_keys)
        )
        score_history.extend(
            [float(fallback_score)] * max(0, restored_count - len(score_history))
        )
        objective_history.extend(
            [list(fallback_objectives)]
            * max(0, restored_count - len(objective_history))
        )
        score_history = score_history[:restored_count]
        objective_history = objective_history[:restored_count]
        search_evaluations = restored_count
        budget_controller.restore(restored.get("budget_controller"))

    if optimizer_key in {"differential_evolution", "nsga2"}:
        population_size = min(
            budget,
            max(8, (2 * dimension + 2) if optimizer_key == "differential_evolution" else 4 * dimension),
        )
        if optimizer_key == "nsga2" and population_size % 2:
            population_size = max(8, population_size - 1)
        population: list[WingCandidate] = []
        generation = 0
        advisor = RBFSurrogateAdvisor(bounds, surrogate_settings)
        restored = (
            checkpoint_store.load(checkpoint_key)
            if checkpoint_store is not None and checkpoint_key
            else None
        )
        if restored and restored.get("optimizer") == optimizer_key:
            try:
                rng.bit_generator.state = restored["rng_state"]
                generation = int(restored.get("generation", 0))
                for raw_vector in restored["population_vectors"]:
                    population.append(
                        evaluate(
                            geometry_from_vector(np.asarray(raw_vector, dtype=float)),
                            search_method,
                            alpha_step_search_deg,
                            False,
                            search_mesh,
                        )
                    )
                candidates = list(population)
                restore_histories(restored, population)
                if optimizer_key == "differential_evolution":
                    surrogate_state = restored.get("surrogate", {})
                    advisor.restore(
                        list(surrogate_state.get("vectors", [])),
                        list(surrogate_state.get("scores", [])),
                    )
                checkpoint_report.update(
                    resumed=True,
                    evaluations_restored=search_evaluations,
                    generation_restored=generation,
                )
                _emit_progress(
                    progress_callback,
                    "wing_search",
                    search_evaluations,
                    budget_controller.progress_total,
                    f"{optimizer_key.upper()} kanat checkpoint'i yüklendi: nesil {generation}",
                )
            except (KeyError, TypeError, ValueError):
                population = []
                candidates = []
                score_history = []
                objective_history = []
                search_evaluations = 0

        def save_population_checkpoint() -> None:
            if checkpoint_store is None or not checkpoint_key or not population:
                return
            checkpoint_store.save(
                checkpoint_key,
                {
                    "optimizer": optimizer_key,
                    "generation": generation,
                    "evaluations_completed": search_evaluations,
                    "population_vectors": [
                        vector_from_geometry(item.geometry).tolist()
                        for item in population
                    ],
                    "rng_state": rng.bit_generator.state,
                    "surrogate": advisor.state() if optimizer_key == "differential_evolution" else {},
                    "score_history": score_history,
                    "objective_history": objective_history,
                    "budget_controller": budget_controller.state(),
                },
            )

        if not population:
            for vector in latin_hypercube(population_size):
                _check_cancelled(cancel_event)
                candidate = evaluate(
                    geometry_from_vector(vector),
                    search_method,
                    alpha_step_search_deg,
                    False,
                    search_mesh,
                )
                population.append(candidate)
                record_candidate(candidate)
                if optimizer_key == "differential_evolution":
                    advisor.record(vector, candidate.score)
                _emit_progress(
                    progress_callback,
                    "wing_search",
                    search_evaluations,
                    budget_controller.progress_total,
                    f"{optimizer_key.upper()} kanat popülasyonu: {search_evaluations}/{budget_controller.progress_total}",
                )
            observe_wing_budget()
            save_population_checkpoint()

        if optimizer_key == "differential_evolution":
            while not budget_controller.stopped:
                _check_cancelled(cancel_event)
                if not budget_controller.should_evaluate(search_evaluations):
                    observe_wing_budget()
                    save_population_checkpoint()
                    continue
                generation += 1
                for target_index in range(len(population)):
                    if not budget_controller.should_evaluate(search_evaluations):
                        break
                    target_vector = vector_from_geometry(
                        population[target_index].geometry
                    )
                    proposal_vectors: list[np.ndarray] = []
                    proposal_count = (
                        surrogate_settings.proposals_per_real_evaluation
                        if advisor.ready
                        else 1
                    )
                    for _ in range(proposal_count):
                        choices = [
                            index for index in range(len(population)) if index != target_index
                        ]
                        a_index, b_index, c_index = rng.choice(choices, size=3, replace=False)
                        source_vectors = [
                            vector_from_geometry(population[int(index)].geometry)
                            for index in (a_index, b_index, c_index)
                        ]
                        mutant = source_vectors[0] + float(rng.uniform(0.55, 0.90)) * (
                            source_vectors[1] - source_vectors[2]
                        )
                        crossover = rng.random(dimension) < 0.82
                        crossover[int(rng.integers(0, dimension))] = True
                        proposal_vectors.append(
                            np.clip(
                                np.where(crossover, mutant, target_vector),
                                bounds[:, 0],
                                bounds[:, 1],
                            )
                        )
                    trial_vector = proposal_vectors[advisor.choose(proposal_vectors)]
                    trial = evaluate(
                        geometry_from_vector(trial_vector),
                        search_method,
                        alpha_step_search_deg,
                        False,
                        search_mesh,
                    )
                    record_candidate(trial)
                    advisor.record(trial_vector, trial.score)
                    if trial.score <= population[target_index].score:
                        population[target_index] = trial
                    _emit_progress(
                        progress_callback,
                        "wing_search",
                        search_evaluations,
                        budget_controller.progress_total,
                        f"DE kanat nesli {generation}: {search_evaluations}/{budget_controller.progress_total}",
                    )
                observe_wing_budget()
                save_population_checkpoint()
            surrogate_report = advisor.report(
                real_evaluations=search_evaluations,
                budget=budget_controller.maximum_budget,
            )
        else:
            def tournament_index(ranks: dict[int, int], crowding: dict[int, float]) -> int:
                left, right = [int(value) for value in rng.choice(len(population), size=2, replace=False)]
                left_key = (ranks.get(left, math.inf), -crowding.get(left, 0.0), population[left].score)
                right_key = (ranks.get(right, math.inf), -crowding.get(right, 0.0), population[right].score)
                return left if left_key <= right_key else right

            def sbx_children(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                eta = 15.0
                random_values = np.clip(rng.random(dimension), 1e-12, 1.0 - 1e-12)
                beta = np.where(
                    random_values <= 0.5,
                    (2.0 * random_values) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - random_values))) ** (1.0 / (eta + 1.0)),
                )
                beta = np.where(rng.random(dimension) < 0.5, beta, 1.0)
                first = 0.5 * ((1.0 + beta) * left + (1.0 - beta) * right)
                second = 0.5 * ((1.0 - beta) * left + (1.0 + beta) * right)
                span = bounds[:, 1] - bounds[:, 0]
                mutation_probability = 1.0 / dimension
                for child in (first, second):
                    mask = rng.random(dimension) < mutation_probability
                    for variable in np.flatnonzero(mask):
                        eta_mutation = 20.0
                        lower, upper = bounds[variable]
                        if span[variable] <= 1.0e-15:
                            child[variable] = lower
                            continue
                        value = float(np.clip(child[variable], lower, upper))
                        delta_lower = (value - lower) / span[variable]
                        delta_upper = (upper - value) / span[variable]
                        random_value = float(rng.random())
                        exponent = 1.0 / (eta_mutation + 1.0)
                        if random_value <= 0.5:
                            factor = 1.0 - delta_lower
                            transformed = (
                                2.0 * random_value
                                + (1.0 - 2.0 * random_value)
                                * factor ** (eta_mutation + 1.0)
                            )
                            delta = transformed**exponent - 1.0
                        else:
                            factor = 1.0 - delta_upper
                            transformed = (
                                2.0 * (1.0 - random_value)
                                + 2.0
                                * (random_value - 0.5)
                                * factor ** (eta_mutation + 1.0)
                            )
                            delta = 1.0 - transformed**exponent
                        child[variable] = value + delta * span[variable]
                    np.clip(child, bounds[:, 0], bounds[:, 1], out=child)
                return first, second

            while not budget_controller.stopped:
                _check_cancelled(cancel_event)
                if not budget_controller.should_evaluate(search_evaluations):
                    observe_wing_budget()
                    save_population_checkpoint()
                    continue
                generation += 1
                _, ranks, crowding = nsga2_rank_and_crowding(
                    population,
                    objective_keys,
                )
                remaining = min(
                    population_size,
                    budget_controller.current_target - search_evaluations,
                )
                offspring_vectors: list[np.ndarray] = []
                while len(offspring_vectors) < remaining:
                    left = vector_from_geometry(
                        population[tournament_index(ranks, crowding)].geometry
                    )
                    right = vector_from_geometry(
                        population[tournament_index(ranks, crowding)].geometry
                    )
                    offspring_vectors.extend(sbx_children(left, right))
                offspring: list[WingCandidate] = []
                for vector in offspring_vectors[:remaining]:
                    candidate = evaluate(
                        geometry_from_vector(vector),
                        search_method,
                        alpha_step_search_deg,
                        False,
                        search_mesh,
                    )
                    offspring.append(candidate)
                    record_candidate(candidate)
                    _emit_progress(
                        progress_callback,
                        "wing_search",
                        search_evaluations,
                        budget_controller.progress_total,
                        f"NSGA-II kanat nesli {generation}: {search_evaluations}/{budget_controller.progress_total}",
                    )
                population = nsga2_environmental_selection(
                    [*population, *offspring],
                    population_size,
                    objective_keys,
                )
                observe_wing_budget()
                save_population_checkpoint()
            final_fronts, _, _ = nsga2_rank_and_crowding(
                population,
                objective_keys,
            )
            nsga2_report.update(
                population_size=population_size,
                generations_completed=generation,
                final_population_frontier_size=(len(final_fronts[0]) if final_fronts else 0),
                real_solver_evaluations=search_evaluations,
                variation="simulated binary crossover + polynomial bounded mutation",
            )
            surrogate_report["real_solver_evaluations"] = search_evaluations
        completed_generation = generation
    else:
        queue: list[np.ndarray] = [
            np.clip(
                vector_from_geometry(initial_geometry),
                bounds[:, 0],
                bounds[:, 1],
            )
            if initial_geometry is not None
            else midpoint
        ]
        while not budget_controller.stopped:
            _check_cancelled(cancel_event)
            if not budget_controller.should_evaluate(search_evaluations):
                observe_wing_budget()
                continue
            finite = sorted(
                (item for item in candidates if math.isfinite(item.score)),
                key=lambda item: item.score,
            )
            elites = finite[: min(7, len(finite))]
            progress = search_evaluations / max(budget_controller.maximum_budget - 1, 1)
            if queue:
                vector = queue.pop(0)
            elif elites and rng.random() < 0.76:
                parent = vector_from_geometry(
                    elites[int(rng.integers(0, len(elites)))].geometry
                )
                scale = (0.20 * (1.0 - progress) + 0.035 * progress) * (
                    bounds[:, 1] - bounds[:, 0]
                )
                vector = np.clip(
                    parent + rng.normal(0.0, scale), bounds[:, 0], bounds[:, 1]
                )
            else:
                vector = rng.uniform(bounds[:, 0], bounds[:, 1])
            record_candidate(
                evaluate(
                    geometry_from_vector(vector),
                    search_method,
                    alpha_step_search_deg,
                    False,
                    search_mesh,
                )
            )
            _emit_progress(
                progress_callback,
                "wing_search",
                search_evaluations,
                budget_controller.progress_total,
                f"Kanat adayları çözüldü: {search_evaluations}/{budget_controller.progress_total}",
            )
            observe_wing_budget()

    valid_search = [item for item in candidates if math.isfinite(item.score) and item.response]
    if not valid_search:
        errors = "; ".join(item.error for item in candidates if item.error)[:900]
        raise RuntimeError(f"flow5 kanat aramasında geçerli aday bulunamadı: {errors}")
    selected_indices, finalist_selection = select_wing_finalist_indices(
        valid_search,
        finalists,
        objective_keys,
        optimizer=optimizer_key,
    )
    selected_for_final = [valid_search[index] for index in selected_indices]
    if optimizer_key == "nsga2":
        nsga2_report["finalist_selection"] = (
            "lowest hard-constraint violation, then scalar compromise + "
            "crowding-diverse Pareto representatives; scalar-only winner is "
            "always retained as a diagnostic"
        )
    final_candidates: list[WingCandidate] = []
    for index, item in enumerate(selected_for_final):
        _check_cancelled(cancel_event)
        final_candidates.append(
            evaluate(item.geometry, final_method, alpha_step_final_deg, False, final_mesh)
        )
        _emit_progress(
            progress_callback,
            "wing_final",
            index + 1,
            len(selected_for_final),
            f"Panel finalistleri çözüldü: {index + 1}/{len(selected_for_final)}",
        )
    valid_final = [item for item in final_candidates if math.isfinite(item.score) and item.response]
    if not valid_final:
        errors = "; ".join(item.error for item in final_candidates if item.error)[:900]
        raise RuntimeError(f"flow5 son panel doğrulamasında geçerli kanat bulunamadı: {errors}")
    coarse_optimum = min(valid_final, key=wing_candidate_selection_key)
    scalar_coarse_optimum = min(valid_final, key=lambda item: item.score)

    def candidate_is_valid(candidate: WingCandidate) -> bool:
        return bool(
            math.isfinite(candidate.score)
            and candidate.response
            and candidate.conditions
        )

    def finalize_output_candidate(
        coarse: WingCandidate,
    ) -> tuple[
        WingCandidate | None,
        dict[str, Any] | None,
        Flow5Mesh,
        bool,
        str | None,
    ]:
        requested_mesh = convergence_mesh if mesh_convergence_enabled else final_mesh
        attempted = evaluate(
            coarse.geometry,
            final_method,
            alpha_step_final_deg,
            True,
            requested_mesh,
        )
        if candidate_is_valid(attempted):
            if mesh_convergence_enabled:
                report = mesh_convergence_report(
                    coarse,
                    attempted,
                    cd_tolerance_percent=mesh_cd_tolerance_percent,
                    alpha_tolerance_deg=mesh_alpha_tolerance_deg,
                )
                report.update(
                    {
                        "status": "passed" if report["passed"] else "tolerance_exceeded",
                        "fine_mesh_valid": True,
                        "fallback_used": False,
                        "reason": None,
                        "output_mesh": (attempted.response or {}).get(
                            "mesh", requested_mesh.to_dict()
                        ),
                    }
                )
            else:
                report = {
                    "enabled": False,
                    "passed": True,
                    "status": "disabled",
                    "fine_mesh_valid": None,
                    "fallback_used": False,
                    "reason": None,
                    "coarse_mesh": final_mesh.to_dict(),
                    "fine_mesh": final_mesh.to_dict(),
                    "output_mesh": (attempted.response or {}).get(
                        "mesh", final_mesh.to_dict()
                    ),
                    "conditions": [],
                }
            return attempted, report, requested_mesh, False, None

        attempted_error = attempted.error or (
            "İnce ağ flow5 yanıtı hedef taşıma noktasında geçerli koşul üretmedi"
        )
        if not mesh_convergence_enabled:
            return None, None, requested_mesh, False, attempted_error

        # Mesh convergence is a verification gate, not a reason to discard a
        # solver-verified finalist.  Re-run the already valid final mesh with
        # artifact/panel export enabled and return it as a review result.
        fallback = evaluate(
            coarse.geometry,
            final_method,
            alpha_step_final_deg,
            True,
            final_mesh,
        )
        if not candidate_is_valid(fallback):
            fallback_error = fallback.error or (
                "Final ağ tekrarında hedef taşıma noktasında geçerli koşul üretilemedi"
            )
            return (
                None,
                None,
                final_mesh,
                False,
                f"İnce ağ: {attempted_error}; final ağ tekrarı: {fallback_error}",
            )
        report = {
            "enabled": True,
            "passed": False,
            "status": "fine_mesh_invalid_fallback_to_final_mesh",
            "fine_mesh_valid": False,
            "fallback_used": True,
            "reason": attempted_error,
            "cd_tolerance_percent": float(mesh_cd_tolerance_percent),
            "alpha_tolerance_deg": float(mesh_alpha_tolerance_deg),
            "max_cd_change_percent": None,
            "max_alpha_change_deg": None,
            "conditions": [],
            "coarse_mesh": (coarse.response or {}).get("mesh", final_mesh.to_dict()),
            "fine_mesh": (attempted.response or {}).get(
                "mesh", convergence_mesh.to_dict()
            ),
            "output_mesh": (fallback.response or {}).get(
                "mesh", final_mesh.to_dict()
            ),
        }
        return fallback, report, final_mesh, True, attempted_error

    optimum, convergence, output_mesh, output_fallback_used, output_error = (
        finalize_output_candidate(coarse_optimum)
    )
    _emit_progress(
        progress_callback,
        "mesh_convergence",
        1,
        2,
        (
            "İnce ağ hedef CL'yi kapsamadı; doğrulanmış final ağ sonucu korunuyor"
            if output_fallback_used
            else "Seçilen kanat yüksek çözünürlüklü ağda çözüldü"
        ),
    )
    if optimum is None or convergence is None:
        raise RuntimeError(
            "flow5 yüksek çözünürlüklü son kanadı çözemedi: "
            + (output_error or "bilinmeyen final ağ hatası")
        )

    same_scalar_geometry = bool(
        np.allclose(
            vector_from_geometry(coarse_optimum.geometry),
            vector_from_geometry(scalar_coarse_optimum.geometry),
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    scalar_optimum: WingCandidate | None
    scalar_convergence: dict[str, Any] | None
    scalar_output_error: str | None = None
    if same_scalar_geometry:
        scalar_optimum = optimum
        scalar_convergence = convergence
    else:
        (
            scalar_optimum,
            scalar_convergence,
            _,
            scalar_fallback_used,
            scalar_finalize_error,
        ) = finalize_output_candidate(
            scalar_coarse_optimum
        )
        _emit_progress(
            progress_callback,
            "scalar_only_final",
            1,
            1,
            (
                "Skaler alternatif ince ağda çözülemedi; final ağ çıktısı korundu"
                if scalar_fallback_used
                else "Fizibilite önceliği olmayan alternatif yüksek çözünürlükte çözüldü"
            ),
        )
        if scalar_optimum is None or scalar_convergence is None:
            scalar_output_error = scalar_finalize_error or (
                "Skaler-puan alternatifi son çıktı ağında çözülemedi"
            )
            scalar_optimum = None
            scalar_convergence = None

    # The reference is deliberately equal-area; it is a comparison, not a search candidate.
    baseline_chord = float(optimum.geometry.area / optimum.geometry.span)
    baseline_geometry = WingGeometry(
        optimum.geometry.span,
        baseline_chord,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0 if multi_section_geometry_enabled else None,
    )
    baseline = evaluate(
        baseline_geometry,
        final_method,
        alpha_step_final_deg,
        False,
        output_mesh,
    )
    _emit_progress(
        progress_callback,
        "mesh_convergence",
        2,
        2,
        "Ağ yakınsaması ve dikdörtgen referans tamamlandı",
    )
    if not math.isfinite(baseline.score) or not baseline.response:
        raise RuntimeError(f"flow5 dikdörtgen referans kanadı çözemedi: {baseline.error}")

    pareto_analysis = build_pareto_analysis(
        valid_search, optimum, optimizer=optimizer_key
    )

    def as_result(candidate: WingCandidate) -> dict[str, Any]:
        assert candidate.conditions
        condition = min(
            candidate.conditions,
            key=lambda item: abs(float(item["speed_m_s"]) - reference_speed_m_s),
        )
        point = condition["point"]
        alpha = float(point["alpha_deg"])
        geometry = WingGeometry(
            candidate.geometry.span,
            candidate.geometry.root_chord,
            candidate.geometry.taper,
            candidate.geometry.sweep_deg,
            candidate.geometry.tip_twist_deg,
            alpha,
            candidate.geometry.mid_chord_factor,
            candidate.geometry.mid_twist_deg,
            candidate.geometry.winglet_enabled,
            candidate.geometry.winglet_height,
            candidate.geometry.winglet_cant_deg,
            candidate.geometry.winglet_toe_deg,
            candidate.geometry.winglet_taper,
        )
        cd_total = float(point["cd"])
        cd_profile = float(point.get("cdv", 0.0))
        cd_induced = float(point.get("cdi", max(cd_total - cd_profile, 0.0)))
        cl = float(condition["target_cl"])
        efficiency = (
            cl**2 / (math.pi * geometry.aspect_ratio * cd_induced)
            if cd_induced > 1e-12
            else 0.0
        )
        return {
            "geometry": geometry.to_dict(),
            "cl": cl,
            "cd_profile": cd_profile,
            "cd_induced": cd_induced,
            "cd_total": cd_total,
            "lift_n": float(condition["lift_n"]),
            "drag_n": float(condition["drag_n"]),
            "ld": float(condition["ld"]),
            "span_efficiency": float(efficiency),
            "induced_drag_fraction_percent": float(
                100.0 * cd_induced / max(cd_total, 1.0e-12)
            ),
            "root_bending_moment_nm": float(point.get("root_bending_moment_nm", 0.0)),
            "stall_ratio": float(condition["stall_ratio"]),
            "reynolds_root": fluid.reynolds(reference_speed_m_s, geometry.root_chord),
            "reynolds_tip": fluid.reynolds(reference_speed_m_s, geometry.tip_chord),
            "section_polar_source": "flow5 viscous on-the-fly / embedded XFoil",
            "distribution": point.get("distribution", []),
            "method": condition["method"],
            "conditions": candidate.conditions,
            "structural": candidate.structural
            or {"enabled": False, "performed": False, "passed": True},
            "hydro": candidate.hydro
            or {"enabled": False, "performed": False, "passed": True},
        }

    optimum_result = as_result(optimum)
    baseline_result = as_result(baseline)
    feasible = bool(
        wing_constraint_violation(optimum) <= 1.0e-12
        and convergence["passed"]
    )
    scalar_result = as_result(scalar_optimum) if scalar_optimum is not None else None
    scalar_violation = (
        wing_constraint_violation(scalar_optimum)
        if scalar_optimum is not None
        else wing_constraint_violation(scalar_coarse_optimum)
    )
    scalar_feasible = bool(
        scalar_optimum is not None
        and scalar_violation <= 1.0e-12
        and scalar_convergence is not None
        and scalar_convergence.get("passed", False)
    )
    selection_comparison = {
        "definition": (
            "Aynı finalist havuzunda sert-kısıt önceliği ile yalnız skaler toplam "
            "amaç seçiminin karşılaştırması"
        ),
        "feasibility_first": {
            "policy": "önce en düşük sert-kısıt ihlali, eşitlikte en düşük skaler amaç",
            "available": True,
            "deliverable": True,
            "feasible": bool(feasible),
            "constraint_violation": float(wing_constraint_violation(optimum)),
            "objective": float(optimum.score),
            "mesh_convergence": convergence,
            "wing": optimum_result,
        },
        "scalar_only": {
            "policy": "fizibilite önceliği yok; yalnız en düşük skaler toplam amaç",
            "available": scalar_result is not None,
            "deliverable": scalar_result is not None,
            "same_as_feasibility_first": bool(same_scalar_geometry),
            "feasible": bool(scalar_feasible),
            "constraint_violation": float(scalar_violation),
            "objective": float(
                scalar_optimum.score
                if scalar_optimum is not None
                else scalar_coarse_optimum.score
            ),
            "mesh_convergence": scalar_convergence,
            "wing": scalar_result,
            "coarse_wing": (
                None if scalar_result is not None else as_result(scalar_coarse_optimum)
            ),
            "error": scalar_output_error,
        },
        "finalist_selection": finalist_selection,
    }
    if checkpoint_store is not None and checkpoint_key:
        checkpoint_store.clear(checkpoint_key)
        checkpoint_report["generations_completed"] = completed_generation
        checkpoint_report["store"] = checkpoint_store.stats()
    metadata = {
        "success": True,
        "feasible": bool(feasible),
        "source": "flow5 3D only",
        "optimizer": optimizer_key,
        "search_method": search_method.upper(),
        "final_method": final_method.upper(),
        "objective": float(optimum.score),
        "candidate_budget": budget,
        "maximum_candidate_budget": budget_controller.maximum_budget,
        "candidates_evaluated": int(search_evaluations),
        "budget_convergence": budget_controller.report(
            evaluations=search_evaluations
        ),
        "valid_search_candidates": len(valid_search),
        "finalists_requested": int(finalist_selection["requested_finalists"]),
        "finalists_evaluated": len(final_candidates),
        "high_resolution_final_evaluated": True,
        "threads_inside_flow5": int(total_threads),
        "foil_coordinate_points": int(coordinate_points),
        "outer_parallel_runners": 1,
        "oversubscription_prevented": True,
        "surrogate": surrogate_report,
        "multi_objective": nsga2_report,
        "checkpoint": checkpoint_report,
        "drag_reduction_vs_rectangular_percent": float(
            100.0
            * (baseline_result["drag_n"] - optimum_result["drag_n"])
            / max(baseline_result["drag_n"], 1e-12)
        ),
        "conditions": optimum.conditions,
        "solver_telemetry": {
            "out_of_mesh_points": sum(
                bool(item["point"].get("out_of_mesh", False))
                for item in optimum.conditions or []
            ),
            "nonconverged_viscous_points": sum(
                not bool(item["point"].get("viscous_converged", True))
                for item in optimum.conditions or []
            ),
            "spanwise_distribution_available": all(
                bool(item["point"].get("distribution")) for item in optimum.conditions or []
            ),
            "cp_min_available": all(
                item["point"].get("cp_min") is not None for item in optimum.conditions or []
            ),
            "panel_map_available": bool(
                (optimum.hydro or {}).get("panel_map_available", False)
            ),
        },
        "mesh_convergence": convergence,
        "search_mesh": search_mesh.to_dict(),
        "final_mesh": final_mesh.to_dict(),
        "output_mesh": output_mesh.to_dict(),
        "multi_section_geometry_enabled": bool(multi_section_geometry_enabled),
        "section_count": 4 if winglet_optimization_enabled else 3,
        "independent_mid_geometry_variables": bool(multi_section_geometry_enabled),
        "winglet_optimization_enabled": bool(winglet_optimization_enabled),
        "winglet_geometry_model": (
            "flow5 high-dihedral fourth section; zero quarter-chord winglet sweep"
            if winglet_optimization_enabled
            else "disabled"
        ),
        "winglet_bounds": {
            "height_m": list(winglet_height_bounds),
            "cant_deg": list(winglet_cant_bounds),
            "toe_deg": list(winglet_toe_bounds),
            "taper": list(winglet_taper_bounds),
        },
        "spanwise_airfoil_count": len(section_foils) if section_foils is not None else 1,
        "spanwise_airfoil_names": (
            [section_foil.name for section_foil in section_foils]
            if section_foils is not None
            else [foil.name]
        ),
        "structural_check": optimum.structural
        or {"enabled": False, "performed": False, "passed": True},
        "hydro_check": optimum.hydro
        or {"enabled": False, "performed": False, "passed": True},
        "pareto_analysis": pareto_analysis,
        "selection_comparison": selection_comparison,
        "solver": optimum.response.get("solver", {}) if optimum.response else {},
        "top_search_candidates": [
            {
                "rank": rank + 1,
                "score": float(item.score),
                "geometry": item.geometry.to_dict(),
            }
            for rank, item in enumerate(sorted(valid_search, key=lambda item: item.score)[:20])
        ],
    }
    output_response = dict(optimum.response or {})
    if scalar_optimum is not None and not same_scalar_geometry:
        output_response["_scalar_only_alternative_response"] = dict(
            scalar_optimum.response or {}
        )
    return optimum_result, baseline_result, metadata, output_response
