from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
from typing import Any
import zipfile

import numpy as np

from .exporters import project_json


def _all_seed_failure_message(records: list[dict[str, Any]]) -> str:
    """Keep the optimizer's real per-seed errors when every run fails."""
    failures: list[tuple[Any, str]] = []
    for item in records:
        diagnosis = item.get("failure_diagnosis")
        if not isinstance(diagnosis, dict):
            diagnosis = {}
        raw_error = (
            item.get("error")
            or diagnosis.get("evidence")
            or diagnosis.get("title")
            or "Bilinmeyen optimizer hatası"
        )
        error = " ".join(str(raw_error).split()).strip()
        failures.append((item.get("seed", "?"), error or "Bilinmeyen optimizer hatası"))

    if not failures:
        return "Hiçbir seed optimizasyonu tamamlanamadı; seed hata kaydı oluşmadı."

    unique_errors = list(dict.fromkeys(error for _, error in failures))
    if len(unique_errors) == 1:
        detail = (
            f"Ortak kök hata ({len(failures)}/{len(failures)} seed): "
            f"{unique_errors[0]}"
        )
    else:
        detail = "Seed hataları: " + "; ".join(
            f"seed {seed}: {error}" for seed, error in failures
        )
    return f"Hiçbir seed optimizasyonu tamamlanamadı. {detail}"[:4000]


def build_multi_seed_report(
    records: list[dict[str, Any]],
    *,
    objective_cv_tolerance_percent: float = 3.0,
    geometry_cv_tolerance_percent: float = 2.0,
    geometry_reference_ranges: dict[str, float] | None = None,
) -> dict[str, Any]:
    successful = [
        (index, item) for index, item in enumerate(records) if item.get("result") is not None
    ]
    if not successful:
        raise RuntimeError(_all_seed_failure_message(records))
    selected_index, selected = min(
        successful,
        key=lambda indexed: (
            not bool(indexed[1]["result"]["wing_optimization"].get("feasible", False)),
            float(indexed[1]["result"]["wing_optimization"]["objective"]),
        ),
    )
    objectives = np.asarray(
        [
            float(item["result"]["wing_optimization"]["objective"])
            for _, item in successful
        ]
    )
    objective_cv = float(100.0 * np.std(objectives) / max(abs(float(np.mean(objectives))), 1e-12))
    geometry_keys = (
        "span",
        "root_chord",
        "taper",
        "sweep_deg",
        "tip_twist_deg",
        "winglet_height",
        "winglet_cant_deg",
        "winglet_toe_deg",
        "winglet_taper",
    )
    geometry_cv: dict[str, float] = {}
    geometry_scales: dict[str, float] = {}
    for key in geometry_keys:
        values = np.asarray(
            [float(item["result"]["wing"]["geometry"][key]) for _, item in successful]
        )
        configured_range = float((geometry_reference_ranges or {}).get(key, 0.0))
        scale = max(
            abs(float(np.mean(values))), configured_range, float(np.ptp(values)), 1e-9
        )
        geometry_scales[key] = scale
        geometry_cv[key] = float(100.0 * np.std(values) / scale)
    enabled = len(records) > 1
    stable = None if not enabled else bool(
        len(successful) == len(records)
        and objective_cv <= objective_cv_tolerance_percent
        and max(geometry_cv.values(), default=0.0) <= geometry_cv_tolerance_percent
    )
    summaries = []
    for item in records:
        result = item.get("result")
        if result is None:
            summaries.append(
                {
                    "seed": item["seed"],
                    "completed": False,
                    "error": item.get("error", ""),
                    "diagnosis": item.get("failure_diagnosis", {}),
                }
            )
            continue
        geometry = result["wing"]["geometry"]
        summaries.append(
            {
                "seed": item["seed"],
                "completed": True,
                "selected": item is selected,
                "feasible": bool(result["wing_optimization"].get("feasible", False)),
                "objective": float(result["wing_optimization"]["objective"]),
                "ld": float(result["wing"]["ld"]),
                "drag_n": float(result["wing"]["drag_n"]),
                "airfoil_name": result["airfoil"]["name"],
                "geometry": {key: float(geometry[key]) for key in geometry_keys},
                "validation_passed": bool(result.get("validation_report", {}).get("passed", True)),
                "cache": result.get("solver_run", {}).get("evaluation_cache", {}),
            }
        )
    return {
        "enabled": enabled,
        "runs_requested": len(records),
        "runs_completed": len(successful),
        "selected_seed": selected["seed"],
        "selected_record_index": selected_index,
        "stable": stable,
        "status": "single_run" if not enabled else "stable" if stable else "review",
        "objective_mean": float(np.mean(objectives)),
        "objective_std": float(np.std(objectives)),
        "objective_cv_percent": objective_cv,
        "objective_cv_tolerance_percent": float(objective_cv_tolerance_percent),
        "geometry_cv_percent": geometry_cv,
        "geometry_normalization_scale": geometry_scales,
        "geometry_cv_tolerance_percent": float(geometry_cv_tolerance_percent),
        "runs": summaries,
    }


def refresh_flow5_exports(result: dict[str, Any], request: dict[str, Any]) -> None:
    """Refresh project/report JSON files after outer reliability analysis."""
    exports = result.get("exports")
    if not isinstance(exports, dict):
        return
    snapshot = deepcopy(result)
    snapshot.pop("exports", None)
    project_text = project_json(request, snapshot)
    reports = {
        "aeropt-project.json": project_text,
        "aeropt-validation.json": json.dumps(
            result.get("validation_report", {}), ensure_ascii=False, indent=2, allow_nan=False
        ),
        "aeropt-multi-seed.json": json.dumps(
            result.get("multi_seed_stability", {}), ensure_ascii=False, indent=2, allow_nan=False
        ),
        "aeropt-pareto.json": json.dumps(
            result.get("pareto_analysis", {}), ensure_ascii=False, indent=2, allow_nan=False
        ),
        "aeropt-diagnostics.json": json.dumps(
            result.get("diagnostic_report", {}), ensure_ascii=False, indent=2, allow_nan=False
        ),
    }
    hydro = result.get("hydro_analysis", {})
    if isinstance(hydro, dict) and hydro.get("enabled"):
        reports["aeropt-cavitation.json"] = json.dumps(
            hydro, ensure_ascii=False, indent=2, allow_nan=False
        )
    exports["project_json"] = project_text
    exports["validation_filename"] = "aeropt-validation.json"
    exports["validation_json"] = reports["aeropt-validation.json"]
    exports["multi_seed_filename"] = "aeropt-multi-seed.json"
    exports["multi_seed_json"] = reports["aeropt-multi-seed.json"]
    exports["pareto_filename"] = "aeropt-pareto.json"
    exports["pareto_json"] = reports["aeropt-pareto.json"]
    exports["diagnostics_filename"] = "aeropt-diagnostics.json"
    exports["diagnostics_json"] = reports["aeropt-diagnostics.json"]
    if "aeropt-cavitation.json" in reports:
        exports["cavitation_filename"] = "aeropt-cavitation.json"
        exports["cavitation_json"] = reports["aeropt-cavitation.json"]
    else:
        exports.pop("cavitation_filename", None)
        exports.pop("cavitation_json", None)
    encoded = exports.get("flow5_bundle_base64")
    if not encoded:
        return
    old_bytes = base64.b64decode(encoded)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_bytes), "r") as source, zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for info in source.infolist():
            if info.filename not in reports:
                target.writestr(info, source.read(info.filename))
        for filename, text in reports.items():
            target.writestr(filename, text)
    exports["flow5_bundle_base64"] = base64.b64encode(output.getvalue()).decode("ascii")
