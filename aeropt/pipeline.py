from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
from math import ceil
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np

from .airfoil import generate_polar, naca4_coordinates, naca4_design
from .baselines import build_baseline_profile
from .convergence import BudgetEscalationSettings
from .exporters import (
    airfoil_dat,
    flow5_bundle_bytes,
    flow5_plane_xml,
    project_json,
    results_csv,
    wing_obj,
    xfoil_polar_csv,
)
from .hybrid import build_xfoil_polar_mesh, run_closed_loop_airfoil
from .diagnostics import build_diagnostic_report, diagnose_runtime_failure
from .hydro import HydroSettings
from .flow5_pipeline import Flow5NativeSettings, run_flow5_native_design
from .flow5 import Flow5CancelledError, Flow5Mesh, resolve_flow5_runner_path
from .models import FLUID_PRESETS, Fluid
from .optimization import EFFORTS, optimize_airfoil, optimize_wing
from .structures import StructuralSettings
from .surrogate import SurrogateSettings
from .validation import ValidationSettings, build_validation_report
from .reliability import build_multi_seed_report, refresh_flow5_exports


DEFAULT_REQUEST: dict[str, Any] = {
    "workflow": {
        "mode": "coupled",
    },
    "flow": {
        "fluid": "air",
        "density_kg_m3": 1.225,
        "dynamic_viscosity_pa_s": 1.7894e-5,
        "speed_of_sound_m_s": 340.3,
        "speed_m_s": 18.0,
        "speed_min_m_s": 14.0,
        "speed_max_m_s": 22.0,
        "speed_samples": 5,
        "target_lift_n": 120.0,
    },
    "airfoil": {
        "baseline_profile": "e818",
        "baseline_dat": "",
        "cst_order": 6,
        "solver_coordinate_points": 100,
        "camber_min_percent": 0.0,
        "camber_max_percent": 6.0,
        "camber_position_min_percent": 20.0,
        "camber_position_max_percent": 80.0,
        "thickness_min_percent": 7.0,
        "thickness_max_percent": 18.0,
        "design_cl": None,
    },
    "wing": {
        "span_min_m": 1.0,
        "span_max_m": 3.0,
        "root_chord_min_m": 0.12,
        "root_chord_max_m": 0.50,
        "taper_min": 0.25,
        "taper_max": 1.0,
        "sweep_min_deg": 0.0,
        "sweep_max_deg": 20.0,
        "tip_twist_min_deg": -5.0,
        "tip_twist_max_deg": 0.0,
        "alpha_min_deg": -2.0,
        "alpha_max_deg": 12.0,
        "multi_section_geometry_enabled": True,
        "mid_chord_factor_min": 0.85,
        "mid_chord_factor_max": 1.15,
        "mid_twist_min_deg": -4.0,
        "mid_twist_max_deg": 1.0,
        "winglet_optimization_enabled": False,
        "winglet_naca_code": "0012",
        "winglet_height_min_m": 0.05,
        "winglet_height_max_m": 0.40,
        "winglet_cant_min_deg": 60.0,
        "winglet_cant_max_deg": 90.0,
        "winglet_toe_min_deg": -3.0,
        "winglet_toe_max_deg": 3.0,
        "winglet_taper_min": 0.30,
        "winglet_taper_max": 1.0,
    },
    "solver": {
        "quality": "balanced",
        "seed": 42,
        "lifting_line_modes": 10,
        "airfoil_strategy": "flow5_native",
        "flow5_runner_path": "",
        "flow5_threads": 16,
        "flow5_timeout_seconds": 900.0,
        "flow5_foil_candidate_budget": 48,
        "flow5_wing_candidate_budget": 48,
        "flow5_winglet_candidate_budget": 48,
        "flow5_finalists": 3,
        "flow5_search_method": "VLM2",
        "flow5_final_method": "TRIUNIFORM",
        "flow5_alpha_step_search_deg": 2.0,
        "flow5_alpha_step_final_deg": 0.5,
        "flow5_ncrit": 9.0,
        "flow5_xtr_top": 1.0,
        "flow5_xtr_bottom": 1.0,
        "flow5_foil_minimum_improvement_percent": 1.0,
        "flow5_search_chordwise_panels": 10,
        "flow5_search_half_span_panels": 14,
        "flow5_final_chordwise_panels": 14,
        "flow5_final_half_span_panels": 22,
        "flow5_convergence_chordwise_panels": 20,
        "flow5_convergence_half_span_panels": 32,
        "flow5_mesh_convergence_enabled": True,
        "flow5_mesh_cd_tolerance_percent": 2.0,
        "flow5_mesh_alpha_tolerance_deg": 0.25,
        "flow5_cache_enabled": True,
        "flow5_cache_dir": "",
        "flow5_foil_optimizer": "differential_evolution",
        "flow5_wing_optimizer": "nsga2",
        "flow5_coupled_iterations": 2,
        "flow5_coupling_cl_tolerance_percent": 3.0,
        "flow5_coupling_objective_tolerance_percent": 1.0,
        "flow5_spanwise_airfoil_optimization_enabled": False,
        "flow5_spanwise_foil_budget_fraction": 0.5,
        "flow5_spanwise_foil_acceptance_tolerance_percent": 2.0,
        "flow5_surrogate_enabled": True,
        "flow5_surrogate_proposals_per_evaluation": 6,
        "flow5_surrogate_minimum_real_fraction": 0.65,
        "flow5_surrogate_maximum_error_percent": 8.0,
        "flow5_surrogate_early_stop_improvement_percent": 0.25,
        "flow5_budget_escalation_enabled": True,
        "flow5_budget_growth_factor": 2.0,
        "flow5_budget_maximum_multiplier": 4.0,
        "flow5_budget_convergence_tolerance_percent": 3.0,
        "flow5_budget_stable_checkpoints": 1,
        "flow5_checkpoint_enabled": True,
        "flow5_checkpoint_dir": "",
        "flow5_multi_seed_runs": 1,
        "flow5_multi_seed_stride": 100003,
        "flow5_multi_seed_objective_cv_tolerance_percent": 3.0,
        "flow5_multi_seed_geometry_cv_tolerance_percent": 2.0,
        "xfoil_path": "",
        "xfoil_cl_tolerance_percent": 5.0,
        "xfoil_cd_tolerance_percent": 15.0,
        "cst_candidate_budget": 128,
        "parallel_workers": 16,
        "xfoil_timeout_seconds": 45.0,
    },
    "structure": {
        "enabled": False,
        "youngs_modulus_gpa": 70.0,
        "material_density_kg_m3": 1600.0,
        "allowable_stress_mpa": 300.0,
        "safety_factor": 1.5,
        "spar_height_fraction_of_foil": 0.75,
        "spar_cap_width_fraction_chord": 0.08,
        "spar_cap_thickness_mm": 2.0,
        "skin_thickness_mm": 1.0,
        "torsion_box_width_fraction_chord": 0.45,
        "poisson_ratio": 0.30,
        "max_tip_deflection_percent_semispan": 8.0,
        "max_elastic_twist_deg": 2.0,
    },
    "hydro": {
        "enabled": True,
        "constraint_mode": "hard",
        "submergence_depth_m": 1.0,
        "ambient_pressure_pa": 101325.0,
        "vapor_pressure_pa": 1705.0,
        "cavitation_safety_factor": 1.20,
        "minimum_submergence_chords": 2.0,
        "free_surface_screen_enabled": True,
    },
    "validation": {
        "enabled": True,
        "force_closure_tolerance_percent": 1.0,
        "drag_decomposition_tolerance_percent": 12.0,
        "minimum_span_efficiency": 0.20,
        "maximum_span_efficiency": 1.20,
    },
}


class InputError(ValueError):
    pass


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise InputError(f"'{key}' bölümü bir nesne olmalı")
    return value


def _number(data: dict[str, Any], key: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError):
        raise InputError(f"'{key}' geçerli bir sayı olmalı") from None
    if not np.isfinite(value):
        raise InputError(f"'{key}' sonlu bir sayı olmalı")
    if minimum is not None and value < minimum:
        raise InputError(f"'{key}' en az {minimum:g} olmalı")
    if maximum is not None and value > maximum:
        raise InputError(f"'{key}' en çok {maximum:g} olmalı")
    return value


def _bounds(data: dict[str, Any], low_key: str, high_key: str, **kwargs: Any) -> tuple[float, float]:
    low = _number(data, low_key, **kwargs)
    high = _number(data, high_key, **kwargs)
    if high < low:
        raise InputError(f"'{high_key}', '{low_key}' değerinden küçük olamaz")
    return low, high


def _merge_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_REQUEST)
    for section, values in payload.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    # Deprecated project files can still contain this field.  Root bending
    # moment is telemetry only and must never re-enter design selection.
    if isinstance(merged.get("wing"), dict):
        merged["wing"].pop("max_root_bending_moment_nm", None)
    # A direct JSON caller may select a preset without echoing the three
    # editable property fields that the browser always submits.  Do not leave
    # the air defaults attached to a water preset in that case; explicit
    # property overrides still win one by one.
    flow_payload = payload.get("flow")
    if isinstance(flow_payload, dict):
        fluid_key = str(flow_payload.get("fluid", merged["flow"].get("fluid", "air")))
        preset = FLUID_PRESETS.get(fluid_key)
        if preset is not None:
            preset_values = {
                "density_kg_m3": preset.density,
                "dynamic_viscosity_pa_s": preset.dynamic_viscosity,
                "speed_of_sound_m_s": preset.speed_of_sound,
            }
            for key, value in preset_values.items():
                if key not in flow_payload:
                    merged["flow"][key] = value
    return merged


def _boolean(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", "off"}:
        return False
    if value in {0, 1}:
        return bool(value)
    raise InputError(f"'{key}' doğru/yanlış olmalı")


def _fluid_from_input(flow: dict[str, Any]) -> Fluid:
    fluid_key = str(flow.get("fluid", "custom"))
    if fluid_key in FLUID_PRESETS:
        preset = FLUID_PRESETS[fluid_key]
        # The UI writes preset properties into the fields, but all three remain editable.
        name = preset.name
    else:
        name = "Özel akışkan"
    return Fluid(
        name=name,
        density=_number(flow, "density_kg_m3", minimum=1e-6),
        dynamic_viscosity=_number(flow, "dynamic_viscosity_pa_s", minimum=1e-10),
        speed_of_sound=_number(flow, "speed_of_sound_m_s", minimum=1.0),
    )


def _insights(result: dict[str, Any], fluid_key: str, bounds: dict[str, tuple[float, float]]) -> list[dict[str, str]]:
    wing = result["wing"]["geometry"]
    foil = result["airfoil"]
    meta = result["wing_optimization"]
    messages: list[dict[str, str]] = []
    if wing["sweep_deg"] < 1.0:
        messages.append(
            {
                "level": "info",
                "title": "Sweep gerekli görünmüyor",
                "text": "Bu düşük-Mach tasarım noktasında sweep kaldırmayı azaltıp ek profil sürüklemesi getiriyor; optimum neredeyse düz ön kenar seçti.",
            }
        )
    else:
        messages.append(
            {
                "level": "info",
                "title": "Sınırlı sweep seçildi",
                "text": f"Optimum çeyrek-kord sweep açısı {wing['sweep_deg']:.2f}°. Bu model sweep'i paketleme veya stabilite için ödüllendirmez; seçimin kazancı mutlaka yüksek doğruluklı çözücüyle doğrulanmalı.",
            }
        )
    if wing["taper"] < 0.88:
        messages.append(
            {
                "level": "good",
                "title": "Taper faydalı",
                "text": f"Taper oranı {wing['taper']:.3f}; yük dağılımını eliptik dağılıma yaklaştırarak indüklenmiş sürüklemeyi düşürüyor.",
            }
        )
    if wing["tip_twist_deg"] < -0.25:
        messages.append(
            {
                "level": "good",
                "title": "Washout kullanıldı",
                "text": f"Uçta {wing['tip_twist_deg']:.2f}° twist, uç yükünü ve erken tip stall riskini azaltıyor.",
            }
        )
    if abs(wing["span"] - bounds["span"][1]) <= 0.01 * max(bounds["span"][1], 1e-9):
        messages.append(
            {
                "level": "warn",
                "title": "Açıklık üst sınıra dayandı",
                "text": "Sadece aerodinamik sürükleme hedeflendiğinde daha yüksek açıklık oranı avantajlıdır. Yapısal kütle/rijitlik etkisi için kök eğilme momenti sınırı girin.",
            }
        )
    if abs(foil["thickness"] - bounds["thickness"][0]) <= 0.002:
        messages.append(
            {
                "level": "warn",
                "title": "Profil kalınlığı alt sınıra yakın",
                "text": "Aerodinamik amaç fonksiyonu ince profili tercih etti. Spar yüksekliği, dayanım ve üretim payı gerektiriyorsa minimum t/c değerini yükseltin.",
            }
        )
    if fluid_key in {"fresh_water", "sea_water"}:
        messages.append(
            {
                "level": "warn",
                "title": "Hidrofoil doğrulaması gerekli",
                "text": "Dahili model serbest yüzey, kavitasyon, havalanma ve dalga sürüklemesini içermez. Nihai hidrofoil tasarımını flow5/CFD ve kavitasyon kontrolüyle doğrulayın.",
            }
        )
    if not meta["feasible"]:
        messages.append(
            {
                "level": "bad",
                "title": "Kısıtlar altında tam fizibil değil",
                "text": "Hedef taşıma, stall marjı veya hücum açısı koşullarından en az biri sağlanamadı. Boyut sınırlarını genişletin ya da hızı artırın.",
            }
        )
    return messages


def run_design(
    payload: dict[str, Any],
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise InputError("İstek bir JSON nesnesi olmalı")
    request = _merge_defaults(payload)
    workflow_cfg = _section(request, "workflow")
    flow = _section(request, "flow")
    airfoil_cfg = _section(request, "airfoil")
    wing_cfg = _section(request, "wing")
    solver_cfg = _section(request, "solver")
    structure_cfg = _section(request, "structure")
    hydro_cfg = _section(request, "hydro")
    validation_cfg = _section(request, "validation")

    fluid = _fluid_from_input(flow)
    fluid_key = str(flow.get("fluid", "custom"))
    speed = _number(flow, "speed_m_s", minimum=0.05)
    speed_bounds = _bounds(flow, "speed_min_m_s", "speed_max_m_s", minimum=0.05)
    speed_samples_value = _number(flow, "speed_samples", minimum=1.0, maximum=9.0)
    if not float(speed_samples_value).is_integer():
        raise InputError("'speed_samples' tam sayı olmalı")
    speed_samples = int(speed_samples_value)
    target_lift = _number(flow, "target_lift_n", minimum=0.01)
    if fluid.mach(max(speed, speed_bounds[1])) >= 0.70:
        raise InputError("Bu ön tasarım zinciri Mach 0.70 ve üzeri için uygun değil")

    camber_pct = _bounds(
        airfoil_cfg, "camber_min_percent", "camber_max_percent", minimum=0.0, maximum=9.0
    )
    camber_position_pct = _bounds(
        airfoil_cfg,
        "camber_position_min_percent",
        "camber_position_max_percent",
        minimum=10.0,
        maximum=85.0,
    )
    thickness_pct = _bounds(
        airfoil_cfg,
        "thickness_min_percent",
        "thickness_max_percent",
        minimum=5.0,
        maximum=24.0,
    )
    camber_bounds = (camber_pct[0] / 100.0, camber_pct[1] / 100.0)
    camber_position_bounds = (
        camber_position_pct[0] / 100.0,
        camber_position_pct[1] / 100.0,
    )
    thickness_bounds = (thickness_pct[0] / 100.0, thickness_pct[1] / 100.0)
    span_bounds = _bounds(wing_cfg, "span_min_m", "span_max_m", minimum=0.02)
    root_chord_bounds = _bounds(
        wing_cfg, "root_chord_min_m", "root_chord_max_m", minimum=0.005
    )
    taper_bounds = _bounds(
        wing_cfg, "taper_min", "taper_max", minimum=0.08, maximum=1.2
    )
    sweep_bounds = _bounds(
        wing_cfg, "sweep_min_deg", "sweep_max_deg", minimum=-15.0, maximum=55.0
    )
    twist_bounds = _bounds(
        wing_cfg,
        "tip_twist_min_deg",
        "tip_twist_max_deg",
        minimum=-12.0,
        maximum=8.0,
    )
    alpha_bounds = _bounds(
        wing_cfg, "alpha_min_deg", "alpha_max_deg", minimum=-15.0, maximum=25.0
    )
    structural_settings = StructuralSettings(
        enabled=_boolean(structure_cfg, "enabled"),
        youngs_modulus_pa=1.0e9
        * _number(structure_cfg, "youngs_modulus_gpa", minimum=0.01, maximum=1000.0),
        material_density_kg_m3=_number(
            structure_cfg, "material_density_kg_m3", minimum=1.0, maximum=30000.0
        ),
        allowable_stress_pa=1.0e6
        * _number(
            structure_cfg, "allowable_stress_mpa", minimum=0.01, maximum=10000.0
        ),
        safety_factor=_number(structure_cfg, "safety_factor", minimum=1.0, maximum=10.0),
        spar_height_fraction_of_foil=_number(
            structure_cfg,
            "spar_height_fraction_of_foil",
            minimum=0.1,
            maximum=1.0,
        ),
        spar_cap_width_fraction_chord=_number(
            structure_cfg,
            "spar_cap_width_fraction_chord",
            minimum=0.005,
            maximum=0.5,
        ),
        spar_cap_thickness_m=0.001
        * _number(
            structure_cfg, "spar_cap_thickness_mm", minimum=0.05, maximum=100.0
        ),
        skin_thickness_m=0.001
        * _number(structure_cfg, "skin_thickness_mm", minimum=0.05, maximum=100.0),
        torsion_box_width_fraction_chord=_number(
            structure_cfg,
            "torsion_box_width_fraction_chord",
            minimum=0.05,
            maximum=0.9,
        ),
        poisson_ratio=_number(structure_cfg, "poisson_ratio", minimum=0.0, maximum=0.49),
        max_tip_deflection_fraction_semispan=0.01
        * _number(
            structure_cfg,
            "max_tip_deflection_percent_semispan",
            minimum=0.01,
            maximum=100.0,
        ),
        max_elastic_twist_deg=_number(
            structure_cfg, "max_elastic_twist_deg", minimum=0.01, maximum=45.0
        ),
    )
    hydro_requested = _boolean(hydro_cfg, "enabled")
    hydro_constraint_mode = str(
        hydro_cfg.get("constraint_mode", "hard")
    ).strip().lower()
    if hydro_constraint_mode not in {"hard", "report_only"}:
        raise InputError(
            "'constraint_mode' yalnız 'hard' veya 'report_only' olabilir"
        )
    hydro_settings = HydroSettings(
        enabled=bool(hydro_requested and fluid_key in {"fresh_water", "sea_water"}),
        constraint_mode=hydro_constraint_mode,
        submergence_depth_m=_number(
            hydro_cfg, "submergence_depth_m", minimum=0.001, maximum=10000.0
        ),
        ambient_pressure_pa=_number(
            hydro_cfg, "ambient_pressure_pa", minimum=100.0, maximum=10.0e7
        ),
        vapor_pressure_pa=_number(
            hydro_cfg, "vapor_pressure_pa", minimum=0.0, maximum=1.0e6
        ),
        cavitation_safety_factor=_number(
            hydro_cfg, "cavitation_safety_factor", minimum=1.0, maximum=10.0
        ),
        minimum_submergence_chords=_number(
            hydro_cfg, "minimum_submergence_chords", minimum=0.1, maximum=100.0
        ),
        free_surface_screen_enabled=_boolean(
            hydro_cfg, "free_surface_screen_enabled"
        ),
    )
    multi_section_geometry_enabled = _boolean(
        wing_cfg, "multi_section_geometry_enabled"
    )
    mid_chord_factor_bounds = _bounds(
        wing_cfg,
        "mid_chord_factor_min",
        "mid_chord_factor_max",
        minimum=0.5,
        maximum=1.5,
    )
    mid_twist_bounds = _bounds(
        wing_cfg,
        "mid_twist_min_deg",
        "mid_twist_max_deg",
        minimum=-12.0,
        maximum=8.0,
    )
    winglet_optimization_enabled = _boolean(
        wing_cfg, "winglet_optimization_enabled"
    )
    raw_winglet_naca_code = str(wing_cfg.get("winglet_naca_code", "0012"))
    try:
        winglet_naca_code = naca4_design(
            raw_winglet_naca_code if winglet_optimization_enabled else "0012"
        ).name.removeprefix("NACA")
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    winglet_height_bounds = _bounds(
        wing_cfg,
        "winglet_height_min_m",
        "winglet_height_max_m",
        minimum=0.005,
        maximum=10.0,
    )
    winglet_cant_bounds = _bounds(
        wing_cfg,
        "winglet_cant_min_deg",
        "winglet_cant_max_deg",
        minimum=45.0,
        maximum=90.0,
    )
    winglet_toe_bounds = _bounds(
        wing_cfg,
        "winglet_toe_min_deg",
        "winglet_toe_max_deg",
        minimum=-10.0,
        maximum=10.0,
    )
    winglet_taper_bounds = _bounds(
        wing_cfg,
        "winglet_taper_min",
        "winglet_taper_max",
        minimum=0.10,
        maximum=1.20,
    )

    quality = str(solver_cfg.get("quality", "balanced"))
    if quality not in EFFORTS:
        raise InputError("'quality' quick, balanced veya thorough olmalı")
    seed = int(_number(solver_cfg, "seed", minimum=0.0, maximum=2_147_483_647.0))
    modes = int(_number(solver_cfg, "lifting_line_modes", minimum=6.0, maximum=24.0))
    workers = int(_number(solver_cfg, "parallel_workers", minimum=1.0, maximum=64.0))
    candidate_budget = int(
        _number(solver_cfg, "cst_candidate_budget", minimum=16.0, maximum=4096.0)
    )
    timeout_seconds = _number(
        solver_cfg, "xfoil_timeout_seconds", minimum=5.0, maximum=300.0
    )
    cl_tolerance = _number(
        solver_cfg, "xfoil_cl_tolerance_percent", minimum=0.1, maximum=100.0
    )
    cd_tolerance = _number(
        solver_cfg, "xfoil_cd_tolerance_percent", minimum=0.1, maximum=200.0
    )
    requested_strategy = str(solver_cfg.get("airfoil_strategy", "auto"))
    workflow_mode = str(workflow_cfg.get("mode", "coupled")).strip().lower()
    if workflow_mode not in {"coupled", "foil_only", "wing_only"}:
        raise InputError("Çalışma modu coupled, foil_only veya wing_only olmalı")
    allowed_strategies = {
        "flow5_native",
        "auto",
        "internal",
        "xfoil_closed_loop",
        "xfoil_cst_always",
    }
    if requested_strategy not in allowed_strategies:
        raise InputError("Geçersiz airfoil çözüm stratejisi")
    xfoil_path = str(solver_cfg.get("xfoil_path", "")).strip()
    strategy = (
        "xfoil_closed_loop" if requested_strategy == "auto" and xfoil_path else requested_strategy
    )
    if strategy == "auto":
        strategy = "internal"
    if strategy.startswith("xfoil_") and not xfoil_path:
        raise InputError("XFOIL kapalı döngüsü için XFOIL çalıştırılabilir dosya yolunu girin")
    if strategy.startswith("xfoil_") and not Path(xfoil_path).expanduser().is_file():
        raise InputError(f"XFOIL çalıştırılabilir dosyası bulunamadı: {xfoil_path}")
    if workflow_mode != "coupled" and strategy != "flow5_native":
        raise InputError("Ayrık profil/kanat çalışma modları yalnız flow5-native zincirinde kullanılabilir")

    flow5_runner_path = str(solver_cfg.get("flow5_runner_path", "")).strip()
    flow5_threads = int(_number(solver_cfg, "flow5_threads", minimum=1.0, maximum=64.0))
    flow5_timeout = _number(
        solver_cfg, "flow5_timeout_seconds", minimum=30.0
    )
    flow5_foil_budget = int(
        _number(solver_cfg, "flow5_foil_candidate_budget", minimum=8.0, maximum=4096.0)
    )
    flow5_wing_budget = int(
        _number(solver_cfg, "flow5_wing_candidate_budget", minimum=8.0, maximum=2048.0)
    )
    flow5_winglet_budget = int(
        _number(
            solver_cfg,
            "flow5_winglet_candidate_budget",
            minimum=8.0,
            maximum=2048.0,
        )
    )
    flow5_finalists = int(
        _number(solver_cfg, "flow5_finalists", minimum=1.0, maximum=8.0)
    )
    flow5_search_method = str(solver_cfg.get("flow5_search_method", "VLM2")).upper()
    flow5_final_method = str(solver_cfg.get("flow5_final_method", "TRIUNIFORM")).upper()
    flow5_methods = {"LLT", "VLM1", "VLM2", "QUADS", "TRIUNIFORM", "TRILINEAR"}
    if flow5_search_method not in flow5_methods or flow5_final_method not in flow5_methods:
        raise InputError("Geçersiz flow5 analiz yöntemi")
    flow5_alpha_step_search = _number(
        solver_cfg, "flow5_alpha_step_search_deg", minimum=0.25, maximum=5.0
    )
    flow5_alpha_step_final = _number(
        solver_cfg, "flow5_alpha_step_final_deg", minimum=0.1, maximum=2.0
    )
    flow5_ncrit = _number(solver_cfg, "flow5_ncrit", minimum=1.0, maximum=20.0)
    flow5_xtr_top = _number(solver_cfg, "flow5_xtr_top", minimum=0.01, maximum=1.0)
    flow5_xtr_bottom = _number(
        solver_cfg, "flow5_xtr_bottom", minimum=0.01, maximum=1.0
    )
    flow5_minimum_improvement = _number(
        solver_cfg,
        "flow5_foil_minimum_improvement_percent",
        minimum=0.0,
        maximum=20.0,
    )
    flow5_search_mesh = Flow5Mesh(
        int(_number(solver_cfg, "flow5_search_chordwise_panels", minimum=4, maximum=200)),
        int(_number(solver_cfg, "flow5_search_half_span_panels", minimum=4, maximum=400)),
    )
    flow5_final_mesh = Flow5Mesh(
        int(_number(solver_cfg, "flow5_final_chordwise_panels", minimum=4, maximum=200)),
        int(_number(solver_cfg, "flow5_final_half_span_panels", minimum=4, maximum=400)),
    )
    flow5_convergence_mesh = Flow5Mesh(
        int(
            _number(
                solver_cfg,
                "flow5_convergence_chordwise_panels",
                minimum=4,
                maximum=200,
            )
        ),
        int(
            _number(
                solver_cfg,
                "flow5_convergence_half_span_panels",
                minimum=4,
                maximum=400,
            )
        ),
    )
    if flow5_final_mesh.nominal_panels < flow5_search_mesh.nominal_panels:
        raise InputError("Final flow5 ağı, arama ağından daha kaba olamaz")
    if flow5_convergence_mesh.nominal_panels <= flow5_final_mesh.nominal_panels:
        raise InputError("Yakınsama flow5 ağı, final ağından daha ince olmalı")
    if winglet_optimization_enabled and min(
        flow5_search_mesh.half_span_panels,
        flow5_final_mesh.half_span_panels,
        flow5_convergence_mesh.half_span_panels,
    ) < 6:
        raise InputError("Winglet çözümü için yarı açıklık panel sayıları en az 6 olmalı")
    flow5_mesh_convergence_enabled = _boolean(
        solver_cfg, "flow5_mesh_convergence_enabled"
    )
    flow5_mesh_cd_tolerance = _number(
        solver_cfg, "flow5_mesh_cd_tolerance_percent", minimum=0.05, maximum=25.0
    )
    flow5_mesh_alpha_tolerance = _number(
        solver_cfg, "flow5_mesh_alpha_tolerance_deg", minimum=0.01, maximum=5.0
    )
    flow5_cache_enabled = _boolean(solver_cfg, "flow5_cache_enabled")
    flow5_cache_dir_value = solver_cfg.get("flow5_cache_dir", "")
    if not isinstance(flow5_cache_dir_value, str):
        raise InputError("'flow5_cache_dir' metin yol olmalı")
    flow5_cache_dir = flow5_cache_dir_value.strip()
    flow5_foil_optimizer = str(
        solver_cfg.get("flow5_foil_optimizer", "differential_evolution")
    ).strip().lower()
    flow5_wing_optimizer = str(
        solver_cfg.get("flow5_wing_optimizer", "nsga2")
    ).strip().lower()
    allowed_foil_optimizers = {"differential_evolution", "adaptive_elite"}
    allowed_wing_optimizers = {"nsga2", *allowed_foil_optimizers}
    if flow5_foil_optimizer not in allowed_foil_optimizers:
        raise InputError("Foil optimizeri differential_evolution veya adaptive_elite olmalı")
    if flow5_wing_optimizer not in allowed_wing_optimizers:
        raise InputError(
            "Kanat optimizeri nsga2, differential_evolution veya adaptive_elite olmalı"
        )
    flow5_coupled_iterations_value = _number(
        solver_cfg, "flow5_coupled_iterations", minimum=1.0, maximum=8.0
    )
    if not float(flow5_coupled_iterations_value).is_integer():
        raise InputError("'flow5_coupled_iterations' tam sayı olmalı")
    flow5_coupled_iterations = int(flow5_coupled_iterations_value)
    flow5_coupling_cl_tolerance = _number(
        solver_cfg,
        "flow5_coupling_cl_tolerance_percent",
        minimum=0.1,
        maximum=25.0,
    )
    flow5_coupling_objective_tolerance = _number(
        solver_cfg,
        "flow5_coupling_objective_tolerance_percent",
        minimum=0.05,
        maximum=25.0,
    )
    flow5_spanwise_airfoil_optimization_enabled = _boolean(
        solver_cfg, "flow5_spanwise_airfoil_optimization_enabled"
    )
    flow5_spanwise_foil_budget_fraction = _number(
        solver_cfg,
        "flow5_spanwise_foil_budget_fraction",
        minimum=0.2,
        maximum=1.0,
    )
    flow5_spanwise_foil_acceptance_tolerance = _number(
        solver_cfg,
        "flow5_spanwise_foil_acceptance_tolerance_percent",
        minimum=0.0,
        maximum=20.0,
    )
    flow5_surrogate_proposals_value = _number(
        solver_cfg,
        "flow5_surrogate_proposals_per_evaluation",
        minimum=2.0,
        maximum=16.0,
    )
    if not float(flow5_surrogate_proposals_value).is_integer():
        raise InputError("'flow5_surrogate_proposals_per_evaluation' tam sayı olmalı")
    surrogate_settings = SurrogateSettings(
        enabled=_boolean(solver_cfg, "flow5_surrogate_enabled"),
        proposals_per_real_evaluation=int(flow5_surrogate_proposals_value),
        minimum_real_fraction=_number(
            solver_cfg,
            "flow5_surrogate_minimum_real_fraction",
            minimum=0.5,
            maximum=1.0,
        ),
        maximum_validation_error_percent=_number(
            solver_cfg,
            "flow5_surrogate_maximum_error_percent",
            minimum=1.0,
            maximum=50.0,
        ),
        early_stop_improvement_percent=_number(
            solver_cfg,
            "flow5_surrogate_early_stop_improvement_percent",
            minimum=0.0,
            maximum=10.0,
        ),
    )
    budget_stable_checkpoints_value = _number(
        solver_cfg,
        "flow5_budget_stable_checkpoints",
        minimum=1.0,
        maximum=3.0,
    )
    if not float(budget_stable_checkpoints_value).is_integer():
        raise InputError("'flow5_budget_stable_checkpoints' tam sayı olmalı")
    try:
        budget_escalation_settings = BudgetEscalationSettings(
            enabled=_boolean(solver_cfg, "flow5_budget_escalation_enabled"),
            growth_factor=_number(
                solver_cfg,
                "flow5_budget_growth_factor",
                minimum=1.1,
                maximum=4.0,
            ),
            maximum_multiplier=_number(
                solver_cfg,
                "flow5_budget_maximum_multiplier",
                minimum=1.0,
                maximum=16.0,
            ),
            convergence_tolerance_percent=_number(
                solver_cfg,
                "flow5_budget_convergence_tolerance_percent",
                minimum=0.01,
                maximum=10.0,
            ),
            stable_checkpoints_required=int(budget_stable_checkpoints_value),
        )
    except ValueError as exc:
        raise InputError(str(exc)) from None
    flow5_checkpoint_enabled = _boolean(solver_cfg, "flow5_checkpoint_enabled")
    flow5_checkpoint_dir_value = solver_cfg.get("flow5_checkpoint_dir", "")
    if not isinstance(flow5_checkpoint_dir_value, str):
        raise InputError("'flow5_checkpoint_dir' metin yol olmalı")
    flow5_checkpoint_dir = flow5_checkpoint_dir_value.strip()
    multi_seed_runs_value = _number(
        solver_cfg, "flow5_multi_seed_runs", minimum=1.0, maximum=5.0
    )
    multi_seed_stride_value = _number(
        solver_cfg, "flow5_multi_seed_stride", minimum=1.0, maximum=10_000_000.0
    )
    if not float(multi_seed_runs_value).is_integer() or not float(multi_seed_stride_value).is_integer():
        raise InputError("Multi-seed koşu sayısı ve seed adımı tam sayı olmalı")
    flow5_multi_seed_runs = int(multi_seed_runs_value)
    flow5_multi_seed_stride = int(multi_seed_stride_value)
    flow5_multi_seed_objective_cv_tolerance = _number(
        solver_cfg,
        "flow5_multi_seed_objective_cv_tolerance_percent",
        minimum=0.1,
        maximum=50.0,
    )
    flow5_multi_seed_geometry_cv_tolerance = _number(
        solver_cfg,
        "flow5_multi_seed_geometry_cv_tolerance_percent",
        minimum=0.1,
        maximum=50.0,
    )
    validation_settings = ValidationSettings(
        enabled=_boolean(validation_cfg, "enabled"),
        force_closure_tolerance_percent=_number(
            validation_cfg,
            "force_closure_tolerance_percent",
            minimum=0.01,
            maximum=25.0,
        ),
        drag_decomposition_tolerance_percent=_number(
            validation_cfg,
            "drag_decomposition_tolerance_percent",
            minimum=0.1,
            maximum=100.0,
        ),
        minimum_span_efficiency=_number(
            validation_cfg,
            "minimum_span_efficiency",
            minimum=0.01,
            maximum=2.0,
        ),
        maximum_span_efficiency=_number(
            validation_cfg,
            "maximum_span_efficiency",
            minimum=0.01,
            maximum=2.0,
        ),
    )
    if validation_settings.maximum_span_efficiency < validation_settings.minimum_span_efficiency:
        raise InputError("Maksimum span verimi minimumdan küçük olamaz")
    baseline_profile = None
    if strategy == "flow5_native":
        if not speed_bounds[0] <= speed <= speed_bounds[1]:
            raise InputError("Referans hız, minimum ve maksimum hız aralığının içinde olmalı")
        resolved_flow5_runner = resolve_flow5_runner_path(flow5_runner_path)
        if resolved_flow5_runner is None:
            raise InputError(
                "Paket içi flow5 runner bulunamadı; kaynak sürümde aeropt-flow5-runner yolunu girin"
            )
        flow5_runner_path = str(resolved_flow5_runner)
        baseline_identifier = str(airfoil_cfg.get("baseline_profile", "e818")).strip()
        baseline_dat_value = airfoil_cfg.get("baseline_dat", "")
        if not isinstance(baseline_dat_value, str):
            raise InputError("'baseline_dat' metin içerikli bir DAT dosyası olmalı")
        cst_order_value = _number(airfoil_cfg, "cst_order", minimum=5.0, maximum=6.0)
        cst_order = int(cst_order_value)
        if not float(cst_order_value).is_integer() or cst_order not in {5, 6}:
            raise InputError("'cst_order' 5 veya 6 olmalı")
        solver_point_value = _number(
            airfoil_cfg, "solver_coordinate_points", minimum=100.0, maximum=100.0
        )
        if not float(solver_point_value).is_integer():
            raise InputError("'solver_coordinate_points' tam sayı olmalı")
        solver_coordinate_points = int(solver_point_value)
        try:
            baseline_profile = build_baseline_profile(
                baseline_identifier,
                custom_dat=baseline_dat_value,
                cst_order=cst_order,
                solver_point_count=solver_coordinate_points,
            )
        except ValueError as exc:
            raise InputError(str(exc)) from None

    q = fluid.dynamic_pressure(speed)
    reference_chord = 0.72 * root_chord_bounds[1]
    estimated_taper = float(np.clip(0.55, *taper_bounds))
    estimated_area = (
        0.72
        * span_bounds[1]
        * root_chord_bounds[1]
        * (1.0 + estimated_taper)
        / 2.0
    )
    design_cl_raw = airfoil_cfg.get("design_cl")
    auto_design_cl = design_cl_raw in (None, "", 0, 0.0)
    if auto_design_cl:
        raw_design_cl = target_lift / max(q * estimated_area, 1e-9)
        design_cl = float(raw_design_cl if strategy == "flow5_native" else np.clip(raw_design_cl, 0.25, 1.18))
    else:
        try:
            design_cl = float(design_cl_raw)
        except (TypeError, ValueError):
            raise InputError("'design_cl' boş veya geçerli bir sayı olmalı") from None
        if not 0.1 <= design_cl <= 1.6:
            raise InputError("'design_cl' 0.1 ile 1.6 arasında olmalı")

    if strategy == "flow5_native":
        assert baseline_profile is not None
        settings = Flow5NativeSettings(
            runner_path=flow5_runner_path,
            timeout_seconds=flow5_timeout,
            threads=flow5_threads,
            foil_candidate_budget=flow5_foil_budget,
            wing_candidate_budget=flow5_wing_budget,
            finalists=flow5_finalists,
            search_method=flow5_search_method,
            final_method=flow5_final_method,
            alpha_step_search_deg=flow5_alpha_step_search,
            alpha_step_final_deg=flow5_alpha_step_final,
            ncrit=flow5_ncrit,
            xtr_top=flow5_xtr_top,
            xtr_bottom=flow5_xtr_bottom,
            baseline_profile=baseline_profile,
            cst_order=baseline_profile.cst_order,
            foil_coordinate_points=baseline_profile.solver_point_count,
            foil_minimum_improvement_percent=flow5_minimum_improvement,
            seed=seed,
            search_mesh=flow5_search_mesh,
            final_mesh=flow5_final_mesh,
            convergence_mesh=flow5_convergence_mesh,
            mesh_convergence_enabled=flow5_mesh_convergence_enabled,
            mesh_cd_tolerance_percent=flow5_mesh_cd_tolerance,
            mesh_alpha_tolerance_deg=flow5_mesh_alpha_tolerance,
            cache_enabled=flow5_cache_enabled,
            cache_dir=flow5_cache_dir,
            foil_optimizer=flow5_foil_optimizer,
            wing_optimizer=flow5_wing_optimizer,
            coupled_iterations=flow5_coupled_iterations,
            coupling_cl_tolerance_percent=flow5_coupling_cl_tolerance,
            coupling_objective_tolerance_percent=flow5_coupling_objective_tolerance,
            multi_section_geometry_enabled=multi_section_geometry_enabled,
            mid_chord_factor_bounds=mid_chord_factor_bounds,
            mid_twist_bounds=mid_twist_bounds,
            winglet_optimization_enabled=winglet_optimization_enabled,
            winglet_naca_code=winglet_naca_code,
            winglet_candidate_budget=flow5_winglet_budget,
            winglet_height_bounds=winglet_height_bounds,
            winglet_cant_bounds=winglet_cant_bounds,
            winglet_toe_bounds=winglet_toe_bounds,
            winglet_taper_bounds=winglet_taper_bounds,
            structural_settings=structural_settings,
            hydro_settings=hydro_settings,
            spanwise_airfoil_optimization_enabled=flow5_spanwise_airfoil_optimization_enabled,
            spanwise_foil_budget_fraction=flow5_spanwise_foil_budget_fraction,
            spanwise_foil_acceptance_tolerance_percent=flow5_spanwise_foil_acceptance_tolerance,
            surrogate_settings=surrogate_settings,
            budget_escalation_settings=budget_escalation_settings,
            checkpoint_enabled=flow5_checkpoint_enabled,
            checkpoint_dir=flow5_checkpoint_dir,
            multi_seed_runs=flow5_multi_seed_runs,
            multi_seed_stride=flow5_multi_seed_stride,
            multi_seed_objective_cv_tolerance_percent=flow5_multi_seed_objective_cv_tolerance,
            multi_seed_geometry_cv_tolerance_percent=flow5_multi_seed_geometry_cv_tolerance,
            validation_settings=validation_settings,
        )
        native_kwargs = {
            "workflow_mode": workflow_mode,
            "fluid": fluid,
            "fluid_key": fluid_key,
            "reference_speed_m_s": speed,
            "speed_bounds_m_s": speed_bounds,
            "speed_samples": speed_samples,
            "target_lift_n": target_lift,
            "design_cl_at_reference": design_cl,
            "design_cl_was_auto": auto_design_cl,
            "reference_chord_m": reference_chord,
            "camber_bounds": camber_bounds,
            "camber_position_bounds": camber_position_bounds,
            "thickness_bounds": thickness_bounds,
            "span_bounds": span_bounds,
            "root_chord_bounds": root_chord_bounds,
            "taper_bounds": taper_bounds,
            "sweep_bounds": sweep_bounds,
            "twist_bounds": twist_bounds,
            "alpha_bounds": alpha_bounds,
            "cancel_event": cancel_event,
        }
        if workflow_mode == "foil_only":
            foil_result = run_flow5_native_design(
                request=request,
                settings=replace(settings, multi_seed_runs=1),
                progress_callback=progress_callback,
                **native_kwargs,
            )
            foil_result["multi_seed_stability"] = {
                "enabled": False,
                "runs_requested": 1,
                "runs_completed": 1,
                "selected_seed": settings.seed,
                "stable": None,
                "status": "not_applicable",
                "runs": [],
            }
            return foil_result
        seed_records: list[dict[str, Any]] = []
        for run_index in range(settings.multi_seed_runs):
            run_seed = settings.seed + run_index * settings.multi_seed_stride
            run_settings = replace(settings, seed=run_seed)
            run_request = deepcopy(request)
            run_request["solver"]["seed"] = run_seed

            def seed_progress(
                event: dict[str, Any],
                *,
                index: int = run_index,
                total: int = settings.multi_seed_runs,
                seed_value: int = run_seed,
            ) -> None:
                if progress_callback is None:
                    return
                local_percent = float(event.get("percent", 0.0))
                overall_percent = 100.0 * (index + local_percent / 100.0) / total
                prefix = f"Seed {index + 1}/{total} · " if total > 1 else ""
                progress_callback(
                    {
                        **event,
                        "seed_run": index + 1,
                        "seed_run_total": total,
                        "seed": seed_value,
                        "percent": overall_percent,
                        "message": prefix + str(event.get("message", "flow5 çalışıyor")),
                    }
                )

            try:
                run_result = run_flow5_native_design(
                    request=run_request,
                    settings=run_settings,
                    progress_callback=seed_progress,
                    **native_kwargs,
                )
                run_result["validation_report"] = build_validation_report(
                    run_result, settings.validation_settings
                )
                seed_records.append(
                    {"seed": run_seed, "result": run_result, "request": run_request}
                )
            except Flow5CancelledError:
                raise
            except Exception as exc:
                if settings.multi_seed_runs == 1:
                    raise
                seed_records.append(
                    {
                        "seed": run_seed,
                        "result": None,
                        "request": run_request,
                        "error": str(exc)[-800:],
                        "failure_diagnosis": diagnose_runtime_failure(exc),
                    }
                )

        stability = build_multi_seed_report(
            seed_records,
            objective_cv_tolerance_percent=settings.multi_seed_objective_cv_tolerance_percent,
            geometry_cv_tolerance_percent=settings.multi_seed_geometry_cv_tolerance_percent,
            geometry_reference_ranges={
                "span": span_bounds[1] - span_bounds[0],
                "root_chord": root_chord_bounds[1] - root_chord_bounds[0],
                "taper": taper_bounds[1] - taper_bounds[0],
                "sweep_deg": sweep_bounds[1] - sweep_bounds[0],
                "tip_twist_deg": twist_bounds[1] - twist_bounds[0],
                "winglet_height": winglet_height_bounds[1] - winglet_height_bounds[0],
                "winglet_cant_deg": winglet_cant_bounds[1] - winglet_cant_bounds[0],
                "winglet_toe_deg": winglet_toe_bounds[1] - winglet_toe_bounds[0],
                "winglet_taper": winglet_taper_bounds[1] - winglet_taper_bounds[0],
            },
        )
        selected_record = seed_records[int(stability["selected_record_index"])]
        selected_result = selected_record["result"]
        assert isinstance(selected_result, dict)
        selected_result["multi_seed_stability"] = stability
        selected_result["validation_report"] = build_validation_report(
            selected_result, settings.validation_settings
        )
        selected_result["solver_run"]["multi_seed_runs"] = settings.multi_seed_runs
        selected_result["solver_run"]["selected_seed"] = stability["selected_seed"]
        selected_result["flow5_native_analysis"]["multi_seed_stability"] = stability
        selected_result["flow5_native_analysis"]["validation_report"] = selected_result[
            "validation_report"
        ]
        diagnostic_bounds = {
            "span": span_bounds,
            "root_chord": root_chord_bounds,
            "taper": taper_bounds,
            "sweep_deg": sweep_bounds,
            "tip_twist_deg": twist_bounds,
            "winglet_height": winglet_height_bounds,
            "winglet_cant_deg": winglet_cant_bounds,
            "winglet_toe_deg": winglet_toe_bounds,
            "winglet_taper": winglet_taper_bounds,
        }
        selected_result["diagnostic_report"] = build_diagnostic_report(
            selected_result, diagnostic_bounds
        )
        if not selected_result["validation_report"].get("passed", True) or stability.get("stable") is False:
            selected_result["status"] = "review"
        validation = selected_result["validation_report"]
        selected_result["insights"].insert(
            0,
            {
                "level": "good" if validation.get("passed") else "bad",
                "title": "Doğrulama/regresyon sözleşmesi " + ("geçti" if validation.get("passed") else "inceleme istiyor"),
                "text": (
                    f"{validation.get('checks_passed', 0)}/{validation.get('checks_total', 0)} kontrol geçti; "
                    f"regresyon imzası {str(validation.get('regression_signature_sha256', ''))[:12]}."
                ),
            },
        )
        if stability["enabled"]:
            selected_result["insights"].insert(
                1,
                {
                    "level": "good" if stability.get("stable") else "warn",
                    "title": "Multi-seed sonuçları " + ("kararlı" if stability.get("stable") else "dağılıyor"),
                    "text": (
                        f"{stability['runs_completed']}/{stability['runs_requested']} koşu tamamlandı; "
                        f"amaç CV %{stability['objective_cv_percent']:.2f}, seçilen seed {stability['selected_seed']}."
                    ),
                },
            )
        primary = selected_result["diagnostic_report"].get("primary_cause")
        if primary:
            selected_result["insights"].append(
                {
                    "level": "bad" if primary["severity"] == "critical" else "warn",
                    "title": "Otomatik teşhis: " + primary["title"],
                    "text": primary["evidence"] + " " + primary["recommendation"],
                }
            )
        refresh_flow5_exports(selected_result, selected_record["request"])
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "completed",
                    "current": settings.multi_seed_runs,
                    "total": settings.multi_seed_runs,
                    "fraction": 1.0,
                    "percent": 100.0,
                    "message": "Doğrulama, Pareto ve kararlılık raporları tamamlandı",
                }
            )
        return selected_result

    effort = EFFORTS[quality]
    initial_foil, foil_meta = optimize_airfoil(
        fluid=fluid,
        speed=speed,
        reference_chord=reference_chord,
        target_cl=design_cl,
        camber_bounds=camber_bounds,
        camber_position_bounds=camber_position_bounds,
        thickness_bounds=thickness_bounds,
        alpha_bounds=alpha_bounds,
        effort=effort,
        seed=seed,
        parallel_workers=workers,
    )
    alpha_start = max(-8.0, alpha_bounds[0] - 2.0)
    alpha_stop = min(22.0, alpha_bounds[1] + 4.0)
    final_foil = initial_foil
    final_xfoil_polar: dict[str, Any] | None = None
    airfoil_validation: dict[str, Any] | None = None
    polar_mesh: list[dict[str, Any]] | None = None

    if strategy.startswith("xfoil_"):
        final_foil, final_xfoil_polar, airfoil_validation = run_closed_loop_airfoil(
            executable=xfoil_path,
            initial_foil=initial_foil,
            internal_point=foil_meta["design_point"],
            reynolds=foil_meta["reynolds"],
            mach=foil_meta["mach"],
            target_cl=design_cl,
            alpha_range=(alpha_start, alpha_stop),
            alpha_bounds=alpha_bounds,
            camber_bounds=camber_bounds,
            camber_position_bounds=camber_position_bounds,
            thickness_bounds=thickness_bounds,
            strategy=strategy,
            cl_tolerance_percent=cl_tolerance,
            cd_tolerance_percent=cd_tolerance,
            candidate_budget=candidate_budget,
            workers=workers,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        polar_mesh = build_xfoil_polar_mesh(
            executable=xfoil_path,
            foil=final_foil,
            reynolds_values=[
                fluid.reynolds(speed, root_chord_bounds[0] * taper_bounds[0]),
                foil_meta["reynolds"],
                fluid.reynolds(speed, root_chord_bounds[1]),
            ],
            mach=foil_meta["mach"],
            alpha_range=(alpha_start, alpha_stop),
            timeout_seconds=timeout_seconds,
            workers=workers,
            reference_polar=final_xfoil_polar,
        )

    wing, baseline, wing_meta = optimize_wing(
        foil=final_foil,
        fluid=fluid,
        speed=speed,
        target_lift=target_lift,
        span_bounds=span_bounds,
        root_chord_bounds=root_chord_bounds,
        taper_bounds=taper_bounds,
        sweep_bounds=sweep_bounds,
        twist_bounds=twist_bounds,
        alpha_bounds=alpha_bounds,
        effort=effort,
        seed=seed,
        modes=modes,
        polar_mesh=polar_mesh,
        parallel_workers=workers,
    )

    if final_xfoil_polar:
        polar_rows = [
            {**point, "ld": float(point["cl"] / max(point["cd"], 1e-12))}
            for point in final_xfoil_polar["points"]
        ]
        polar_source = "XFOIL"
        cruise_2d = airfoil_validation["final_xfoil_point"] if airfoil_validation else None
    else:
        count = int(ceil((alpha_stop - alpha_start) / 0.5)) + 1
        internal_polar = generate_polar(
            final_foil,
            foil_meta["reynolds"],
            foil_meta["mach"],
            np.linspace(alpha_start, alpha_stop, count),
        )
        polar_rows = [point.to_dict() for point in internal_polar]
        polar_source = "AeroOpt internal"
        cruise_2d = foil_meta["design_point"]

    x, y = naca4_coordinates(final_foil)
    airfoil_model = (
        "XFOIL viscous polar + CST/Kulfan free-shape search"
        if final_foil.family.startswith("CST")
        else (
            "XFOIL-validated NACA geometry and viscous polar"
            if final_xfoil_polar
            else "Thin-airfoil + empirical viscous/stall correlations"
        )
    )
    result: dict[str, Any] = {
        "status": "feasible" if wing_meta["feasible"] else "review",
        "model": {
            "airfoil": airfoil_model,
            "wing": f"Prandtl lifting-line, {modes} odd Fourier modes; {wing.section_polar_source}",
            "scope": "Incompressible/low-subsonic preliminary design",
        },
        "solver_run": {
            "strategy_requested": requested_strategy,
            "strategy_used": strategy,
            "parallel_workers_requested": workers,
            "candidate_budget": candidate_budget,
            "polar_source": polar_source,
        },
        "flow": {
            **fluid.to_dict(),
            "speed_m_s": speed,
            "target_lift_n": target_lift,
            "dynamic_pressure_pa": q,
            "mach": fluid.mach(speed),
        },
        "airfoil": final_foil.to_dict(),
        "airfoil_optimization": {
            **foil_meta,
            "design_cl_was_auto": auto_design_cl,
            "initial_naca": initial_foil.to_dict(),
            "final_cruise_point": cruise_2d,
        },
        "airfoil_coordinates": [
            {"x_over_c": float(xi), "y_over_c": float(yi)} for xi, yi in zip(x, y)
        ],
        "polar": polar_rows,
        "polar_source": polar_source,
        "wing": wing.to_dict(),
        "rectangular_baseline": baseline.to_dict(),
        "wing_optimization": wing_meta,
    }
    if airfoil_validation is not None:
        result["airfoil_validation"] = airfoil_validation
        result["xfoil_validation"] = final_xfoil_polar
        result["xfoil_polar_mesh"] = polar_mesh

    result["insights"] = _insights(
        result,
        fluid_key,
        {"span": span_bounds, "thickness": thickness_bounds},
    )
    if airfoil_validation:
        initial_check = airfoil_validation["initial_check"]
        if airfoil_validation["escalated_to_cst"]:
            cst = airfoil_validation.get("cst_optimization") or {}
            cl_error = initial_check.get("cl_error_percent")
            cd_error = initial_check.get("cd_error_percent")
            cl_error_text = f"{cl_error:.2f}%" if cl_error is not None else "yakınsama yok"
            cd_error_text = f"{cd_error:.2f}%" if cd_error is not None else "yakınsama yok"
            result["insights"].insert(
                0,
                {
                    "level": "good",
                    "title": "XFOIL farkı CST aramasını tetikledi",
                    "text": (
                        f"İlk NACA adayında CL farkı {cl_error_text}, "
                        f"CD farkı {cd_error_text} oldu. "
                        f"{cst.get('candidates_evaluated', 0)} CST adayı "
                        f"{cst.get('parallel_workers_used', 1)} paralel işçiyle doğrudan XFOIL'de değerlendirildi."
                    ),
                },
            )
        else:
            result["insights"].insert(
                0,
                {
                    "level": "good",
                    "title": "NACA adayı XFOIL kontrolünü geçti",
                    "text": (
                        f"Aynı hücum açısındaki CL farkı {initial_check['cl_error_percent']:.2f}%, "
                        f"aynı CL'deki CD farkı {initial_check['cd_error_percent']:.2f}% sınırlar içinde kaldı."
                    ),
                },
            )
    if wing.ld < 15.0:
        result["insights"].append(
            {
                "level": "warn",
                "title": "Kanat L/D düşük",
                "text": (
                    f"Hesaplanan 3B L/D {wing.ld:.1f}. Profil CD'si, düşük Reynolds, stall marjı veya "
                    "boyut kısıtları baskın olabilir; CD_profile/CD_induced ayrımını ve XFOIL yakınsama aralığını kontrol edin."
                ),
            }
        )

    foil_text = airfoil_dat(final_foil)
    plane_text = flow5_plane_xml(final_foil, wing.geometry)
    obj_text = wing_obj(final_foil, wing.geometry)
    results_text = results_csv(final_foil, wing)
    snapshot = deepcopy(result)
    project_text = project_json(request, snapshot)
    polar_text = xfoil_polar_csv(final_xfoil_polar["points"]) if final_xfoil_polar else None
    bundle = flow5_bundle_bytes(
        foil_dat_text=foil_text,
        plane_xml_text=plane_text,
        wing_obj_text=obj_text,
        results_csv_text=results_text,
        project_json_text=project_text,
        polar_csv_text=polar_text,
    )
    result["exports"] = {
        "airfoil_filename": f"{final_foil.name}.dat",
        "airfoil_dat": foil_text,
        "plane_filename": "aeropt-wing.xml",
        "plane_xml": plane_text,
        "wing_obj_filename": "aeropt-wing.obj",
        "wing_obj": obj_text,
        "results_filename": "aeropt-results.csv",
        "results_csv": results_text,
        "project_filename": "aeropt-project.json",
        "project_json": project_text,
        "xfoil_polar_filename": "xfoil-polar.csv" if polar_text else None,
        "xfoil_polar_csv": polar_text,
        "flow5_bundle_filename": "aeropt-flow5-package.zip",
        "flow5_bundle_base64": base64.b64encode(bundle).decode("ascii"),
    }
    return result
