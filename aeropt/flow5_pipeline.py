from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import threading
from typing import Any, Callable

import numpy as np

from .airfoil import airfoil_coordinates, naca4_design
from .baselines import BaselineProfile, build_derived_baseline_profile
from .checkpoint import OptimizerCheckpointStore, optimizer_fingerprint
from .convergence import BudgetEscalationSettings
from .exporters import (
    airfoil_dat,
    flow5_analysis_xml,
    flow5_bundle_bytes,
    flow5_native_results_csv,
    flow5_plane_xml,
    project_json,
    wing_obj,
    xfoil_polar_csv,
)
from .flow5 import Flow5CancelledError, Flow5Mesh, Flow5Runner
from .flow5_optimization import (
    evaluate_fixed_airfoil_with_flow5,
    optimize_airfoil_with_flow5,
    optimize_wing_with_flow5,
)
from .hydro import HydroSettings
from .models import AirfoilLike, CSTAirfoilDesign, Fluid, WingGeometry
from .structures import StructuralSettings
from .surrogate import SurrogateSettings
from .validation import ValidationSettings


@dataclass(frozen=True)
class Flow5NativeSettings:
    runner_path: str
    timeout_seconds: float
    threads: int
    foil_candidate_budget: int
    wing_candidate_budget: int
    finalists: int
    search_method: str
    final_method: str
    alpha_step_search_deg: float
    alpha_step_final_deg: float
    ncrit: float
    xtr_top: float
    xtr_bottom: float
    baseline_profile: BaselineProfile
    cst_order: int
    foil_coordinate_points: int
    foil_minimum_improvement_percent: float
    seed: int
    search_mesh: Flow5Mesh
    final_mesh: Flow5Mesh
    convergence_mesh: Flow5Mesh
    mesh_convergence_enabled: bool
    mesh_cd_tolerance_percent: float
    mesh_alpha_tolerance_deg: float
    cache_enabled: bool
    cache_dir: str
    foil_optimizer: str
    wing_optimizer: str
    coupled_iterations: int
    coupling_cl_tolerance_percent: float
    coupling_objective_tolerance_percent: float
    multi_section_geometry_enabled: bool
    mid_chord_factor_bounds: tuple[float, float]
    mid_twist_bounds: tuple[float, float]
    winglet_optimization_enabled: bool
    winglet_naca_code: str
    winglet_candidate_budget: int
    winglet_height_bounds: tuple[float, float]
    winglet_cant_bounds: tuple[float, float]
    winglet_toe_bounds: tuple[float, float]
    winglet_taper_bounds: tuple[float, float]
    structural_settings: StructuralSettings
    hydro_settings: HydroSettings
    spanwise_airfoil_optimization_enabled: bool
    spanwise_foil_budget_fraction: float
    spanwise_foil_acceptance_tolerance_percent: float
    surrogate_settings: SurrogateSettings
    budget_escalation_settings: BudgetEscalationSettings
    checkpoint_enabled: bool
    checkpoint_dir: str
    multi_seed_runs: int
    multi_seed_stride: int
    multi_seed_objective_cv_tolerance_percent: float
    multi_seed_geometry_cv_tolerance_percent: float
    validation_settings: ValidationSettings


def _sample_speeds(
    bounds_m_s: tuple[float, float], reference_speed_m_s: float, count: int
) -> list[float]:
    """Return the reference alone for one point, otherwise an endpoint mesh."""
    if count <= 0:
        raise ValueError("speed sample count must be positive")
    if count == 1:
        return [float(reference_speed_m_s)]
    if np.isclose(bounds_m_s[0], bounds_m_s[1], rtol=0.0, atol=1e-10):
        return [float(reference_speed_m_s)]
    values = np.linspace(*bounds_m_s, count)
    if count >= 3 and not np.any(
        np.isclose(values, reference_speed_m_s, rtol=0.0, atol=1e-10)
    ):
        internal = np.arange(1, count - 1)
        replace = int(internal[np.argmin(np.abs(values[internal] - reference_speed_m_s))])
        values[replace] = reference_speed_m_s
    return [float(value) for value in np.sort(values)]


def _representative_section_cls(wing: dict[str, Any]) -> list[float]:
    """Return an area-weighted RMS section Cl schedule from flow5 span telemetry."""
    values: list[float] = []
    for condition in wing.get("conditions", []):
        point = condition.get("point", {})
        distribution = point.get("distribution", [])
        local_cls: list[float] = []
        weights: list[float] = []
        for station in distribution:
            try:
                local_cl = float(station["local_cl"])
                chord = max(float(station["chord_m"]), 1e-9)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(local_cl) and np.isfinite(chord):
                local_cls.append(local_cl)
                weights.append(chord)
        if local_cls:
            representative = float(
                np.sqrt(np.average(np.square(local_cls), weights=weights))
            )
        else:
            representative = float(condition["target_cl"])
        values.append(float(np.clip(representative, 0.05, 1.8)))
    return values


def _span_station_cls(wing: dict[str, Any], eta: float) -> list[float]:
    values: list[float] = []
    semispan = 0.5 * float(wing["geometry"]["span"])
    target_y = float(np.clip(eta, 0.0, 1.0)) * semispan
    for condition in wing.get("conditions", []):
        distribution = condition.get("point", {}).get("distribution", [])
        usable = []
        for station in distribution:
            try:
                y = abs(float(station["y_m"]))
                local_cl = float(station["local_cl"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(y) and np.isfinite(local_cl):
                usable.append((y, local_cl))
        if usable:
            local_cl = min(usable, key=lambda row: abs(row[0] - target_y))[1]
        else:
            local_cl = float(condition["target_cl"])
        values.append(float(np.clip(local_cl, 0.05, 1.8)))
    return values


def _rename_section_foil(
    foil: CSTAirfoilDesign, dat_text: str, station: str
) -> tuple[CSTAirfoilDesign, str]:
    renamed = CSTAirfoilDesign(
        foil.upper_weights,
        foil.lower_weights,
        foil.max_camber,
        foil.camber_position,
        foil.thickness,
        f"{station}-{foil.name}",
        foil.trailing_edge_gap,
    )
    lines = dat_text.splitlines()
    renamed_dat = "\n".join([renamed.name, *lines[1:]]) + "\n"
    return renamed, renamed_dat


def _native_insights(
    *,
    wing: dict[str, Any],
    wing_meta: dict[str, Any],
    foil_meta: dict[str, Any],
    fluid_key: str,
    span_upper: float,
    coupled_design: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    geometry = wing["geometry"]
    conditions = wing.get("conditions", [])
    messages: list[dict[str, str]] = [
        {
            "level": "good",
            "title": "Aerodinamik puanların tamamı flow5 kaynaklı",
            "text": (
                f"Profil için {foil_meta['candidates_evaluated']} CST adayı gömülü XFoil ile; "
                f"kanat için {wing_meta['candidates_evaluated']} planform {wing_meta['search_method']} ile tarandı. "
                f"Son {wing_meta['finalists_evaluated']} aday {wing_meta['final_method']} ile yeniden çözüldü."
            ),
        }
    ]
    if coupled_design and coupled_design.get("enabled"):
        messages.append(
            {
                "level": "good" if coupled_design.get("converged") else "warn",
                "title": "Foil–kanat geri beslemesi tamamlandı",
                "text": (
                    f"{coupled_design.get('iterations_completed', 1)} profil–kanat turu ve "
                    f"{coupled_design.get('feedback_cycles_completed', 0)} gerçek MAC/Re + spanwise kesit Cl geri beslemesi tamamlandı; "
                    f"çıktı son profil–kanat çifti olan "
                    f"{coupled_design.get('selected_iteration', 1)}. turdan üretildi."
                ),
            }
        )
    mesh_check = wing_meta.get("mesh_convergence", {})
    if mesh_check.get("enabled"):
        messages.append(
            {
                "level": "good" if mesh_check.get("passed") else "warn",
                "title": (
                    "Panel ağı yakınsadı"
                    if mesh_check.get("passed")
                    else "Panel ağı yakınsama toleransını aştı"
                ),
                "text": (
                    f"Final ve ince ağ arasında en büyük CD farkı "
                    f"%{float(mesh_check.get('max_cd_change_percent', 0.0)):.2f}, "
                    f"α farkı {float(mesh_check.get('max_alpha_change_deg', 0.0)):.3f}°."
                ),
            }
        )
    telemetry = wing_meta.get("solver_telemetry", {})
    if telemetry.get("spanwise_distribution_available"):
        messages.append(
            {
                "level": "info",
                "title": "Spanwise flow5 telemetrisi alındı",
                "text": "Yerel Cl, Reynolds, profil/indüklenmiş sürükleme, yakınsama ve yük dağılımı flow5 çalışma noktalarından aktarıldı.",
            }
        )
    winglet = wing_meta.get("winglet_comparison", {})
    if winglet.get("enabled"):
        if winglet.get("performed"):
            drag_delta = float(
                winglet.get("delta_winglet_vs_planar", {}).get("drag_percent", 0.0)
            )
            messages.append(
                {
                    "level": "good" if winglet.get("selected") else "info",
                    "title": (
                        "Winglet net kazanç sağladı ve seçildi"
                        if winglet.get("selected")
                        else "Planar kanat winglete üstün kaldı"
                    ),
                    "text": (
                        "Aynı izdüşüm açıklığı ve taşıma hedefinde flow5 karşılaştırması "
                        f"winglet için toplam sürükleme değişimini %{drag_delta:+.2f} buldu. "
                        "Kök momenti seçim amacına veya fizibilite kararına girmez."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "level": "warn",
                    "title": "Winglet aşaması tamamlanamadı; planar sonuç korundu",
                    "text": str(winglet.get("error", "flow5 winglet adayı üretmedi")),
                }
            )
    spanwise_foils = wing_meta.get("spanwise_airfoil_refinement", {})
    if spanwise_foils.get("performed"):
        messages.append(
            {
                "level": "good" if spanwise_foils.get("selected") else "info",
                "title": (
                    "Kök–orta–uç profilleri seçildi"
                    if spanwise_foils.get("selected")
                    else "Üç profilli kanat kazanç sağlamadı"
                ),
                "text": (
                    "Yerel Reynolds ve Cl hedefleriyle orta/uç profilleri ayrı flow5/XFoil aramasından geçti; "
                    + (
                        "üç profil çözümlenmiş .fl5 projesine yazıldı."
                        if spanwise_foils.get("selected")
                        else "tek profil daha iyi amaç değeri verdiği için korundu."
                    )
                ),
            }
        )
    structure = wing_meta.get("structural_check", {})
    if structure.get("enabled"):
        if structure.get("performed"):
            messages.append(
                {
                    "level": "good" if structure.get("passed") else "bad",
                    "title": (
                        "İsteğe bağlı yapısal tarama geçti"
                        if structure.get("passed")
                        else "Yapısal tarama sınır aştı"
                    ),
                    "text": (
                        f"Gerilme kullanımı {float(structure.get('stress_utilization', 0.0)):.2f}, "
                        f"sehim kullanımı {float(structure.get('deflection_utilization', 0.0)):.2f}, "
                        f"burulma kullanımı {float(structure.get('twist_utilization', 0.0)):.2f}; "
                        f"tahmini malzeme kütlesi {float(structure.get('estimated_wing_material_mass_kg', 0.0)):.2f} kg."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "level": "bad",
                    "title": "Yapısal tarama için yük verisi yok",
                    "text": str(structure.get("reason", "Spanwise yük dağılımı alınamadı.")),
                }
            )
    hydro = wing_meta.get("hydro_check", {})
    if hydro.get("enabled"):
        hydro_report_only = hydro.get("constraint_mode") == "report_only"
        if hydro.get("performed"):
            messages.append(
                {
                    "level": (
                        "good"
                        if hydro.get("passed")
                        else "warn"
                        if hydro_report_only
                        else "bad"
                    ),
                    "title": (
                        "Kavitasyon taraması geçti"
                        if hydro.get("passed")
                        else "Kavitasyon riski raporlandı"
                        if hydro_report_only
                        else "Kavitasyon marjı yetersiz"
                    ),
                    "text": (
                        f"En kötü kavitasyon kullanımı "
                        f"{float(hydro.get('cavitation_utilization', 0.0)):.2f}; "
                        f"minimum emniyetli marj oranı "
                        f"{float(hydro.get('minimum_cavitation_margin_ratio', 0.0)):.2f}. "
                        f"Riskli finalist alanı "
                        f"%{float(hydro.get('risk_area_percent', 0.0)):.2f}. "
                        f"Serbest yüzey risk bayrağı: "
                        f"{'evet' if hydro.get('free_surface_risk') else 'hayır'}. "
                        + (
                            "Yalnız rapor modu L/D seçimini değiştirmedi."
                            if hydro_report_only
                            else "Sert kısıt seçimde uygulandı."
                        )
                    ),
                }
            )
        else:
            messages.append(
                {
                    "level": "warn" if hydro_report_only else "bad",
                    "title": "Kavitasyon için Cp_min alınamadı",
                    "text": str(hydro.get("reason", "flow5 panel basınçları bulunamadı.")),
                }
            )
    selection = foil_meta.get("selection", {})
    baseline = foil_meta.get("baseline", {})
    if selection.get("selected_baseline"):
        messages.append(
            {
                "level": "info",
                "title": "Başlangıç profili korundu",
                "text": (
                    f"{baseline.get('display_name', 'Başlangıç profili')} ince flow5 doğrulamasında "
                    "yeterince aşılmadığı için kanat optimizasyonuna bu profil aktarıldı."
                ),
            }
        )
    else:
        improvement = selection.get("selected_improvement_vs_baseline_percent")
        messages.append(
            {
                "level": "good",
                "title": "Başlangıç profilinden daha iyi profil seçildi",
                "text": (
                    f"Seçilen {selection.get('selected_family', 'CST')} profilinin çok-noktalı amaç "
                    f"iyileşmesi başlangıca göre %{float(improvement or 0.0):.2f}; aynı profil kanat aşamasına aktarıldı."
                ),
            }
        )
    if geometry["sweep_deg"] < 1.0:
        messages.append(
            {
                "level": "info",
                "title": "Sweep kazancı bulunmadı",
                "text": "flow5 araması verilen hız ve boyut zarfında çeyrek-kord sweep'i yaklaşık sıfırda seçti.",
            }
        )
    else:
        messages.append(
            {
                "level": "info",
                "title": "Sweep çözümde kaldı",
                "text": f"Son panel çözümündeki çeyrek-kord sweep {geometry['sweep_deg']:.2f}°.",
            }
        )
    if geometry["taper"] < 0.90:
        messages.append(
            {
                "level": "good",
                "title": "Taper seçildi",
                "text": f"flow5 amaç fonksiyonu taper oranını {geometry['taper']:.3f} seçti; dikdörtgen referansla drag farkı sonuç tablosunda gösteriliyor.",
            }
        )
    if geometry["tip_twist_deg"] < -0.25:
        messages.append(
            {
                "level": "good",
                "title": "Washout seçildi",
                "text": f"Uç twist değeri {geometry['tip_twist_deg']:.2f}°; bu değer tüm hız noktalarının ortak amaç fonksiyonundan çıktı.",
            }
        )
    if abs(geometry["span"] - span_upper) <= 0.01 * max(span_upper, 1e-9):
        messages.append(
            {
                "level": "warn",
                "title": "Açıklık üst sınıra dayandı",
                "text": "Aerodinamik amaç daha uzun açıklık istiyor olabilir. Nihai yapı, yükler raporlanarak ayrıca doğrulanmalıdır.",
            }
        )
    worst = max(conditions, key=lambda item: item.get("stall_ratio", 0.0), default=None)
    if worst and worst["stall_ratio"] > 0.90:
        messages.append(
            {
                "level": "warn",
                "title": "Düşük hız stall marjı dar",
                "text": f"{worst['speed_m_s']:.2f} m/s noktasında hedef CL / yakınsayan CLmax = {worst['stall_ratio']:.3f}.",
            }
        )
    if fluid_key in {"fresh_water", "sea_water"} and not hydro.get("enabled"):
        messages.append(
            {
                "level": "warn",
                "title": "Hidrofoil taraması kapalı",
                "text": "Kavitasyon ve serbest yüzey ön taraması kullanıcı ayarıyla kapalı; nihai tasarımda çok-fazlı CFD/deney doğrulaması gerekir.",
            }
        )
    if wing["ld"] < 15.0:
        messages.append(
            {
                "level": "warn",
                "title": "3B L/D düşük",
                "text": f"Referans hızdaki flow5 panel sonucu L/D = {wing['ld']:.1f}. CDv/CDi ayrımını, Reynolds aralığını ve hedef CL'nin polar içindeki yerini inceleyin.",
            }
        )
    if not wing_meta["feasible"]:
        messages.append(
            {
                "level": "bad",
                "title": "Bütün akış noktaları fizibil değil",
                "text": "En az bir hızda stall, viskoz polar ağı ya da etkin yapı/hidrofoil koşulu ihlal edildi. Hız/boyut aralığını değiştirin.",
            }
        )
    for label, budget_report in (
        ("Profil", foil_meta.get("budget_convergence", {})),
        ("Kanat", wing_meta.get("budget_convergence", {})),
    ):
        if budget_report.get("converged") is True:
            messages.append(
                {
                    "level": "good",
                    "title": f"{label} aday bütçesi yeterli",
                    "text": (
                        f"{budget_report.get('evaluations_completed', 0)} gerçek çözümde izlenen "
                        f"değişim %{float((budget_report.get('checkpoints') or [{}])[-1].get('controlling_change_percent') or 0.0):.3f} ile tolerans içine girdi."
                    ),
                }
            )
        elif budget_report.get("converged") is False:
            messages.append(
                {
                    "level": "warn",
                    "title": f"{label} araması bütçe-sınırlı",
                    "text": (
                        f"Azami {budget_report.get('maximum_budget', 0)} gerçek çözümde amaç/Pareto cephesi hâlâ hareketli; "
                        "bütçe çarpanını veya multi-seed sayısını artırın."
                    ),
                }
            )
    return messages


def run_flow5_native_design(
    *,
    request: dict[str, Any],
    workflow_mode: str = "coupled",
    fluid: Fluid,
    fluid_key: str,
    reference_speed_m_s: float,
    speed_bounds_m_s: tuple[float, float],
    speed_samples: int,
    target_lift_n: float,
    design_cl_at_reference: float,
    design_cl_was_auto: bool,
    reference_chord_m: float,
    camber_bounds: tuple[float, float],
    camber_position_bounds: tuple[float, float],
    thickness_bounds: tuple[float, float],
    span_bounds: tuple[float, float],
    root_chord_bounds: tuple[float, float],
    taper_bounds: tuple[float, float],
    sweep_bounds: tuple[float, float],
    twist_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    settings: Flow5NativeSettings,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    if workflow_mode not in {"coupled", "foil_only", "wing_only"}:
        raise ValueError("Geçersiz flow5 çalışma modu")
    effective_coupled_iterations = (
        1 if workflow_mode in {"foil_only", "wing_only"} else settings.coupled_iterations
    )
    effective_spanwise_optimization = bool(
        settings.spanwise_airfoil_optimization_enabled and workflow_mode == "coupled"
    )
    # Each coupled pass is strictly planar.  The optional winglet search gets
    # its own final pass after coupling and spanwise-profile refinement.
    stage_ranges = {
        "foil_search": (0.01, 0.35),
        "foil_budget": (0.01, 0.35),
        "foil_final": (0.35, 0.45),
        "wing_search": (0.45, 0.75),
        "wing_budget": (0.45, 0.75),
        "wing_final": (0.75, 0.87),
        "mesh_convergence": (0.87, 0.97),
        "winglet_search": (0.01, 0.62),
        "winglet_budget": (0.01, 0.62),
        "winglet_final": (0.62, 0.82),
        "winglet_convergence": (0.82, 0.97),
    }

    active_iteration = 0
    progress_passes = (
        effective_coupled_iterations
        + int(effective_spanwise_optimization)
        + int(settings.winglet_optimization_enabled)
    )
    last_reported_percent = 0.0

    def report(event: dict[str, Any]) -> None:
        nonlocal last_reported_percent
        start, stop = stage_ranges.get(str(event.get("stage")), (0.0, 0.99))
        fraction = float(event.get("fraction", 0.0))
        local_fraction = start + (stop - start) * np.clip(fraction, 0.0, 1.0)
        overall_fraction = (
            active_iteration + local_fraction
        ) / max(progress_passes, 1)
        calculated_percent = float(100.0 * overall_fraction)
        last_reported_percent = max(last_reported_percent, calculated_percent)
        payload = {
            **event,
            "iteration": active_iteration + 1,
            "iteration_total": progress_passes,
            "percent": last_reported_percent,
        }
        if progress_callback is not None:
            progress_callback(payload)

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "initializing",
                "current": 0,
                "total": 1,
                "fraction": 0.0,
                "percent": 0.0,
                "message": "flow5 çalışma koşulları hazırlanıyor",
            }
        )
    requested_speed_samples = int(speed_samples)
    speeds = _sample_speeds(speed_bounds_m_s, reference_speed_m_s, speed_samples)
    speed_samples = len(speeds)
    target_cls = [
        float(design_cl_at_reference * (reference_speed_m_s / speed) ** 2) for speed in speeds
    ]
    runner = Flow5Runner(
        settings.runner_path,
        timeout_seconds=settings.timeout_seconds,
        cache_enabled=settings.cache_enabled,
        cache_dir=settings.cache_dir or None,
        cancel_event=cancel_event,
    )
    checkpoint_store = OptimizerCheckpointStore(
        enabled=settings.checkpoint_enabled,
        directory=settings.checkpoint_dir or None,
    )
    try:
        runner_stat = runner.path.stat()
        checkpoint_runner_identity: dict[str, Any] = {
            "path": str(runner.path),
            "size": runner_stat.st_size,
            "mtime_ns": runner_stat.st_mtime_ns,
        }
    except OSError:
        checkpoint_runner_identity = {"path": str(runner.path)}

    checkpoint_contract = {
        "contract": 6,
        "flow5_api_version": "7.57",
        "seed": settings.seed,
        "fluid": fluid.to_dict(),
        "speeds_m_s": speeds,
        "reference_speed_m_s": reference_speed_m_s,
        "target_lift_n": target_lift_n,
        "alpha_bounds": alpha_bounds,
        "foil_bounds": {
            "camber": camber_bounds,
            "camber_position": camber_position_bounds,
            "thickness": thickness_bounds,
            "cst_order": settings.cst_order,
            "coordinate_points": settings.foil_coordinate_points,
        },
        "wing_bounds": {
            "span": span_bounds,
            "root_chord": root_chord_bounds,
            "taper": taper_bounds,
            "sweep": sweep_bounds,
            "twist": twist_bounds,
            "multi_section": settings.multi_section_geometry_enabled,
            "mid_chord_factor": settings.mid_chord_factor_bounds,
            "mid_twist": settings.mid_twist_bounds,
            "winglet_enabled": settings.winglet_optimization_enabled,
            "winglet_naca_code": settings.winglet_naca_code,
            "winglet_execution_policy": "post_coupling_once",
            "winglet_height": settings.winglet_height_bounds,
            "winglet_cant": settings.winglet_cant_bounds,
            "winglet_toe": settings.winglet_toe_bounds,
            "winglet_taper": settings.winglet_taper_bounds,
        },
        "search": {
            "runner": checkpoint_runner_identity,
            "threads": settings.threads,
            "foil_budget": settings.foil_candidate_budget,
            "wing_budget": settings.wing_candidate_budget,
            "winglet_budget": settings.winglet_candidate_budget,
            "foil_optimizer": settings.foil_optimizer,
            "wing_optimizer": settings.wing_optimizer,
            "method": settings.search_method,
            "mesh": settings.search_mesh.to_dict(),
            "alpha_step_deg": settings.alpha_step_search_deg,
            "ncrit": settings.ncrit,
            "xtr_top": settings.xtr_top,
            "xtr_bottom": settings.xtr_bottom,
            "surrogate": settings.surrogate_settings.to_dict(),
            "budget_escalation": settings.budget_escalation_settings.to_dict(),
        },
        "structure": settings.structural_settings.to_dict(),
        "hydro": settings.hydro_settings.to_dict(),
    }

    def checkpoint_key(label: str, payload: dict[str, Any]) -> str:
        return optimizer_fingerprint(
            f"flow5-native-v6:{label}",
            {
                "problem": checkpoint_contract,
                "payload": payload,
            },
        )

    def geometry_from_result(wing_result: dict[str, Any]) -> WingGeometry:
        geometry = wing_result["geometry"]
        return WingGeometry(
            span=float(geometry["span"]),
            root_chord=float(geometry["root_chord"]),
            taper=float(geometry["taper"]),
            sweep_deg=float(geometry["sweep_deg"]),
            tip_twist_deg=float(geometry["tip_twist_deg"]),
            alpha_deg=float(geometry.get("alpha_deg", 0.0)),
            mid_chord_factor=float(geometry.get("mid_chord_factor", 1.0)),
            mid_twist_deg=geometry.get("mid_twist_deg"),
            winglet_enabled=bool(geometry.get("winglet_enabled", False)),
            winglet_height=float(geometry.get("winglet_height", 0.0)),
            winglet_cant_deg=float(geometry.get("winglet_cant_deg", 90.0)),
            winglet_toe_deg=float(geometry.get("winglet_toe_deg", 0.0)),
            winglet_taper=float(geometry.get("winglet_taper", 1.0)),
        )

    def planar_copy(geometry: WingGeometry | None) -> WingGeometry | None:
        if geometry is None:
            return None
        return WingGeometry(
            span=geometry.span,
            root_chord=geometry.root_chord,
            taper=geometry.taper,
            sweep_deg=geometry.sweep_deg,
            tip_twist_deg=geometry.tip_twist_deg,
            alpha_deg=geometry.alpha_deg,
            mid_chord_factor=geometry.mid_chord_factor,
            mid_twist_deg=geometry.mid_twist_deg,
        )

    def winglet_summary(
        wing_result: dict[str, Any], wing_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        geometry = wing_result["geometry"]
        return {
            "feasible": bool(wing_metadata.get("feasible", False)),
            "objective": float(wing_metadata["objective"]),
            "drag_n": float(wing_result["drag_n"]),
            "cd_total": float(wing_result["cd_total"]),
            "cd_induced": float(wing_result["cd_induced"]),
            "cd_profile": float(wing_result["cd_profile"]),
            "induced_drag_fraction_percent": float(
                wing_result.get(
                    "induced_drag_fraction_percent",
                    100.0
                    * float(wing_result["cd_induced"])
                    / max(float(wing_result["cd_total"]), 1.0e-12),
                )
            ),
            "ld": float(wing_result["ld"]),
            "root_bending_moment_nm": float(
                wing_result["root_bending_moment_nm"]
            ),
            "geometry": geometry,
            "candidates_evaluated": int(wing_metadata["candidates_evaluated"]),
            "budget_convergence": wing_metadata.get("budget_convergence", {}),
            "mesh_convergence": wing_metadata.get("mesh_convergence", {}),
        }

    def same_wing_geometry(
        left_result: dict[str, Any], right_result: dict[str, Any]
    ) -> bool:
        left = left_result.get("geometry", {})
        right = right_result.get("geometry", {})
        numeric_keys = (
            "span",
            "root_chord",
            "taper",
            "sweep_deg",
            "tip_twist_deg",
            "mid_chord_factor",
            "effective_mid_twist_deg",
            "winglet_height",
            "winglet_cant_deg",
            "winglet_toe_deg",
            "winglet_taper",
        )
        if bool(left.get("winglet_active")) != bool(right.get("winglet_active")):
            return False
        return all(
            math.isclose(
                float(left.get(key, 0.0)),
                float(right.get(key, 0.0)),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            for key in numeric_keys
        )

    def scalar_stage_choice(
        stage: str,
        wing_result: dict[str, Any],
        wing_metadata: dict[str, Any],
        wing_response: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        comparison = wing_metadata.get("selection_comparison", {})
        scalar = deepcopy(comparison.get("scalar_only", {}))
        if not scalar:
            scalar = {
                "policy": "fizibilite önceliği yok; yalnız en düşük skaler toplam amaç",
                "available": True,
                "deliverable": True,
                "same_as_feasibility_first": True,
                "feasible": bool(wing_metadata.get("feasible", False)),
                "constraint_violation": 0.0,
                "objective": float(wing_metadata["objective"]),
                "mesh_convergence": wing_metadata.get("mesh_convergence"),
                "wing": wing_result,
                "coarse_wing": None,
                "error": None,
            }
        scalar["stage"] = stage
        if scalar.get("same_as_feasibility_first"):
            return scalar, wing_response
        response = wing_response.get("_scalar_only_alternative_response")
        return scalar, response if isinstance(response, dict) else None

    def optimize_wing_stage(
        *,
        foil: CSTAirfoilDesign,
        foil_dat_text: str,
        seed: int,
        checkpoint_label: str,
        checkpoint_payload: dict[str, Any],
        initial_geometry: WingGeometry | None = None,
        section_foils: tuple[AirfoilLike, ...] | None = None,
        section_foil_dat_texts: tuple[str, ...] | None = None,
        run_winglet: bool = False,
        fixed_planar: tuple[
            dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
        ]
        | None = None,
        winglet_section_foils: tuple[AirfoilLike, ...] | None = None,
        winglet_section_foil_dat_texts: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        common = {
            "runner": runner,
            "foil": foil,
            "foil_dat_text": foil_dat_text,
            "fluid": fluid,
            "speeds_m_s": speeds,
            "reference_speed_m_s": reference_speed_m_s,
            "target_lift_n": target_lift_n,
            "span_bounds": span_bounds,
            "root_chord_bounds": root_chord_bounds,
            "taper_bounds": taper_bounds,
            "sweep_bounds": sweep_bounds,
            "twist_bounds": twist_bounds,
            "alpha_bounds": alpha_bounds,
            "finalists": settings.finalists,
            "total_threads": settings.threads,
            "coordinate_points": settings.foil_coordinate_points,
            "search_method": settings.search_method,
            "final_method": settings.final_method,
            "alpha_step_search_deg": settings.alpha_step_search_deg,
            "alpha_step_final_deg": settings.alpha_step_final_deg,
            "ncrit": settings.ncrit,
            "xtr_top": settings.xtr_top,
            "xtr_bottom": settings.xtr_bottom,
            "search_mesh": settings.search_mesh,
            "final_mesh": settings.final_mesh,
            "convergence_mesh": settings.convergence_mesh,
            "mesh_convergence_enabled": settings.mesh_convergence_enabled,
            "mesh_cd_tolerance_percent": settings.mesh_cd_tolerance_percent,
            "mesh_alpha_tolerance_deg": settings.mesh_alpha_tolerance_deg,
            "cancel_event": cancel_event,
            "optimizer": settings.wing_optimizer,
            "multi_section_geometry_enabled": settings.multi_section_geometry_enabled,
            "mid_chord_factor_bounds": settings.mid_chord_factor_bounds,
            "mid_twist_bounds": settings.mid_twist_bounds,
            "section_foils": section_foils,
            "section_foil_dat_texts": section_foil_dat_texts,
            "structural_settings": settings.structural_settings,
            "hydro_settings": settings.hydro_settings,
            "surrogate_settings": settings.surrogate_settings,
            "budget_escalation_settings": settings.budget_escalation_settings,
            "checkpoint_store": checkpoint_store,
        }
        if fixed_planar is None:
            planar_wing, planar_baseline, planar_meta, planar_response = (
                optimize_wing_with_flow5(
                    **common,
                    candidate_budget=settings.wing_candidate_budget,
                    seed=seed,
                    progress_callback=report,
                    initial_geometry=planar_copy(initial_geometry),
                    winglet_optimization_enabled=False,
                    checkpoint_key=checkpoint_key(
                        f"{checkpoint_label}-planar",
                        {**checkpoint_payload, "stage": "planar"},
                    ),
                )
            )
        else:
            planar_wing, planar_baseline, planar_meta, planar_response = fixed_planar
        if not run_winglet:
            planar_meta["winglet_comparison"] = {
                "enabled": settings.winglet_optimization_enabled,
                "performed": False,
                "selected": False,
                "selection": "planar",
                "reason": (
                    "deferred_until_post_coupling"
                    if settings.winglet_optimization_enabled
                    else "disabled_by_user"
                ),
                "execution_policy": "post_coupling_once",
                "included_in_coupled_loop": False,
            }
            return planar_wing, planar_baseline, planar_meta, planar_response

        if not settings.winglet_optimization_enabled:
            raise RuntimeError("Kapalı winglet aşaması çalıştırılamaz")
        if winglet_section_foils is None or winglet_section_foil_dat_texts is None:
            raise RuntimeError("Winglet aşaması için NACA kesiti ve DAT metni gerekli")

        planar_geometry = geometry_from_result(planar_wing)
        seed_winglet = (
            initial_geometry
            if initial_geometry is not None and initial_geometry.winglet_active
            else None
        )
        winglet_initial = WingGeometry(
            span=planar_geometry.span,
            root_chord=planar_geometry.root_chord,
            taper=planar_geometry.taper,
            sweep_deg=planar_geometry.sweep_deg,
            tip_twist_deg=planar_geometry.tip_twist_deg,
            alpha_deg=0.0,
            mid_chord_factor=planar_geometry.mid_chord_factor,
            mid_twist_deg=planar_geometry.mid_twist_deg,
            winglet_enabled=True,
            winglet_height=(
                seed_winglet.winglet_height
                if seed_winglet is not None
                else float(np.mean(settings.winglet_height_bounds))
            ),
            winglet_cant_deg=(
                seed_winglet.winglet_cant_deg
                if seed_winglet is not None
                else float(np.mean(settings.winglet_cant_bounds))
            ),
            winglet_toe_deg=(
                seed_winglet.winglet_toe_deg
                if seed_winglet is not None
                else float(np.mean(settings.winglet_toe_bounds))
            ),
            winglet_taper=(
                seed_winglet.winglet_taper
                if seed_winglet is not None
                else float(np.mean(settings.winglet_taper_bounds))
            ),
        )
        # Freeze the optimized planar planform so this stage varies only the
        # four winglet parameters. The reported delta is therefore a direct
        # same-planform, same-projected-span, same-target-lift comparison.
        winglet_common = {
            **common,
            "section_foils": winglet_section_foils,
            "section_foil_dat_texts": winglet_section_foil_dat_texts,
            "span_bounds": (planar_geometry.span, planar_geometry.span),
            "root_chord_bounds": (
                planar_geometry.root_chord,
                planar_geometry.root_chord,
            ),
            "taper_bounds": (planar_geometry.taper, planar_geometry.taper),
            "sweep_bounds": (
                planar_geometry.sweep_deg,
                planar_geometry.sweep_deg,
            ),
            "twist_bounds": (
                planar_geometry.tip_twist_deg,
                planar_geometry.tip_twist_deg,
            ),
            "mid_chord_factor_bounds": (
                planar_geometry.mid_chord_factor,
                planar_geometry.mid_chord_factor,
            ),
            "mid_twist_bounds": (
                planar_geometry.effective_mid_twist_deg,
                planar_geometry.effective_mid_twist_deg,
            ),
        }

        stage_map = {
            "wing_search": "winglet_search",
            "wing_budget": "winglet_budget",
            "wing_final": "winglet_final",
            "mesh_convergence": "winglet_convergence",
        }

        def winglet_report(event: dict[str, Any]) -> None:
            mapped_stage = stage_map.get(str(event.get("stage")), event.get("stage"))
            report(
                {
                    **event,
                    "stage": mapped_stage,
                    "message": "Winglet · " + str(event.get("message", "flow5 çalışıyor")),
                }
            )

        try:
            winglet_wing, winglet_baseline, winglet_meta, winglet_response = (
                optimize_wing_with_flow5(
                    **winglet_common,
                    candidate_budget=settings.winglet_candidate_budget,
                    seed=seed + 15401,
                    progress_callback=winglet_report,
                    initial_geometry=winglet_initial,
                    winglet_optimization_enabled=True,
                    winglet_height_bounds=settings.winglet_height_bounds,
                    winglet_cant_bounds=settings.winglet_cant_bounds,
                    winglet_toe_bounds=settings.winglet_toe_bounds,
                    winglet_taper_bounds=settings.winglet_taper_bounds,
                    checkpoint_key=checkpoint_key(
                        f"{checkpoint_label}-winglet",
                        {**checkpoint_payload, "stage": "winglet"},
                    ),
                )
            )
        except Flow5CancelledError:
            raise
        except Exception as exc:
            comparison = {
                "enabled": True,
                "performed": False,
                "selected": False,
                "selection": "planar",
                "reason": "winglet_stage_failed",
                "error": str(exc)[-800:],
                "planar": winglet_summary(planar_wing, planar_meta),
                "airfoil": winglet_section_foils[-1].to_dict(),
                "airfoil_name": winglet_section_foils[-1].name,
                "execution_policy": "post_coupling_once",
                "execution_count": 1,
                "included_in_coupled_loop": False,
                "model": "flow5 high-dihedral fourth section",
            }
            planar_meta["winglet_comparison"] = comparison
            return planar_wing, planar_baseline, planar_meta, planar_response

        planar_summary = winglet_summary(planar_wing, planar_meta)
        winglet_result_summary = winglet_summary(winglet_wing, winglet_meta)
        planar_feasible = bool(planar_meta["feasible"])
        winglet_feasible = bool(winglet_meta["feasible"])
        if winglet_feasible and not planar_feasible:
            select_winglet = True
            selection_reason = "yalnız winglet tasarımı fizibil"
        elif planar_feasible and not winglet_feasible:
            select_winglet = False
            selection_reason = "winglet tasarımı kısıtları geçemedi"
        else:
            select_winglet = float(winglet_meta["objective"]) < float(
                planar_meta["objective"]
            )
            selection_reason = "daha düşük kısıtlı toplam amaç"

        def percent_change(winglet_value: float, planar_value: float) -> float:
            return float(
                100.0
                * (float(winglet_value) - float(planar_value))
                / max(abs(float(planar_value)), 1.0e-12)
            )

        comparison = {
            "enabled": True,
            "performed": True,
            "selected": bool(select_winglet),
            "selection": "winglet" if select_winglet else "planar",
            "selection_reason": selection_reason,
            "same_projected_span_and_target_lift": True,
            "frozen_planar_geometry": True,
            "optimized_variables": ["height", "cant", "toe", "taper"],
            "airfoil": winglet_section_foils[-1].to_dict(),
            "airfoil_name": winglet_section_foils[-1].name,
            "airfoil_policy": "user-selected fixed NACA 4-digit section",
            "execution_policy": "post_coupling_once",
            "execution_count": 1,
            "included_in_coupled_loop": False,
            "planar": planar_summary,
            "winglet": winglet_result_summary,
            "delta_winglet_vs_planar": {
                "drag_percent": percent_change(
                    winglet_wing["drag_n"], planar_wing["drag_n"]
                ),
                "induced_cd_percent": percent_change(
                    winglet_wing["cd_induced"], planar_wing["cd_induced"]
                ),
                "profile_cd_percent": percent_change(
                    winglet_wing["cd_profile"], planar_wing["cd_profile"]
                ),
                "ld_percent": percent_change(winglet_wing["ld"], planar_wing["ld"]),
                "root_bending_moment_percent": percent_change(
                    winglet_wing["root_bending_moment_nm"],
                    planar_wing["root_bending_moment_nm"],
                ),
            },
            "model": "flow5 high-dihedral fourth section; fixed projected-span comparison",
            "limitations": [
                "junction fillet interference is not resolved by the preliminary panel model",
                "local winglet-junction stress still requires FEA",
                "water cases still require multiphase cavitation/ventilation CFD",
            ],
        }
        if select_winglet:
            selected = (winglet_wing, winglet_baseline, winglet_meta, winglet_response)
        else:
            selected = (planar_wing, planar_baseline, planar_meta, planar_response)
        selected_wing, selected_baseline, selected_meta, selected_response = selected
        planar_scalar, planar_scalar_response = scalar_stage_choice(
            "planar", planar_wing, planar_meta, planar_response
        )
        winglet_scalar, winglet_scalar_response = scalar_stage_choice(
            "winglet", winglet_wing, winglet_meta, winglet_response
        )
        if float(winglet_scalar.get("objective", math.inf)) < float(
            planar_scalar.get("objective", math.inf)
        ):
            overall_scalar = winglet_scalar
            overall_scalar_response = winglet_scalar_response
        else:
            overall_scalar = planar_scalar
            overall_scalar_response = planar_scalar_response
        scalar_wing = overall_scalar.get("wing") or overall_scalar.get("coarse_wing")
        same_as_selected = bool(
            isinstance(scalar_wing, dict)
            and same_wing_geometry(selected_wing, scalar_wing)
        )
        overall_scalar["same_as_feasibility_first"] = same_as_selected
        overall_scalar["selection_scope"] = "planar sonuç ve tek son işlem winglet aşaması"
        selected_comparison = deepcopy(
            selected_meta.get("selection_comparison", {})
        )
        selected_comparison["scalar_only"] = overall_scalar
        selected_comparison["definition"] = (
            "Planar sonuç ve tek son işlem winglet aşamasında fizibilite-öncelikli sonuç ile "
            "yalnız skaler toplam amacın seçeceği sonuç"
        )
        selected_meta["selection_comparison"] = selected_comparison
        selected_meta["winglet_comparison"] = comparison
        output_response = dict(selected_response)
        output_response.pop("_scalar_only_alternative_response", None)
        if not same_as_selected and overall_scalar_response is not None:
            output_response["_scalar_only_alternative_response"] = dict(
                overall_scalar_response
            )
        return selected_wing, selected_baseline, selected_meta, output_response

    if workflow_mode == "foil_only":
        foil, foil_response, foil_meta, selected_foil_dat_text = optimize_airfoil_with_flow5(
            runner=runner,
            baseline_profile=settings.baseline_profile,
            fluid=fluid,
            speeds_m_s=speeds,
            target_cls=target_cls,
            reference_chord_m=reference_chord_m,
            camber_bounds=camber_bounds,
            camber_position_bounds=camber_position_bounds,
            thickness_bounds=thickness_bounds,
            alpha_bounds=alpha_bounds,
            candidate_budget=settings.foil_candidate_budget,
            seed=settings.seed,
            total_threads=settings.threads,
            cst_order=settings.cst_order,
            coordinate_points=settings.foil_coordinate_points,
            minimum_improvement_percent=settings.foil_minimum_improvement_percent,
            alpha_step_search_deg=settings.alpha_step_search_deg,
            alpha_step_final_deg=settings.alpha_step_final_deg,
            ncrit=settings.ncrit,
            xtr_top=settings.xtr_top,
            xtr_bottom=settings.xtr_bottom,
            progress_callback=report,
            cancel_event=cancel_event,
            optimizer=settings.foil_optimizer,
            surrogate_settings=settings.surrogate_settings,
            budget_escalation_settings=settings.budget_escalation_settings,
            checkpoint_store=checkpoint_store,
            checkpoint_key=checkpoint_key(
                "foil-only",
                {
                    "reference_chord_m": reference_chord_m,
                    "target_cls": target_cls,
                    "baseline_identifier": settings.baseline_profile.identifier,
                    "baseline_dat_sha256": hashlib.sha256(
                        settings.baseline_profile.solver_dat_text.encode("utf-8")
                    ).hexdigest(),
                },
            ),
        )
        reference_polar = min(
            foil_response["polars"],
            key=lambda polar: abs(polar["speed_m_s"] - reference_speed_m_s),
        )
        reference_condition = min(
            foil_meta["conditions"],
            key=lambda item: abs(item["speed_m_s"] - reference_speed_m_s),
        )
        if foil_meta["selection"]["selected_baseline"]:
            x = np.asarray(settings.baseline_profile.solver_x, dtype=float)
            y = np.asarray(settings.baseline_profile.solver_y, dtype=float)
        else:
            x, y = airfoil_coordinates(
                foil, total_points=settings.foil_coordinate_points
            )
        selected_improvement = foil_meta.get("selection", {}).get(
            "selected_improvement_vs_baseline_percent"
        )
        result: dict[str, Any] = {
            "status": (
                "review"
                if foil_meta.get("budget_convergence", {}).get("converged") is False
                else "feasible"
            ),
            "workflow_mode": "foil_only",
            "flow5_native": True,
            "model": {
                "airfoil": (
                    f"{settings.baseline_profile.display_name} baseline + "
                    f"CST{settings.cst_order}/Kulfan; every objective value from "
                    "flow5 embedded XFoil"
                ),
                "wing": "Kanat aşaması kullanıcı seçimiyle atlandı",
                "scope": "flow5 multi-point 2D preliminary airfoil design",
            },
            "solver_run": {
                "strategy_requested": "flow5_native",
                "strategy_used": "flow5_native",
                "workflow_mode": "foil_only",
                "aerodynamic_score_source": "flow5 only",
                "flow5_threads": settings.threads,
                "speed_samples": speed_samples,
                "speed_samples_requested": requested_speed_samples,
                "foil_candidate_budget": settings.foil_candidate_budget,
                "foil_coordinate_points": settings.foil_coordinate_points,
                "baseline_airfoil": settings.baseline_profile.identifier,
                "cst_order": settings.cst_order,
                "foil_optimizer": settings.foil_optimizer,
                "wing_optimizer": "skipped",
                "foil_budget_convergence": foil_meta.get("budget_convergence", {}),
                "evaluation_cache": runner.cache_stats(),
                "optimizer_checkpoint": checkpoint_store.stats(),
                "surrogate": {"foil": foil_meta.get("surrogate", {})},
                "solver": foil_meta.get("solver", {}),
            },
            "flow": {
                **fluid.to_dict(),
                "speed_m_s": reference_speed_m_s,
                "speed_min_m_s": speed_bounds_m_s[0],
                "speed_max_m_s": speed_bounds_m_s[1],
                "speed_samples": speed_samples,
                "speed_samples_requested": requested_speed_samples,
                "sampled_speeds_m_s": speeds,
                "target_lift_n": target_lift_n,
                "dynamic_pressure_pa": fluid.dynamic_pressure(reference_speed_m_s),
                "mach": fluid.mach(reference_speed_m_s),
            },
            "airfoil": foil.to_dict(),
            "baseline_airfoil": foil_meta["baseline"],
            "airfoil_optimization": {
                **foil_meta,
                "design_cl_was_auto": design_cl_was_auto,
                "design_cl_at_reference": design_cl_at_reference,
                "target_cls": target_cls,
                "reynolds": reference_polar["reynolds"],
                "mach": reference_polar["mach"],
                "target_cl": reference_condition["target_cl"],
                "design_alpha_deg": reference_condition["point"]["alpha_deg"],
                "design_point": reference_condition["point"],
                "final_cruise_point": reference_condition["point"],
                "cl_max_estimate": reference_condition["cl_max_converged"],
            },
            "airfoil_coordinates": [
                {"x_over_c": float(xi), "y_over_c": float(yi)}
                for xi, yi in zip(x, y)
            ],
            "polar": reference_polar["points"],
            "foil_polars": foil_response["polars"],
            "polar_source": "flow5 embedded XFoil",
            "coupled_design": {
                "enabled": False,
                "iterations_requested": 0,
                "iterations_completed": 0,
                "history": [],
            },
            "spanwise_airfoil_optimization": {"enabled": False, "performed": False},
            "flow5_native_analysis": {
                "foil_solver": "flow5::XFoilTask",
                "foil_coordinate_points": settings.foil_coordinate_points,
                "foil_conditions": foil_meta["conditions"],
                "solver": foil_meta.get("solver", {}),
                "evaluation_cache": runner.cache_stats(),
                "optimizer_checkpoint": checkpoint_store.stats(),
                "surrogate": {"foil": foil_meta.get("surrogate", {})},
            },
            "insights": [
                {
                    "level": "good",
                    "title": "Profil optimizasyonu tamamlandı",
                    "text": (
                        f"{len(speeds)} akış noktasında seçilen profil doğrulandı; "
                        f"referans noktada CL={float(reference_condition['point']['cl']):.4f}, "
                        f"CD={float(reference_condition['point']['cd']):.5f}."
                    ),
                },
                {
                    "level": "info",
                    "title": "Kanat aşaması çalıştırılmadı",
                    "text": (
                        "Profil DAT dosyası kaydedildi. Arayüzde Yalnız kanat modunu "
                        "seçerek bu profille planform optimizasyonuna devam edebilirsiniz."
                    ),
                },
                {
                    "level": "good" if (selected_improvement or 0.0) > 0.0 else "info",
                    "title": "Baseline karşılaştırması tamamlandı",
                    "text": (
                        f"Seçilen profilin çok-noktalı amaç farkı baseline'a göre "
                        f"%{float(selected_improvement or 0.0):.2f}."
                    ),
                },
            ],
        }
        polar_text = xfoil_polar_csv(reference_polar["points"])
        snapshot = deepcopy(result)
        project_text = project_json(request, snapshot)
        result["exports"] = {
            "airfoil_filename": f"{foil.name}.dat",
            "airfoil_dat": selected_foil_dat_text,
            "project_filename": "aeropt-foil-project.json",
            "project_json": project_text,
            "xfoil_polar_filename": "flow5-xfoil-polar.csv",
            "xfoil_polar_csv": polar_text,
        }
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "completed",
                    "current": 1,
                    "total": 1,
                    "fraction": 1.0,
                    "percent": 100.0,
                    "message": "Profil optimizasyonu tamamlandı; kanat aşaması atlandı",
                }
            )
        return result

    iteration_profile = settings.baseline_profile
    iteration_reference_chord = reference_chord_m
    iteration_target_cls = list(target_cls)
    iteration_results: list[dict[str, Any]] = []
    previous_objective: float | None = None
    coupled_converged = effective_coupled_iterations == 1
    for iteration in range(effective_coupled_iterations):
        active_iteration = iteration
        iteration_seed = settings.seed + 1009 * iteration
        iteration_input_source = (
            ("estimated_seed" if design_cl_was_auto else "user_seed")
            if iteration == 0
            else "previous_wing"
        )
        iteration_reynolds = [
            float(fluid.reynolds(speed, iteration_reference_chord))
            for speed in speeds
        ]
        if workflow_mode == "wing_only":
            foil, foil_response, foil_meta, selected_foil_dat_text = (
                evaluate_fixed_airfoil_with_flow5(
                    runner=runner,
                    baseline_profile=iteration_profile,
                    fluid=fluid,
                    speeds_m_s=speeds,
                    target_cls=iteration_target_cls,
                    reference_chord_m=iteration_reference_chord,
                    alpha_bounds=alpha_bounds,
                    total_threads=settings.threads,
                    coordinate_points=settings.foil_coordinate_points,
                    alpha_step_final_deg=settings.alpha_step_final_deg,
                    ncrit=settings.ncrit,
                    xtr_top=settings.xtr_top,
                    xtr_bottom=settings.xtr_bottom,
                )
            )
            report(
                {
                    "stage": "foil_final",
                    "current": 1,
                    "total": 1,
                    "fraction": 1.0,
                    "message": "Seçilen profil doğrulandı; geometri değiştirilmedi",
                }
            )
        else:
            foil, foil_response, foil_meta, selected_foil_dat_text = optimize_airfoil_with_flow5(
                runner=runner,
                baseline_profile=iteration_profile,
                fluid=fluid,
                speeds_m_s=speeds,
                target_cls=iteration_target_cls,
                reference_chord_m=iteration_reference_chord,
                camber_bounds=camber_bounds,
                camber_position_bounds=camber_position_bounds,
                thickness_bounds=thickness_bounds,
                alpha_bounds=alpha_bounds,
                candidate_budget=settings.foil_candidate_budget,
                seed=iteration_seed,
                total_threads=settings.threads,
                cst_order=settings.cst_order,
                coordinate_points=settings.foil_coordinate_points,
                minimum_improvement_percent=settings.foil_minimum_improvement_percent,
                alpha_step_search_deg=settings.alpha_step_search_deg,
                alpha_step_final_deg=settings.alpha_step_final_deg,
                ncrit=settings.ncrit,
                xtr_top=settings.xtr_top,
                xtr_bottom=settings.xtr_bottom,
                progress_callback=report,
                cancel_event=cancel_event,
                optimizer=settings.foil_optimizer,
                surrogate_settings=settings.surrogate_settings,
                budget_escalation_settings=settings.budget_escalation_settings,
                checkpoint_store=checkpoint_store,
                checkpoint_key=checkpoint_key(
                    f"foil-coupled-{iteration + 1}",
                    {
                        "iteration_seed": iteration_seed,
                        "reference_chord_m": iteration_reference_chord,
                        "target_cls": iteration_target_cls,
                        "baseline_identifier": iteration_profile.identifier,
                        "baseline_dat_sha256": hashlib.sha256(
                            iteration_profile.solver_dat_text.encode("utf-8")
                        ).hexdigest(),
                    },
                ),
            )
        wing, baseline, wing_meta, wing_response = optimize_wing_stage(
            foil=foil,
            foil_dat_text=selected_foil_dat_text,
            seed=iteration_seed,
            checkpoint_label=f"wing-coupled-{iteration + 1}",
            checkpoint_payload={
                "iteration_seed": iteration_seed,
                "foil_dat_sha256": hashlib.sha256(
                    selected_foil_dat_text.encode("utf-8")
                ).hexdigest(),
                "span_bounds": span_bounds,
                "root_chord_bounds": root_chord_bounds,
                "taper_bounds": taper_bounds,
                "sweep_bounds": sweep_bounds,
                "twist_bounds": twist_bounds,
            },
        )
        # A manually entered Cl is an initial seed, not a lock.  Every solved
        # wing supplies the representative section loads and actual MAC for
        # the next foil pass.
        updated_cls = _representative_section_cls(wing)
        updated_chord = float(wing["geometry"]["mean_aerodynamic_chord"])
        updated_reynolds = [
            float(fluid.reynolds(speed, updated_chord)) for speed in speeds
        ]
        cl_change = max(
            (
                100.0 * abs(new - old) / max(abs(old), 1e-9)
                for new, old in zip(updated_cls, iteration_target_cls)
            ),
            default=0.0,
        )
        chord_change = float(
            100.0
            * abs(updated_chord - iteration_reference_chord)
            / max(abs(iteration_reference_chord), 1e-9)
        )
        reynolds_change = max(
            (
                100.0 * abs(new - old) / max(abs(old), 1e-9)
                for new, old in zip(updated_reynolds, iteration_reynolds)
            ),
            default=0.0,
        )
        objective = float(wing_meta["objective"])
        objective_change = (
            None
            if previous_objective is None
            else float(
                100.0
                * abs(objective - previous_objective)
                / max(abs(previous_objective), 1e-12)
            )
        )
        iteration_results.append(
            {
                "iteration": iteration + 1,
                "foil": foil,
                "foil_response": foil_response,
                "foil_meta": foil_meta,
                "foil_dat_text": selected_foil_dat_text,
                "baseline_profile": iteration_profile,
                "wing": wing,
                "rectangular_baseline": baseline,
                "wing_meta": wing_meta,
                "wing_response": wing_response,
                "input_source": iteration_input_source,
                "reference_chord_m": iteration_reference_chord,
                "reynolds": iteration_reynolds,
                "target_cls": list(iteration_target_cls),
                "next_reference_chord_m": updated_chord,
                "next_reynolds": updated_reynolds,
                "next_target_cls": updated_cls,
                "cl_schedule_change_percent": float(cl_change),
                "reference_chord_change_percent": chord_change,
                "reynolds_schedule_change_percent": float(reynolds_change),
                "objective_change_percent": objective_change,
            }
        )
        if (
            iteration > 0
            and cl_change <= settings.coupling_cl_tolerance_percent
            and reynolds_change <= settings.coupling_cl_tolerance_percent
            and objective_change is not None
            and objective_change <= settings.coupling_objective_tolerance_percent
        ):
            coupled_converged = True
            break
        if iteration + 1 >= effective_coupled_iterations:
            break
        previous_objective = objective
        iteration_reference_chord = updated_chord
        iteration_target_cls = updated_cls
        iteration_profile = build_derived_baseline_profile(
            foil,
            selected_foil_dat_text,
            identifier=f"coupled_iteration_{iteration + 1}",
            display_name=f"Bağlı iterasyon {iteration + 1} profili",
            solver_point_count=settings.foil_coordinate_points,
        )

    best_iteration = min(
        iteration_results,
        key=lambda item: (
            not bool(item["wing_meta"]["feasible"]),
            float(item["wing_meta"]["objective"]),
        ),
    )
    # The coupled result must be self-consistent: the final reported wing is
    # always the wing solved with the final reported foil.  Keep the best
    # historical objective as diagnostics, but never replace the final pair
    # with an earlier iteration.
    selected_iteration = iteration_results[-1]
    foil = selected_iteration["foil"]
    foil_response = selected_iteration["foil_response"]
    foil_meta = selected_iteration["foil_meta"]
    selected_foil_dat_text = selected_iteration["foil_dat_text"]
    baseline_profile = selected_iteration["baseline_profile"]
    wing = selected_iteration["wing"]
    baseline = selected_iteration["rectangular_baseline"]
    wing_meta = selected_iteration["wing_meta"]
    wing_response = selected_iteration["wing_response"]
    target_cls = selected_iteration["target_cls"]
    coupling_history = [
        {
            "iteration": int(item["iteration"]),
            "foil_name": item["foil"].name,
            "foil_objective": float(item["foil_meta"]["objective"]),
            "wing_objective": float(item["wing_meta"]["objective"]),
            "feasible": bool(item["wing_meta"]["feasible"]),
            "winglet_selection": "not_run_in_coupled_loop",
            "input_source": item["input_source"],
            "baseline_identifier": item["baseline_profile"].identifier,
            "baseline_display_name": item["baseline_profile"].display_name,
            "reference_chord_m": float(item["reference_chord_m"]),
            "reynolds": item["reynolds"],
            "target_cls": item["target_cls"],
            "next_reference_chord_m": float(item["next_reference_chord_m"]),
            "next_reynolds": item["next_reynolds"],
            "next_target_cls": item["next_target_cls"],
            "cl_schedule_change_percent": float(item["cl_schedule_change_percent"]),
            "reference_chord_change_percent": float(
                item["reference_chord_change_percent"]
            ),
            "reynolds_schedule_change_percent": float(
                item["reynolds_schedule_change_percent"]
            ),
            "objective_change_percent": item["objective_change_percent"],
            "feedback_applied": int(item["iteration"]) < len(iteration_results),
            "best_objective_iteration": item is best_iteration,
            "selected": item is selected_iteration,
        }
        for item in iteration_results
    ]
    reference_condition_index = min(
        range(len(speeds)), key=lambda index: abs(speeds[index] - reference_speed_m_s)
    )
    final_design_cl_at_reference = float(target_cls[reference_condition_index])

    section_foils: tuple[AirfoilLike, ...] | None = None
    section_foil_dat_texts: tuple[str, ...] | None = None
    spanwise_airfoil_meta: dict[str, Any] = {
        "enabled": effective_spanwise_optimization,
        "performed": False,
        "selected": False,
        "station_count": 1,
        "profiles": [{"station": "root", "name": foil.name, "eta": 0.0}],
    }
    if effective_spanwise_optimization:
        active_iteration = effective_coupled_iterations
        station_budget = max(
            8,
            int(round(settings.foil_candidate_budget * settings.spanwise_foil_budget_fraction)),
        )
        derived_profile = build_derived_baseline_profile(
            foil,
            selected_foil_dat_text,
            identifier="spanwise_root_seed",
            display_name="Bağlı tasarım kök profili",
            solver_point_count=settings.foil_coordinate_points,
        )
        optimized_sections: list[tuple[str, float, CSTAirfoilDesign, str, dict[str, Any]]] = []
        for station_index, (station_name, eta) in enumerate((("mid", 0.50), ("tip", 0.85))):
            station_cls = _span_station_cls(wing, eta)
            station_chord = float(geometry_from_result(wing).chord_at(eta))
            station_foil, _, station_meta, station_dat = optimize_airfoil_with_flow5(
                runner=runner,
                baseline_profile=derived_profile,
                fluid=fluid,
                speeds_m_s=speeds,
                target_cls=station_cls,
                reference_chord_m=station_chord,
                camber_bounds=camber_bounds,
                camber_position_bounds=camber_position_bounds,
                thickness_bounds=thickness_bounds,
                alpha_bounds=alpha_bounds,
                candidate_budget=station_budget,
                seed=settings.seed + 7001 + station_index * 379,
                total_threads=settings.threads,
                cst_order=settings.cst_order,
                coordinate_points=settings.foil_coordinate_points,
                minimum_improvement_percent=settings.foil_minimum_improvement_percent,
                alpha_step_search_deg=settings.alpha_step_search_deg,
                alpha_step_final_deg=settings.alpha_step_final_deg,
                ncrit=settings.ncrit,
                xtr_top=settings.xtr_top,
                xtr_bottom=settings.xtr_bottom,
                progress_callback=report,
                cancel_event=cancel_event,
                optimizer=settings.foil_optimizer,
                surrogate_settings=settings.surrogate_settings,
                budget_escalation_settings=settings.budget_escalation_settings,
                checkpoint_store=checkpoint_store,
                checkpoint_key=checkpoint_key(
                    f"foil-spanwise-{station_name}",
                    {
                        "seed": settings.seed + 7001 + station_index * 379,
                        "station": station_name,
                        "station_chord_m": station_chord,
                        "target_cls": station_cls,
                        "baseline_dat_sha256": hashlib.sha256(
                            derived_profile.solver_dat_text.encode("utf-8")
                        ).hexdigest(),
                    },
                ),
            )
            station_foil, station_dat = _rename_section_foil(
                station_foil, station_dat, station_name.capitalize()
            )
            optimized_sections.append(
                (station_name, eta, station_foil, station_dat, station_meta)
            )

        mid_foil = optimized_sections[0][2]
        tip_foil = optimized_sections[1][2]
        proposed_section_foils = (foil, mid_foil, tip_foil)
        proposed_section_dats = (
            selected_foil_dat_text,
            optimized_sections[0][3],
            optimized_sections[1][3],
        )
        initial_geometry = geometry_from_result(wing)
        refined_wing, refined_baseline, refined_meta, refined_response = optimize_wing_stage(
            foil=foil,
            foil_dat_text=selected_foil_dat_text,
            seed=settings.seed + 8093,
            section_foils=proposed_section_foils,
            section_foil_dat_texts=proposed_section_dats,
            initial_geometry=initial_geometry,
            checkpoint_label="wing-spanwise-refinement",
            checkpoint_payload={
                "seed": settings.seed + 8093,
                "foil_dat_sha256": [
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for text in proposed_section_dats
                ],
                "initial_geometry": initial_geometry.to_dict(),
            },
        )
        base_objective = float(wing_meta["objective"])
        refined_objective = float(refined_meta["objective"])
        accepted = bool(
            refined_meta["feasible"]
            and (
                not wing_meta["feasible"]
                or refined_objective
                <= base_objective
                * (1.0 + settings.spanwise_foil_acceptance_tolerance_percent / 100.0)
            )
        )
        spanwise_airfoil_meta = {
            "enabled": True,
            "performed": True,
            "selected": accepted,
            "station_count": 3,
            "station_candidate_budget": station_budget,
            "base_wing_objective": base_objective,
            "three_profile_wing_objective": refined_objective,
            "acceptance_tolerance_percent": settings.spanwise_foil_acceptance_tolerance_percent,
            "profiles": [
                {"station": "root", "eta": 0.0, "name": foil.name},
                *[
                    {
                        "station": item[0],
                        "eta": item[1],
                        "name": item[2].name,
                        "objective": float(item[4]["objective"]),
                        "target_cls": _span_station_cls(wing, item[1]),
                    }
                    for item in optimized_sections
                ],
            ],
        }
        if accepted:
            section_foils = proposed_section_foils
            section_foil_dat_texts = proposed_section_dats
            wing = refined_wing
            baseline = refined_baseline
            wing_meta = refined_meta
            wing_response = refined_response
        wing_meta["spanwise_airfoil_refinement"] = spanwise_airfoil_meta

    # The profile/wing loop is now complete.  Only at this point do we freeze
    # its planar result and run one independent winglet search with a fixed
    # NACA section.  Disabling the option skips this block entirely.
    winglet_foil = naca4_design(settings.winglet_naca_code)
    winglet_dat_text = airfoil_dat(
        winglet_foil, total_points=settings.foil_coordinate_points
    )
    main_section_foils = section_foils or (foil, foil, foil)
    main_section_dats = section_foil_dat_texts or (
        selected_foil_dat_text,
        selected_foil_dat_text,
        selected_foil_dat_text,
    )
    if settings.winglet_optimization_enabled:
        active_iteration = (
            effective_coupled_iterations + int(effective_spanwise_optimization)
        )
        planar_result = (wing, baseline, wing_meta, wing_response)
        wing, baseline, wing_meta, wing_response = optimize_wing_stage(
            foil=foil,
            foil_dat_text=selected_foil_dat_text,
            seed=settings.seed + 12007,
            checkpoint_label="winglet-post-coupling",
            checkpoint_payload={
                "seed": settings.seed + 12007,
                "planar_geometry": geometry_from_result(wing).to_dict(),
                "winglet_naca_code": settings.winglet_naca_code,
                "main_airfoils": [profile.name for profile in main_section_foils],
            },
            initial_geometry=geometry_from_result(wing),
            section_foils=section_foils,
            section_foil_dat_texts=section_foil_dat_texts,
            run_winglet=True,
            fixed_planar=planar_result,
            winglet_section_foils=(*main_section_foils, winglet_foil),
            winglet_section_foil_dat_texts=(*main_section_dats, winglet_dat_text),
        )
        if wing_meta.get("winglet_comparison", {}).get("selection") == "winglet":
            section_foils = (*main_section_foils, winglet_foil)
            section_foil_dat_texts = (*main_section_dats, winglet_dat_text)
    else:
        wing_meta["winglet_comparison"] = {
            "enabled": False,
            "performed": False,
            "selected": False,
            "selection": "planar",
            "reason": "disabled_by_user",
            "airfoil": winglet_foil.to_dict(),
            "airfoil_name": winglet_foil.name,
            "execution_policy": "post_coupling_once",
            "execution_count": 0,
            "included_in_coupled_loop": False,
        }

    reference_polar = min(
        foil_response["polars"], key=lambda polar: abs(polar["speed_m_s"] - reference_speed_m_s)
    )
    reference_foil_condition = min(
        foil_meta["conditions"],
        key=lambda item: abs(item["speed_m_s"] - reference_speed_m_s),
    )
    if foil_meta["selection"]["selected_baseline"]:
        x = np.asarray(baseline_profile.solver_x, dtype=float)
        y = np.asarray(baseline_profile.solver_y, dtype=float)
    else:
        x, y = airfoil_coordinates(foil, total_points=settings.foil_coordinate_points)
    winglet_comparison = wing_meta.get(
        "winglet_comparison",
        {
            "enabled": False,
            "performed": False,
            "selected": False,
            "selection": "planar",
        },
    )
    result: dict[str, Any] = {
        "status": (
            "feasible"
            if wing_meta["feasible"]
            and (effective_coupled_iterations <= 1 or coupled_converged)
            and foil_meta.get("budget_convergence", {}).get("converged") is not False
            and wing_meta.get("budget_convergence", {}).get("converged") is not False
            else "review"
        ),
        "workflow_mode": workflow_mode,
        "flow5_native": True,
        "model": {
            "airfoil": (
                (
                    f"{baseline_profile.display_name} fixed DAT geometry; flow5 embedded XFoil validation"
                    if workflow_mode == "wing_only"
                    else f"{baseline_profile.display_name} baseline + CST{settings.cst_order}/Kulfan geometry; "
                    "every objective value from flow5 embedded XFoil"
                )
            ),
            "wing": (
                f"flow5 {wing_meta['search_method']} search + {wing_meta['final_method']} final; "
                + (
                    "bağlı döngü sonrası tek seferlik planar/winglet karşılaştırması; "
                    if winglet_comparison.get("performed")
                    else ""
                )
                + "viscous on-the-fly"
            ),
            "scope": "flow5 potential-flow preliminary design over multiple operating points",
        },
        "solver_run": {
            "strategy_requested": "flow5_native",
            "strategy_used": "flow5_native",
            "aerodynamic_score_source": "flow5 only",
            "flow5_threads": settings.threads,
            "speed_samples": speed_samples,
            "speed_samples_requested": requested_speed_samples,
            "foil_candidate_budget": settings.foil_candidate_budget,
            "foil_coordinate_points": settings.foil_coordinate_points,
            "baseline_airfoil": settings.baseline_profile.identifier,
            "final_iteration_baseline_airfoil": baseline_profile.identifier,
            "cst_order": settings.cst_order,
            "wing_candidate_budget": settings.wing_candidate_budget,
            "winglet_optimization_enabled": settings.winglet_optimization_enabled,
            "winglet_naca_code": settings.winglet_naca_code,
            "winglet_candidate_budget": settings.winglet_candidate_budget,
            "winglet_selection": winglet_comparison.get("selection", "planar"),
            "winglet_execution_policy": "post_coupling_once",
            "winglet_execution_count": int(
                winglet_comparison.get("execution_count", 0)
            ),
            "budget_escalation": settings.budget_escalation_settings.to_dict(),
            "foil_budget_convergence": foil_meta.get("budget_convergence", {}),
            "wing_budget_convergence": wing_meta.get("budget_convergence", {}),
            "search_method": settings.search_method.upper(),
            "final_method": settings.final_method.upper(),
            "mesh_convergence_enabled": settings.mesh_convergence_enabled,
            "output_mesh": wing_meta["output_mesh"],
            "foil_optimizer": (
                "skipped_fixed_airfoil"
                if workflow_mode == "wing_only"
                else settings.foil_optimizer
            ),
            "wing_optimizer": settings.wing_optimizer,
            "workflow_mode": workflow_mode,
            "coupled_iterations_requested": effective_coupled_iterations,
            "coupled_iterations_completed": len(coupling_history),
            "selected_coupled_iteration": int(selected_iteration["iteration"]),
            "best_objective_coupled_iteration": int(best_iteration["iteration"]),
            "solver": wing_meta.get("solver", {}),
            "evaluation_cache": runner.cache_stats(),
            "optimizer_checkpoint": checkpoint_store.stats(),
            "surrogate": {
                "foil": foil_meta.get("surrogate", {}),
                "wing": wing_meta.get("surrogate", {}),
            },
        },
        "flow": {
            **fluid.to_dict(),
            "speed_m_s": reference_speed_m_s,
            "speed_min_m_s": speed_bounds_m_s[0],
            "speed_max_m_s": speed_bounds_m_s[1],
            "speed_samples": speed_samples,
            "speed_samples_requested": requested_speed_samples,
            "sampled_speeds_m_s": speeds,
            "target_lift_n": target_lift_n,
            "dynamic_pressure_pa": fluid.dynamic_pressure(reference_speed_m_s),
            "mach": fluid.mach(reference_speed_m_s),
        },
        "airfoil": foil.to_dict(),
        "initial_baseline_airfoil": settings.baseline_profile.to_dict(),
        "baseline_airfoil": foil_meta["baseline"],
        "airfoil_optimization": {
            **foil_meta,
            "design_cl_was_auto": design_cl_was_auto,
            "design_cl_seed_only": workflow_mode == "coupled",
            "initial_design_cl_at_reference": design_cl_at_reference,
            "design_cl_at_reference": final_design_cl_at_reference,
            "target_cls": target_cls,
            "reynolds": reference_polar["reynolds"],
            "mach": reference_polar["mach"],
            "target_cl": reference_foil_condition["target_cl"],
            "design_alpha_deg": reference_foil_condition["point"]["alpha_deg"],
            "design_point": reference_foil_condition["point"],
            "final_cruise_point": reference_foil_condition["point"],
            "cl_max_estimate": reference_foil_condition["cl_max_converged"],
        },
        "airfoil_coordinates": [
            {"x_over_c": float(xi), "y_over_c": float(yi)} for xi, yi in zip(x, y)
        ],
        "polar": reference_polar["points"],
        "foil_polars": foil_response["polars"],
        "polar_source": "flow5 embedded XFoil",
        "wing": wing,
        "wing_cases": wing["conditions"],
        "rectangular_baseline": baseline,
        "wing_optimization": wing_meta,
        "winglet_comparison": winglet_comparison,
        "structural_analysis": wing.get(
            "structural",
            {"enabled": False, "performed": False, "passed": True},
        ),
        "hydro_analysis": wing.get(
            "hydro",
            {"enabled": False, "performed": False, "passed": True},
        ),
        "coupled_design": {
            "enabled": effective_coupled_iterations > 1,
            "converged": bool(coupled_converged),
            "iterations_requested": effective_coupled_iterations,
            "iterations_completed": len(coupling_history),
            "selected_iteration": int(selected_iteration["iteration"]),
            "best_objective_iteration": int(best_iteration["iteration"]),
            "feedback_cycles_completed": max(len(coupling_history) - 1, 0),
            "cl_tolerance_percent": settings.coupling_cl_tolerance_percent,
            "objective_tolerance_percent": settings.coupling_objective_tolerance_percent,
            "initial_design_cl_source": (
                "estimated" if design_cl_was_auto else "user"
            ),
            "initial_design_cl_at_reference": design_cl_at_reference,
            "final_design_cl_at_reference": final_design_cl_at_reference,
            "manual_design_cl_is_seed_only": workflow_mode == "coupled",
            "explicit_design_cl_locked": False,
            "winglet_included": False,
            "winglet_policy": "excluded; optional post-coupling stage runs once",
            "history": coupling_history,
        },
        "spanwise_airfoil_optimization": spanwise_airfoil_meta,
        "pareto_analysis": wing_meta.get("pareto_analysis", {}),
        "flow5_native_analysis": {
            "foil_solver": "flow5::XFoilTask",
            "foil_coordinate_points": settings.foil_coordinate_points,
            "wing_solver_search": wing_meta["search_method"],
            "wing_solver_final": wing_meta["final_method"],
            "viscous_drag": "flow5 on-the-fly embedded XFoil",
            "solver": wing_meta.get("solver", {}),
            "foil_conditions": foil_meta["conditions"],
            "wing_conditions": wing_meta["conditions"],
            "solver_telemetry": wing_meta["solver_telemetry"],
            "mesh_convergence": wing_meta["mesh_convergence"],
            "evaluation_cache": runner.cache_stats(),
            "optimizer_checkpoint": checkpoint_store.stats(),
            "surrogate": {
                "foil": foil_meta.get("surrogate", {}),
                "wing": wing_meta.get("surrogate", {}),
            },
            "spanwise_airfoils": spanwise_airfoil_meta,
            "winglet_comparison": winglet_comparison,
            "winglet_execution_policy": "post_coupling_once",
            "winglet_naca_airfoil": winglet_foil.to_dict(),
        },
    }
    result["insights"] = _native_insights(
        wing=wing,
        wing_meta=wing_meta,
        foil_meta=foil_meta,
        fluid_key=fluid_key,
        span_upper=span_bounds[1],
        coupled_design=result["coupled_design"],
    )

    geometry = geometry_from_result(wing)
    foil_text = selected_foil_dat_text
    output_mesh = wing_meta["output_mesh"]
    plane_text = flow5_plane_xml(
        foil,
        geometry,
        chordwise_panels=int(output_mesh["chordwise_panels"]),
        half_span_panels=int(output_mesh["half_span_panels"]),
        section_foils=section_foils,
    )
    analysis_text = flow5_analysis_xml(
        geometry,
        fluid,
        reference_speed_m_s,
        settings.final_method,
        ncrit=settings.ncrit,
        xtr_top=settings.xtr_top,
        xtr_bottom=settings.xtr_bottom,
    )
    obj_text = wing_obj(foil, geometry, section_foils=section_foils)
    results_text = flow5_native_results_csv(foil, wing)
    polar_text = xfoil_polar_csv(reference_polar["points"])
    project_payload = wing_response.get("artifact_payloads", {}).get("project_fl5")
    if not project_payload:
        raise RuntimeError("flow5 son analizi tamamladı ancak çözümlenmiş .fl5 proje artifact'i üretmedi")
    flow5_project_bytes = base64.b64decode(project_payload["base64"])
    step_payload = wing_response.get("artifact_payloads", {}).get("wing_step")
    if not step_payload:
        raise RuntimeError(
            "flow5 son analizi tamamladı ancak OpenCascade STEP katısı üretmedi"
        )
    wing_step_bytes = base64.b64decode(step_payload["base64"])
    if not wing_step_bytes.lstrip().startswith(b"ISO-10303-21;") or (
        b"END-ISO-10303-21;" not in wing_step_bytes
    ):
        raise RuntimeError("flow5 runner geçerli bir ISO 10303 STEP dosyası döndürmedi")
    if b"SI_UNIT($,.METRE.)" not in wing_step_bytes.replace(b" ", b""):
        raise RuntimeError("flow5 runner STEP uzunluk birimini metre olarak doğrulamadı")

    scalar_selection = wing_meta.get("selection_comparison", {}).get(
        "scalar_only", {}
    )
    scalar_same = bool(scalar_selection.get("same_as_feasibility_first", False))
    scalar_wing = scalar_selection.get("wing")
    scalar_obj_text: str | None = None
    scalar_results_text: str | None = None
    scalar_step_bytes: bytes | None = None
    scalar_project_bytes: bytes | None = None
    scalar_project_payload: dict[str, Any] | None = None
    scalar_step_payload: dict[str, Any] | None = None
    if not scalar_same and isinstance(scalar_wing, dict):
        scalar_response = wing_response.get("_scalar_only_alternative_response")
        if not isinstance(scalar_response, dict):
            raise RuntimeError(
                "Skaler-puan alternatifi çözüldü ancak son flow5 yanıtı taşınmadı"
            )
        scalar_artifacts = scalar_response.get("artifact_payloads", {})
        scalar_project_payload = scalar_artifacts.get("project_fl5")
        scalar_step_payload = scalar_artifacts.get("wing_step")
        if not scalar_project_payload or not scalar_step_payload:
            raise RuntimeError(
                "Skaler-puan alternatifi için .fl5 ve STEP çıktıları birlikte üretilemedi"
            )
        scalar_project_bytes = base64.b64decode(scalar_project_payload["base64"])
        scalar_step_bytes = base64.b64decode(scalar_step_payload["base64"])
        if not scalar_step_bytes.lstrip().startswith(b"ISO-10303-21;") or (
            b"END-ISO-10303-21;" not in scalar_step_bytes
        ):
            raise RuntimeError(
                "Skaler-puan alternatifi geçerli bir ISO 10303 STEP dosyası döndürmedi"
            )
        if b"SI_UNIT($,.METRE.)" not in scalar_step_bytes.replace(b" ", b""):
            raise RuntimeError(
                "Skaler-puan alternatifi STEP uzunluk birimini metre olarak doğrulamadı"
            )
        scalar_geometry = geometry_from_result(scalar_wing)
        scalar_section_foils = (
            (*main_section_foils, winglet_foil)
            if scalar_geometry.winglet_active
            else section_foils
        )
        scalar_obj_text = wing_obj(
            foil,
            scalar_geometry,
            section_foils=scalar_section_foils,
        )
        scalar_results_text = flow5_native_results_csv(foil, scalar_wing)
    snapshot = deepcopy(result)
    project_text = project_json(request, snapshot)
    bundle = flow5_bundle_bytes(
        foil_dat_text=foil_text,
        plane_xml_text=plane_text,
        wing_obj_text=obj_text,
        results_csv_text=results_text,
        project_json_text=project_text,
        polar_csv_text=polar_text,
        analysis_xml_text=analysis_text,
        flow5_project_bytes=flow5_project_bytes,
        wing_step_bytes=wing_step_bytes,
        section_foil_dat_texts=section_foil_dat_texts,
        scalar_only_wing_obj_text=scalar_obj_text,
        scalar_only_wing_step_bytes=scalar_step_bytes,
        scalar_only_results_csv_text=scalar_results_text,
        scalar_only_flow5_project_bytes=scalar_project_bytes,
        scalar_only_same_as_primary=scalar_same,
    )
    result["exports"] = {
        "airfoil_filename": f"{foil.name}.dat",
        "airfoil_dat": foil_text,
        "plane_filename": "aeropt-wing.xml",
        "plane_xml": plane_text,
        "analysis_filename": "aeropt-analysis.xml",
        "analysis_xml": analysis_text,
        "wing_obj_filename": "aeropt-wing.obj",
        "wing_obj": obj_text,
        "wing_step_filename": "aeropt-wing.step",
        "wing_step_base64": step_payload["base64"],
        "results_filename": "aeropt-flow5-results.csv",
        "results_csv": results_text,
        "project_filename": "aeropt-project.json",
        "project_json": project_text,
        "xfoil_polar_filename": "flow5-xfoil-polar.csv",
        "xfoil_polar_csv": polar_text,
        "flow5_project_filename": "aeropt-optimized.fl5",
        "flow5_project_base64": project_payload["base64"],
        "flow5_bundle_filename": "aeropt-flow5-native-package.zip",
        "flow5_bundle_base64": base64.b64encode(bundle).decode("ascii"),
        "scalar_only_alternative": {
            "available": bool(scalar_selection.get("available", False)),
            "deliverable": bool(scalar_selection.get("deliverable", False)),
            "same_as_feasibility_first": scalar_same,
            "stage": scalar_selection.get("stage"),
            "error": scalar_selection.get("error"),
            "wing_obj_filename": (
                "aeropt-scalar-only-wing.obj" if scalar_obj_text else None
            ),
            "wing_obj": scalar_obj_text,
            "wing_step_filename": (
                "aeropt-scalar-only-wing.step" if scalar_step_payload else None
            ),
            "wing_step_base64": (
                scalar_step_payload.get("base64") if scalar_step_payload else None
            ),
            "results_filename": (
                "aeropt-scalar-only-results.csv" if scalar_results_text else None
            ),
            "results_csv": scalar_results_text,
            "flow5_project_filename": (
                "aeropt-scalar-only-optimized.fl5"
                if scalar_project_payload
                else None
            ),
            "flow5_project_base64": (
                scalar_project_payload.get("base64")
                if scalar_project_payload
                else None
            ),
        },
        "section_airfoils": (
            [
                {
                    "station": station,
                    "filename": f"aeropt-airfoil-{station}.dat",
                    "airfoil_dat": dat_text,
                }
                for station, dat_text in zip(
                    ("root", "mid", "tip", "winglet"), section_foil_dat_texts
                )
            ]
            if section_foil_dat_texts is not None
            else []
        ),
    }
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "completed",
                "current": 1,
                "total": 1,
                "fraction": 1.0,
                "percent": 100.0,
                "message": "Optimizasyon ve flow5 paketi tamamlandı",
            }
        )
    return result
