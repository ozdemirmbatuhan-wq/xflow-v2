from __future__ import annotations

from typing import Any


def _entry(
    code: str,
    severity: str,
    title: str,
    evidence: str,
    recommendation: str,
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def diagnose_runtime_failure(error: Exception | str) -> dict[str, str]:
    """Classify a failed run into one actionable, evidence-based cause."""
    message = str(error).strip() or "Bilinmeyen çalışma zamanı hatası"
    normalized = message.casefold()
    rules = (
        (
            ("runner bulunamad", "no such file", "başlatılamadı"),
            "runner_missing",
            "flow5 runner bulunamadı veya başlatılamadı",
            "GitHub Actions ile üretilen tam Windows paketini kullanın ya da kaynak kullanımında aeropt-flow5-runner yolunu doğrulayın.",
        ),
        (
            ("sürümü uyumsuz", "beklenen 7.57", "version"),
            "runner_version",
            "flow5 API sürümü uyumsuz",
            "Yalnız flow5 7.57 için derlenmiş runner'ı kullanın; farklı flow5 sürümündeki ikiliyi karıştırmayın.",
        ),
        (
            ("zaman aş", "timeout", "timed out"),
            "solver_timeout",
            "flow5 değerlendirmesi zaman aşımına uğradı",
            "Aday zaman aşımını artırın; ilk teşhis için mesh, alfa aralığı ve aday bütçesini küçültün.",
        ),
        (
            ("hedef cl", "polar", "geçerli finalist", "yakınsam"),
            "polar_coverage",
            "Hedef CL polar/yakınsama aralığının dışında kaldı",
            "Alfa aralığını genişletin, düşük hız hedef yükünü azaltın veya kanat/profil zarfını büyütün.",
        ),
        (
            ("out_of_mesh", "mesh dış", "panel ağı"),
            "mesh_failure",
            "Kanat çözümü mesh veya viskoz polar kapsamı dışında kaldı",
            "Arama/final panel sayılarını ve profil polar alfa aralığını kontrol edip daha küçük bir geometri zarfıyla tekrar deneyin.",
        ),
        (
            (".fl5", "project", "artifact"),
            "project_artifact",
            "flow5 çözümlenmiş proje artifact'i üretilemedi",
            "Runner'ın flow5 7.57 runtime dosyalarıyla aynı pakette olduğunu ve çıktı klasörüne yazma izni bulunduğunu doğrulayın.",
        ),
        (
            ("hiçbir seed", "all seed"),
            "all_seeds_failed",
            "Bütün bağımsız optimizer koşuları başarısız oldu",
            "Önce tek seed ve 8/8 bütçeyle kök hatayı bulun; ardından multi-seed'i yeniden açın.",
        ),
    )
    for patterns, code, title, recommendation in rules:
        if any(pattern in normalized for pattern in patterns):
            return {
                "code": code,
                "title": title,
                "evidence": message[-800:],
                "recommendation": recommendation,
                "model": "deterministic runtime error rules",
            }
    return {
        "code": "unclassified_runtime",
        "title": "Sınıflandırılamayan flow5/optimizer hatası",
        "evidence": message[-800:],
        "recommendation": "Önce 8/8 bütçe, tek seed ve varsayılan mesh ile tekrarlayın; hata sürerse runner log kuyruğu ve istek JSON'unu inceleyin.",
        "model": "deterministic runtime error rules",
    }


def build_diagnostic_report(
    result: dict[str, Any], bounds: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    diagnoses: list[dict[str, str]] = []
    wing = result["wing"]
    geometry = wing["geometry"]
    conditions = result.get("wing_cases", [])
    worst_stall = max((float(item.get("stall_ratio", 0.0)) for item in conditions), default=0.0)
    if worst_stall > 0.92:
        diagnoses.append(
            _entry(
                "stall_margin",
                "critical" if worst_stall > 1.0 else "warning",
                "Stall/polar marjı dar",
                f"En kötü hedef CL / yakınsayan CLmax = {worst_stall:.3f}.",
                "Düşük hız hedefini, kanat alanını veya profil camber/kalınlık zarfını gözden geçirin.",
            )
        )
    reynolds_values = [
        float(item.get("reynolds", 0.0))
        for item in result.get("airfoil_optimization", {}).get("conditions", [])
        if item.get("reynolds") is not None
    ]
    if reynolds_values and min(reynolds_values) < 180_000:
        diagnoses.append(
            _entry(
                "low_reynolds",
                "warning",
                "Düşük Reynolds hassasiyeti",
                f"Minimum profil Reynolds sayısı {min(reynolds_values):.0f}.",
                "Ncrit, yüzey pürüzlülüğü ve zorlanmış geçiş için ayrı koşularla robustluğu kontrol edin.",
            )
        )
    cd_total = float(wing.get("cd_total", 0.0))
    induced_share = float(wing.get("cd_induced", 0.0)) / max(cd_total, 1e-12)
    profile_share = float(wing.get("cd_profile", 0.0)) / max(cd_total, 1e-12)
    if induced_share > 0.58:
        winglet = result.get("winglet_comparison", {})
        recommendation = (
            "Planar/winglet karşılaştırmasını inceleyin; ardından açıklık, taper ve twist sınırlarını kontrollü genişletin."
            if winglet.get("performed")
            else "Açıklık sınırlıysa winglet karşılaştırmasını açın; ayrıca taper–twist yük dağılımını genişletmeyi değerlendirin."
        )
        diagnoses.append(
            _entry(
                "induced_drag_dominant",
                "info",
                "İndüklenmiş sürükleme baskın",
                f"CDi/toplam CD payı %{100.0 * induced_share:.1f}.",
                recommendation,
            )
        )
    if profile_share > 0.68:
        diagnoses.append(
            _entry(
                "profile_drag_dominant",
                "info",
                "Viskoz profil sürüklemesi baskın",
                f"CDv/toplam CD payı %{100.0 * profile_share:.1f}.",
                "Profil zarfı, Reynolds aralığı, Ncrit ve geçiş konumlarını inceleyin.",
            )
        )
    mesh = result.get("wing_optimization", {}).get("mesh_convergence", {})
    if mesh.get("enabled") and not mesh.get("passed"):
        if mesh.get("fine_mesh_valid") is False:
            mesh_evidence = str(
                mesh.get("reason")
                or "İnce ağ hedef taşıma noktasında geçerli polar üretmedi"
            )
        else:
            mesh_evidence = (
                f"ΔCD %{float(mesh.get('max_cd_change_percent', 0.0)):.2f}, "
                f"Δα {float(mesh.get('max_alpha_change_deg', 0.0)):.3f}°."
            )
        diagnoses.append(
            _entry(
                "mesh_not_converged",
                "critical",
                "Panel ağı yakınsamadı",
                mesh_evidence,
                "Final/ince panel sayılarını artırın veya geometri/polar yakınsama sorununu giderin.",
            )
        )
    telemetry = result.get("wing_optimization", {}).get("solver_telemetry", {})
    failed_points = int(telemetry.get("out_of_mesh_points", 0)) + int(
        telemetry.get("nonconverged_viscous_points", 0)
    )
    if failed_points:
        diagnoses.append(
            _entry(
                "solver_point_failure",
                "critical",
                "flow5 çalışma noktası başarısızlığı",
                f"Mesh dışı/viskoz yakınsamayan toplam {failed_points} nokta var.",
                "Alfa aralığını, panel ağını ve profil polar kapsamasını düzeltin.",
            )
        )
    for key, label in (
        ("span", "Açıklık"),
        ("root_chord", "Kök chord"),
        ("taper", "Taper"),
        ("sweep_deg", "Sweep"),
        ("tip_twist_deg", "Uç twist"),
        ("winglet_height", "Winglet yüksekliği"),
        ("winglet_cant_deg", "Winglet cant"),
        ("winglet_toe_deg", "Winglet toe"),
        ("winglet_taper", "Winglet taper"),
    ):
        if key.startswith("winglet_") and not geometry.get("winglet_enabled", False):
            continue
        bound = bounds.get(key)
        if not bound:
            continue
        value = float(geometry[key])
        width = max(bound[1] - bound[0], 1e-12)
        distance = min(abs(value - bound[0]), abs(bound[1] - value)) / width
        if distance <= 0.02:
            edge = "alt" if abs(value - bound[0]) <= abs(bound[1] - value) else "üst"
            diagnoses.append(
                _entry(
                    f"boundary_{key}",
                    "warning",
                    f"{label} {edge} sınıra dayandı",
                    f"Seçilen {value:.4g}; izin verilen aralık {bound[0]:.4g}–{bound[1]:.4g}.",
                    "Gerçek optimumun dışarıda olup olmadığını görmek için bu sınırı kontrollü genişletin.",
                )
            )
    coupled = result.get("coupled_design", {})
    if coupled.get("enabled") and not coupled.get("converged"):
        diagnoses.append(
            _entry(
                "coupling_not_converged",
                "warning",
                "Foil–kanat döngüsü yakınsamadı",
                f"{coupled.get('iterations_completed', 0)}/{coupled.get('iterations_requested', 0)} iterasyon tamamlandı.",
                "İterasyon sayısını artırın veya Cl/amaç toleranslarını fiziksel hassasiyete göre düzenleyin.",
            )
        )
    stability = result.get("multi_seed_stability", {})
    if stability.get("enabled") and stability.get("stable") is False:
        diagnoses.append(
            _entry(
                "seed_instability",
                "warning",
                "Optimizer seed hassasiyeti yüksek",
                f"Amaç CV %{float(stability.get('objective_cv_percent', 0.0)):.2f}.",
                "Aday bütçesini artırın, Pareto cephesini inceleyin veya daha fazla seed çalıştırın.",
            )
        )
    for stage, label in (("airfoil_optimization", "Profil"), ("wing_optimization", "Kanat")):
        budget = result.get(stage, {}).get("budget_convergence", {})
        if budget.get("converged") is False:
            checkpoints = budget.get("checkpoints") or [{}]
            movement = checkpoints[-1].get("controlling_change_percent")
            diagnoses.append(
                _entry(
                    f"budget_not_converged_{stage}",
                    "warning",
                    f"{label} optimizer bütçesi yakınsamadı",
                    (
                        f"{budget.get('evaluations_completed', 0)}/{budget.get('maximum_budget', 0)} "
                        f"adayda son izlenen değişim %{float(movement or 0.0):.3f}."
                    ),
                    "Azami bütçe çarpanını artırın veya bağımsız seed koşularıyla cephe kararlılığını doğrulayın.",
                )
            )
    validation = result.get("validation_report", {})
    if validation.get("enabled") and not validation.get("passed"):
        diagnoses.append(
            _entry(
                "validation_contract",
                "critical",
                "Doğrulama sözleşmesi geçmedi",
                f"{validation.get('checks_passed', 0)}/{validation.get('checks_total', 0)} kontrol geçti.",
                "Başarısız doğrulama satırlarını düzeltmeden aerodinamik optimumu kullanmayın.",
            )
        )
    structure = result.get("structural_analysis", {})
    if structure.get("enabled") and not structure.get("passed"):
        diagnoses.append(
            _entry(
                "structure_limit",
                "critical",
                "Yapısal ön tarama sınır aştı",
                "Gerilme, sehim veya burulma kullanımlarından en az biri 1.0 üzerinde.",
                "Spar/skin boyutlarını veya kanat yük/uzunluk zarfını değiştirin ve FEA ile doğrulayın.",
            )
        )
    hydro = result.get("hydro_analysis", {})
    if hydro.get("enabled") and not hydro.get("passed"):
        report_only = hydro.get("constraint_mode") == "report_only"
        diagnoses.append(
            _entry(
                "cavitation_limit",
                "warning" if report_only else "critical",
                (
                    "Kavitasyon riski raporlandı"
                    if report_only
                    else "Kavitasyon marjı yetersiz"
                ),
                (
                    f"Kavitasyon kullanımı {float(hydro.get('cavitation_utilization', 0.0)):.3f}; "
                    f"riskli finalist alanı %{float(hydro.get('risk_area_percent', 0.0)):.2f}."
                ),
                (
                    "Bu koşu yalnız rapor modunda; L/D seçimi değiştirilmedi. "
                    "Hız, batma derinliği ve profil basınç dağılımını haritadan inceleyin."
                    if report_only
                    else "Hızı/yüklemeyi düşürün, batma derinliğini veya profil basınç dağılımını değiştirin."
                ),
            )
        )
    if float(wing.get("ld", 0.0)) < 15.0 and not diagnoses:
        diagnoses.append(
            _entry(
                "low_ld_general",
                "warning",
                "3B L/D beklenenden düşük",
                f"Referans L/D = {float(wing.get('ld', 0.0)):.1f}.",
                "CDi/CDv ayrımını, hedef CL'yi ve hız aralığını birlikte inceleyin.",
            )
        )
    rank = {"critical": 0, "warning": 1, "info": 2}
    diagnoses.sort(key=lambda item: rank.get(item["severity"], 9))
    return {
        "status": (
            "critical"
            if any(item["severity"] == "critical" for item in diagnoses)
            else "warning" if any(item["severity"] == "warning" for item in diagnoses) else "clear"
        ),
        "primary_cause": diagnoses[0] if diagnoses else None,
        "diagnoses": diagnoses,
        "diagnosis_count": len(diagnoses),
        "model": "deterministic evidence rules over flow5, mesh, geometry and optimizer telemetry",
    }
