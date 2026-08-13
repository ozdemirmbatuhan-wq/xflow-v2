from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exporters import airfoil_dat, flow5_plane_xml, wing_step_sections
from .models import AirfoilLike, Fluid, WingGeometry


PROTOCOL = "aeropt-flow5-v1"


def _hidden_subprocess_kwargs() -> dict[str, Any]:
    """Keep console-mode solver children invisible in the Windows GUI build."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


class Flow5RunnerError(RuntimeError):
    """Raised when the external flow5 API runner fails or violates the protocol."""


class Flow5CancelledError(Flow5RunnerError):
    """Raised when the user cancels an active flow5 subprocess."""


@dataclass(frozen=True)
class Flow5Mesh:
    """Structured-wing mesh controls used in flow5 XML and convergence checks."""

    chordwise_panels: int = 14
    half_span_panels: int = 18

    def __post_init__(self) -> None:
        if not 4 <= int(self.chordwise_panels) <= 200:
            raise ValueError("Kord yönündeki panel sayısı 4 ile 200 arasında olmalı")
        if not 4 <= int(self.half_span_panels) <= 400:
            raise ValueError("Yarı açıklıktaki panel sayısı 4 ile 400 arasında olmalı")

    @property
    def nominal_panels(self) -> int:
        return 2 * int(self.chordwise_panels) * int(self.half_span_panels)

    def to_dict(self) -> dict[str, int]:
        return {
            "chordwise_panels": int(self.chordwise_panels),
            "half_span_panels": int(self.half_span_panels),
            "nominal_panels": self.nominal_panels,
        }


def resolve_flow5_runner_path(configured: str | Path | None = None) -> Path | None:
    """Find an explicit, environment-provided, or PyInstaller-bundled runner."""
    names = ("aeropt-flow5-runner.exe", "aeropt-flow5-runner")
    candidates: list[Path] = []
    if configured and str(configured).strip():
        candidates.append(Path(str(configured)).expanduser())
    environment_path = os.environ.get("AEROPT_FLOW5_RUNNER", "").strip()
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        for name in names:
            candidates.append(Path(bundle_root) / "flow5" / name)
    executable_root = Path(sys.executable).resolve().parent
    for name in names:
        candidates.extend((executable_root / "flow5" / name, executable_root / name))
    source_root = Path(__file__).resolve().parent.parent
    for name in names:
        candidates.append(source_root / "flow5-runtime" / name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise Flow5RunnerError(f"flow5 yanıtında '{label}' sayı değil") from None
    if not math.isfinite(number):
        raise Flow5RunnerError(f"flow5 yanıtında '{label}' sonlu değil")
    return number


def _normalize_points(points: Any, *, wing: bool) -> list[dict[str, Any]]:
    if not isinstance(points, list):
        raise Flow5RunnerError("flow5 yanıtındaki polar noktaları liste değil")
    normalized: list[dict[str, Any]] = []
    required = ("alpha_deg", "cl", "cd")
    optional = (
        "cdp",
        "cm_c4",
        "cdi",
        "cdv",
        "cm",
        "lift_n",
        "drag_n",
        "root_bending_moment_nm",
        "cp_min",
        "viscous_converged_fraction",
        "station_count",
        "panel4_count",
        "panel3_count",
    )
    for index, raw in enumerate(points):
        if not isinstance(raw, dict):
            continue
        try:
            point: dict[str, Any] = {
                key: _finite(raw.get(key), f"points[{index}].{key}") for key in required
            }
        except Flow5RunnerError:
            continue
        if point["cd"] <= 0.0:
            continue
        for key in optional:
            if raw.get(key) is not None:
                try:
                    point[key] = _finite(raw[key], f"points[{index}].{key}")
                except Flow5RunnerError:
                    pass
        point["ld"] = point["cl"] / point["cd"]
        if wing:
            point["out_of_mesh"] = bool(raw.get("out_of_mesh", False))
            point["viscous_converged"] = bool(raw.get("viscous_converged", True))
            distribution = raw.get("distribution")
            if isinstance(distribution, list):
                clean_distribution: list[dict[str, float]] = []
                for station in distribution:
                    if not isinstance(station, dict):
                        continue
                    try:
                        clean_distribution.append(
                            {
                                "y_m": _finite(station.get("y_m"), "distribution.y_m"),
                                "chord_m": _finite(
                                    station.get("chord_m"), "distribution.chord_m"
                                ),
                                "local_cl": _finite(
                                    station.get("local_cl"), "distribution.local_cl"
                                ),
                                "lift_n_per_m": _finite(
                                    station.get("lift_n_per_m"), "distribution.lift_n_per_m"
                                ),
                            }
                        )
                    except Flow5RunnerError:
                        continue
                    for key in (
                        "reynolds",
                        "induced_angle_deg",
                        "cdi",
                        "cdv",
                        "bending_moment_nm",
                        "twist_deg",
                    ):
                        if station.get(key) is not None:
                            try:
                                clean_distribution[-1][key] = _finite(
                                    station[key], f"distribution.{key}"
                                )
                            except Flow5RunnerError:
                                pass
                    clean_distribution[-1]["converged"] = bool(
                        station.get("converged", True)
                    )
                point["distribution"] = clean_distribution
        normalized.append(point)
    normalized.sort(key=lambda row: row["alpha_deg"])
    if len(normalized) < 3:
        raise Flow5RunnerError("flow5 polarında en az üç geçerli yakınsayan nokta gerekli")
    return normalized


def _normalize_panel_telemetry(raw: Any) -> dict[str, Any] | None:
    """Validate finalist-only flow5 panel geometry/Cp without inventing coordinates."""
    if not isinstance(raw, dict) or not isinstance(raw.get("panels"), list):
        return None
    panels: list[dict[str, Any]] = []
    for index, source in enumerate(raw["panels"]):
        if not isinstance(source, dict):
            continue
        try:
            panel: dict[str, Any] = {
                "panel_index": int(source.get("panel_index", index)),
                "wing_index": int(source.get("wing_index", 0)),
                "surface_index": int(source.get("surface_index", -1)),
                "surface": str(source.get("surface", "unknown")),
                "component": str(source.get("component", "main_wing")),
                "side": str(source.get("side", "unknown")),
                "x_m": _finite(source.get("x_m"), f"panels[{index}].x_m"),
                "y_m": _finite(source.get("y_m"), f"panels[{index}].y_m"),
                "z_m": _finite(source.get("z_m"), f"panels[{index}].z_m"),
                "nx": _finite(source.get("nx", 0.0), f"panels[{index}].nx"),
                "ny": _finite(source.get("ny", 0.0), f"panels[{index}].ny"),
                "nz": _finite(source.get("nz", 0.0), f"panels[{index}].nz"),
                "area_m2": _finite(
                    source.get("area_m2"), f"panels[{index}].area_m2"
                ),
                "cp": _finite(source.get("cp"), f"panels[{index}].cp"),
                "leading_edge_panel": bool(source.get("leading_edge_panel", False)),
                "trailing_edge_panel": bool(source.get("trailing_edge_panel", False)),
            }
        except (Flow5RunnerError, TypeError, ValueError):
            continue
        if panel["area_m2"] <= 0.0:
            continue
        vertices: list[list[float]] = []
        for vertex_index, vertex in enumerate(source.get("vertices", [])):
            if not isinstance(vertex, list) or len(vertex) != 3:
                continue
            try:
                vertices.append(
                    [
                        _finite(
                            coordinate,
                            f"panels[{index}].vertices[{vertex_index}]",
                        )
                        for coordinate in vertex
                    ]
                )
            except Flow5RunnerError:
                continue
        if len(vertices) >= 3:
            panel["vertices"] = vertices
        panels.append(panel)
    if not panels:
        return None
    normalized: dict[str, Any] = {
        "panel_count": len(panels),
        "panel_area_sum_m2": float(sum(panel["area_m2"] for panel in panels)),
        "thin_surfaces": bool(raw.get("thin_surfaces", True)),
        "upper_lower_resolved": bool(raw.get("upper_lower_resolved", False)),
        "cp_definition": str(
            raw.get("cp_definition", "flow5 panel pressure coefficient")
        ),
        "panels": panels,
    }
    for key in ("target_cl", "sampled_cl", "sampled_alpha_deg"):
        if raw.get(key) is not None:
            try:
                normalized[key] = _finite(raw[key], f"panel_telemetry.{key}")
            except Flow5RunnerError:
                pass
    return normalized


class Flow5Runner:
    """Subprocess adapter for the small C++ runner shipped in ``flow5_bridge``.

    The adapter deliberately knows nothing about aerodynamics. It writes geometry,
    sends analysis settings, validates flow5's result, and returns those values.
    """

    def __init__(
        self,
        executable: str | Path,
        *,
        timeout_seconds: float = 900.0,
        cache_enabled: bool = True,
        cache_dir: str | Path | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.path = Path(executable).expanduser().resolve()
        if not self.path.is_file():
            raise Flow5RunnerError(f"flow5 runner bulunamadı: {self.path}")
        self.timeout_seconds = float(timeout_seconds)
        self.cache_enabled = bool(cache_enabled)
        self.cancel_event = cancel_event
        self._stats_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        if cache_dir is None:
            platform_cache = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
            if platform_cache:
                base = Path(platform_cache)
            elif sys.platform == "darwin":
                base = Path.home() / "Library" / "Caches"
            else:
                base = Path.home() / ".cache"
            self.cache_dir = base / "AeroOpt" / "flow5-cache-v2"
        else:
            self.cache_dir = Path(cache_dir).expanduser()

    def _check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise Flow5CancelledError("Optimizasyon kullanıcı tarafından durduruldu")

    def _cache_path(self, request: dict[str, Any], files: dict[str, str]) -> Path:
        try:
            stat = self.path.stat()
            executable_identity = {
                "path": str(self.path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            executable_identity = {"path": str(self.path)}
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "cache_schema": 2,
                    "protocol": PROTOCOL,
                    "runner": executable_identity,
                    "request": request,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name, contents in sorted(files.items()):
            digest.update(name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(contents.encode("utf-8"))
            digest.update(b"\x00")
        return self.cache_dir / digest.hexdigest()[:2] / f"{digest.hexdigest()}.json"

    def cache_stats(self) -> dict[str, int | bool | str]:
        with self._stats_lock:
            return {
                "enabled": self.cache_enabled,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "resume_reused_evaluations": self.cache_hits,
            }

    def _command(self, request_path: Path, response_path: Path) -> list[str]:
        if self.path.suffix.lower() == ".py":
            return [sys.executable, str(self.path), "--request", str(request_path), "--response", str(response_path)]
        return [str(self.path), "--request", str(request_path), "--response", str(response_path)]

    def _invoke(self, request: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
        self._check_cancelled()
        cache_path = self._cache_path(request, files)
        if self.cache_enabled:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("protocol") == PROTOCOL and cached.get("ok"):
                    with self._stats_lock:
                        self.cache_hits += 1
                    cached["cache"] = {"hit": True, "key": cache_path.stem}
                    return cached
            except (OSError, json.JSONDecodeError):
                pass
        with self._stats_lock:
            self.cache_misses += 1
        with tempfile.TemporaryDirectory(prefix="aeropt-flow5-") as temp_name:
            workdir = Path(temp_name).resolve()
            paths: dict[str, str] = {}
            for key, contents in files.items():
                path = workdir / key
                path.write_text(contents, encoding="utf-8", newline="\n")
                paths[key] = str(path)
            request = {**request, "protocol": PROTOCOL, "paths": paths, "output_dir": str(workdir)}
            request_path = workdir / "request.json"
            response_path = workdir / "response.json"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
            )
            try:
                process = subprocess.Popen(
                    self._command(request_path, response_path),
                    cwd=workdir,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **_hidden_subprocess_kwargs(),
                )
            except OSError as exc:
                raise Flow5RunnerError(f"flow5 runner başlatılamadı: {exc}") from exc

            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        process.terminate()
                        try:
                            process.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise Flow5CancelledError("Optimizasyon kullanıcı tarafından durduruldu")
                    if time.monotonic() >= deadline:
                        process.kill()
                        stdout, stderr = process.communicate()
                        raise Flow5RunnerError(
                            f"flow5 runner {self.timeout_seconds:g} saniyede tamamlanamadı"
                        )

            completed_returncode = process.returncode

            if not response_path.is_file():
                detail = (stderr or stdout or "yanıt dosyası oluşmadı")[-1200:].strip()
                raise Flow5RunnerError(
                    f"flow5 runner yanıt vermedi (çıkış {completed_returncode}): {detail}"
                )
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise Flow5RunnerError(f"flow5 runner geçersiz JSON döndürdü: {exc}") from exc
            if not isinstance(response, dict) or response.get("protocol") != PROTOCOL:
                raise Flow5RunnerError("flow5 runner protokol sürümü uyuşmuyor")
            if completed_returncode != 0 or not response.get("ok", False):
                detail = str(response.get("error") or stderr or "bilinmeyen flow5 hatası")
                raise Flow5RunnerError(detail[-1600:])
            solver = response.get("solver")
            if not isinstance(solver, dict) or str(solver.get("name", "")).lower() != "flow5":
                raise Flow5RunnerError("runner, çözücü kaynağını flow5 olarak doğrulamadı")
            version = str(solver.get("version", ""))
            is_test_double = "test-double" in version.lower()
            if is_test_double:
                if os.environ.get("AEROPT_ALLOW_TEST_DOUBLE") != "1":
                    raise Flow5RunnerError("test flow5 koşucusu üretim modunda kullanılamaz")
            elif version != "7.57":
                raise Flow5RunnerError(
                    f"flow5 API sürümü uyumsuz: runner {version or 'sürüm bildirmedi'}, beklenen 7.57"
                )

            artifact_payloads: dict[str, dict[str, str]] = {}
            artifacts = response.get("artifacts", {})
            if isinstance(artifacts, dict):
                for key, raw_path in artifacts.items():
                    if not raw_path:
                        continue
                    artifact_path = Path(str(raw_path)).resolve()
                    try:
                        artifact_path.relative_to(workdir)
                    except ValueError:
                        raise Flow5RunnerError("runner çalışma klasörü dışından artifact döndürdü") from None
                    if artifact_path.is_file():
                        artifact_payloads[str(key)] = {
                            "filename": artifact_path.name,
                            "base64": base64.b64encode(artifact_path.read_bytes()).decode("ascii"),
                        }
            response["artifact_payloads"] = artifact_payloads
            response["runner_log_tail"] = (stdout + "\n" + stderr)[-3000:].strip()
            response["cache"] = {"hit": False, "key": cache_path.stem}
            if self.cache_enabled:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_cache = cache_path.with_name(
                        f".{cache_path.name}.{uuid.uuid4().hex}.tmp"
                    )
                    temporary_cache.write_text(
                        json.dumps(response, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8",
                    )
                    os.replace(temporary_cache, cache_path)
                except OSError:
                    pass
            return response

    def analyze_foil(
        self,
        *,
        foil: AirfoilLike,
        fluid: Fluid,
        speeds_m_s: list[float],
        reference_chord_m: float,
        alpha_min_deg: float,
        alpha_max_deg: float,
        alpha_step_deg: float,
        max_threads: int,
        coordinate_points: int = 100,
        foil_dat_text: str | None = None,
        ncrit: float = 9.0,
        xtr_top: float = 1.0,
        xtr_bottom: float = 1.0,
        save_project: bool = False,
    ) -> dict[str, Any]:
        cases = [
            {
                "speed_m_s": float(speed),
                "reynolds": fluid.reynolds(float(speed), reference_chord_m),
                "mach": fluid.mach(float(speed)),
            }
            for speed in speeds_m_s
        ]
        response = self._invoke(
            {
                "mode": "foil",
                "foil_name": foil.name,
                "cases": cases,
                "alpha": {
                    "min_deg": float(alpha_min_deg),
                    "max_deg": float(alpha_max_deg),
                    "step_deg": float(alpha_step_deg),
                },
                "transition": {
                    "ncrit": float(ncrit),
                    "xtr_top": float(xtr_top),
                    "xtr_bottom": float(xtr_bottom),
                },
                "max_threads": int(max_threads),
                "foil_coordinate_points": int(coordinate_points),
                "save_project": bool(save_project),
            },
            {
                "foil.dat": foil_dat_text
                if foil_dat_text is not None
                else airfoil_dat(foil, total_points=coordinate_points)
            },
        )
        if int(response.get("foil_coordinate_points_used", -1)) != coordinate_points:
            raise Flow5RunnerError("flow5 runner profili tam 100 koordinat noktasıyla çözmedi")
        raw_polars = response.get("polars")
        if not isinstance(raw_polars, list) or len(raw_polars) != len(cases):
            raise Flow5RunnerError("flow5, istenen akış noktalarının tümü için 2B polar döndürmedi")
        polars: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_polars):
            if not isinstance(raw, dict):
                raise Flow5RunnerError("flow5 2B polar yanıtı geçersiz")
            polars.append(
                {
                    "speed_m_s": _finite(raw.get("speed_m_s", cases[index]["speed_m_s"]), "speed_m_s"),
                    "reynolds": _finite(raw.get("reynolds", cases[index]["reynolds"]), "reynolds"),
                    "mach": _finite(raw.get("mach", cases[index]["mach"]), "mach"),
                    "points": _normalize_points(raw.get("points"), wing=False),
                }
            )
        response["polars"] = sorted(polars, key=lambda polar: polar["speed_m_s"])
        return response

    def analyze_wing(
        self,
        *,
        foil: AirfoilLike,
        geometry: WingGeometry,
        fluid: Fluid,
        speeds_m_s: list[float],
        method: str,
        alpha_min_deg: float,
        alpha_max_deg: float,
        alpha_step_deg: float,
        max_threads: int,
        coordinate_points: int = 100,
        foil_dat_text: str | None = None,
        ncrit: float = 9.0,
        xtr_top: float = 1.0,
        xtr_bottom: float = 1.0,
        save_project: bool = False,
        panel_telemetry: bool = False,
        panel_telemetry_target_lift_n: float | None = None,
        thin_surfaces: bool = True,
        mesh: Flow5Mesh | None = None,
        section_foils: tuple[AirfoilLike, AirfoilLike, AirfoilLike] | None = None,
        section_foil_dat_texts: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        method = method.upper()
        if method not in {"LLT", "VLM1", "VLM2", "QUADS", "TRIUNIFORM", "TRILINEAR"}:
            raise Flow5RunnerError(f"desteklenmeyen flow5 yöntemi: {method}")
        mesh = mesh or Flow5Mesh()
        if (section_foils is None) != (section_foil_dat_texts is None):
            raise Flow5RunnerError(
                "Kesit profilleri ve DAT metinleri birlikte verilmelidir"
            )
        if panel_telemetry and (
            panel_telemetry_target_lift_n is None
            or not math.isfinite(float(panel_telemetry_target_lift_n))
            or float(panel_telemetry_target_lift_n) <= 0.0
        ):
            raise Flow5RunnerError(
                "Panel telemetrisi için pozitif hedef taşıma kuvveti gerekli"
            )
        wing_files: dict[str, str]
        section_request: list[dict[str, str]] | None = None
        if section_foils is not None and section_foil_dat_texts is not None:
            keys = ("foil_root.dat", "foil_mid.dat", "foil_tip.dat")
            wing_files = {
                key: dat_text
                for key, dat_text in zip(keys, section_foil_dat_texts)
            }
            section_request = [
                {"name": section_foil.name, "path_key": key}
                for section_foil, key in zip(section_foils, keys)
            ]
        else:
            wing_files = {
                "foil.dat": (
                    foil_dat_text
                    if foil_dat_text is not None
                    else airfoil_dat(foil, total_points=coordinate_points)
                )
            }
        response = self._invoke(
            {
                "mode": "wing",
                "foil_name": foil.name,
                "cases": [{"speed_m_s": float(speed)} for speed in speeds_m_s],
                "method": method,
                "fluid": {
                    "density_kg_m3": float(fluid.density),
                    "kinematic_viscosity_m2_s": float(fluid.kinematic_viscosity),
                },
                "alpha": {
                    "min_deg": float(alpha_min_deg),
                    "max_deg": float(alpha_max_deg),
                    "step_deg": float(alpha_step_deg),
                },
                "transition": {
                    "ncrit": float(ncrit),
                    "xtr_top": float(xtr_top),
                    "xtr_bottom": float(xtr_bottom),
                },
                "max_threads": int(max_threads),
                "foil_coordinate_points": int(coordinate_points),
                "save_project": bool(save_project),
                "save_step": bool(save_project),
                "panel_telemetry": bool(panel_telemetry),
                "winglet_active": bool(geometry.winglet_active),
                "thin_surfaces": bool(thin_surfaces),
                **(
                    {
                        "panel_telemetry_target_lift_n": float(
                            panel_telemetry_target_lift_n
                        )
                    }
                    if panel_telemetry
                    else {}
                ),
                "mesh": mesh.to_dict(),
                **({"section_foils": section_request} if section_request else {}),
                **(
                    {
                        "step_sections": wing_step_sections(
                            foil,
                            geometry,
                            section_foils=section_foils,
                        )
                    }
                    if save_project
                    else {}
                ),
            },
            {
                **wing_files,
                "plane.xml": flow5_plane_xml(
                    foil,
                    geometry,
                    chordwise_panels=mesh.chordwise_panels,
                    half_span_panels=mesh.half_span_panels,
                    section_foils=section_foils,
                ),
            },
        )
        if int(response.get("foil_coordinate_points_used", -1)) != coordinate_points:
            raise Flow5RunnerError("flow5 runner kanat kesitini tam 100 koordinat noktasıyla çözmedi")
        raw_cases = response.get("cases")
        if not isinstance(raw_cases, list) or len(raw_cases) != len(speeds_m_s):
            raise Flow5RunnerError("flow5, istenen akış noktalarının tümü için 3B polar döndürmedi")
        cases: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_cases):
            if not isinstance(raw, dict):
                raise Flow5RunnerError("flow5 3B polar yanıtı geçersiz")
            case = {
                "speed_m_s": _finite(
                    raw.get("speed_m_s", speeds_m_s[index]), "speed_m_s"
                ),
                "method": str(raw.get("method", method)).upper(),
                "points": _normalize_points(raw.get("points"), wing=True),
            }
            telemetry = _normalize_panel_telemetry(raw.get("panel_telemetry"))
            if telemetry is not None:
                case["panel_telemetry"] = telemetry
            cases.append(case)
        response["cases"] = sorted(cases, key=lambda case: case["speed_m_s"])
        raw_mesh = response.get("mesh")
        normalized_mesh = mesh.to_dict()
        if isinstance(raw_mesh, dict):
            for key in ("actual_panel4_count", "actual_panel3_count"):
                if raw_mesh.get(key) is not None:
                    try:
                        normalized_mesh[key] = int(raw_mesh[key])
                    except (TypeError, ValueError):
                        pass
        response["mesh"] = normalized_mesh
        return response
