#!/usr/bin/env python3
"""Deterministic protocol double for tests; it is not an aerodynamic solver."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


PROTOCOL = "aeropt-flow5-v1"


def alphas(request: dict) -> list[float]:
    settings = request["alpha"]
    minimum = float(settings["min_deg"])
    maximum = float(settings["max_deg"])
    step = float(settings["step_deg"])
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [minimum + index * step for index in range(count + 1)]
    if not values or values[-1] < maximum - 1e-8:
        values.append(maximum)
    if minimum <= 0.0 <= maximum:
        values.append(0.0)
    return sorted(
        {
            round(value, 12): value
            for value in values
        }.values()
    )


def foil_metrics(path: Path) -> tuple[float, float]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), float(fields[1])))
    thickness = max(y for _, y in rows) - min(y for _, y in rows)
    relevant = [y for x, y in rows if 0.15 <= x <= 0.85]
    mean_y = sum(relevant) / max(1, len(relevant))
    return thickness, mean_y


def assert_coordinate_count(request: dict, path: Path) -> None:
    expected = int(request.get("foil_coordinate_points", 100))
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()[1:]
        if len(line.split()) >= 2
    ]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} foil coordinates, found {len(rows)}")


def write_step_test_double(request: dict, path: Path) -> None:
    sections = request.get("step_sections")
    if not isinstance(sections, list):
        raise ValueError("save_step requires step_sections")
    expected_sections = 7 if request.get("winglet_active") else 5
    if len(sections) != expected_sections:
        raise ValueError(
            f"expected {expected_sections} STEP sections, found {len(sections)}"
        )
    point_count = len(sections[0]) if sections else 0
    if point_count < 40 or any(len(section) != point_count for section in sections):
        raise ValueError("STEP sections must have at least 40 matched contour points")
    samples: list[tuple[float, float, float]] = []
    for section in sections:
        for point in section:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError("STEP contour point must contain x, y and z")
            coordinates = tuple(float(value) for value in point)
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("STEP contour contains a non-finite coordinate")
        samples.append(tuple(float(value) for value in section[0]))
    rows = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('AeroOpt deterministic STEP test double'),'2;1');",
        "FILE_NAME('aeropt-wing.step','','',('AeroOpt'),('AeroOpt'),'','','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
        "ENDSEC;",
        "DATA;",
        f"/* TEST DOUBLE ONLY: {len(sections)} sections x {point_count} points; SI metre coordinates */",
    ]
    for index, (x, y, z) in enumerate(samples, start=1):
        rows.append(f"#{index}=CARTESIAN_POINT('',({x:.12g},{y:.12g},{z:.12g}));")
    rows.append("#1000=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.));")
    rows.extend(("ENDSEC;", "END-ISO-10303-21;", ""))
    path.write_text("\n".join(rows), encoding="ascii")


def run_foil(request: dict) -> dict:
    foil_path = Path(request["paths"]["foil.dat"])
    assert_coordinate_count(request, foil_path)
    thickness, mean_y = foil_metrics(foil_path)
    shape_penalty = 0.020 * (thickness - 0.125) ** 2 + 0.004 * (mean_y - 0.018) ** 2
    polars = []
    for case in request["cases"]:
        reynolds = float(case["reynolds"])
        reynolds_penalty = 0.0020 * (450_000.0 / max(reynolds, 40_000.0)) ** 0.22
        points = []
        for alpha in alphas(request):
            cl = 0.112 * (alpha + 1.25 + 9.0 * mean_y)
            cd = 0.0062 + reynolds_penalty + shape_penalty + 0.0068 * (cl - 0.62) ** 2
            points.append(
                {
                    "alpha_deg": alpha,
                    "cl": cl,
                    "cd": cd,
                    "cdp": 0.72 * cd,
                    "cm_c4": -0.035 - 0.45 * mean_y,
                }
            )
        polars.append({**case, "points": points})
    return {"polars": polars}


def plane_geometry(path: Path) -> dict[str, float | bool]:
    text = path.read_text(encoding="utf-8").replace("<!DOCTYPE flow5>", "")
    root = ET.fromstring(text)
    sections = root.findall(".//Section")
    if len(sections) not in {3, 4}:
        raise ValueError(f"expected three or four wing sections, found {len(sections)}")
    winglet_enabled = len(sections) == 4 and abs(
        float(sections[-2].findtext("Dihedral", "0"))
    ) > 1.0e-8
    main_tip_index = -2 if winglet_enabled else -1
    root_chord = float(sections[0].findtext("Chord"))
    main_tip_chord = float(sections[main_tip_index].findtext("Chord"))
    mid_chord = float(sections[1].findtext("Chord"))
    main_half_span = float(sections[main_tip_index].findtext("y_position"))
    main_tip_offset = float(sections[main_tip_index].findtext("xOffset"))
    installed_incidence = float(sections[0].findtext("Twist"))
    twist = float(sections[main_tip_index].findtext("Twist")) - installed_incidence
    mid_twist = float(sections[1].findtext("Twist")) - installed_incidence
    winglet_length = 0.0
    winglet_height = 0.0
    winglet_projection = 0.0
    winglet_taper = 1.0
    winglet_toe = 0.0
    cant_deg = 90.0
    winglet_tip_chord = main_tip_chord
    if winglet_enabled:
        winglet_length = (
            float(sections[-1].findtext("y_position")) - main_half_span
        )
        cant_deg = float(sections[-2].findtext("Dihedral"))
        cant = math.radians(cant_deg)
        winglet_height = winglet_length * math.sin(cant)
        winglet_projection = winglet_length * math.cos(cant)
        winglet_tip_chord = float(sections[-1].findtext("Chord"))
        winglet_taper = winglet_tip_chord / max(main_tip_chord, 1.0e-12)
        winglet_toe = (
            float(sections[-1].findtext("Twist"))
            - installed_incidence
            - twist
        )
    projected_half_span = main_half_span + winglet_projection
    span = 2.0 * projected_half_span
    taper = main_tip_chord / root_chord
    sweep = math.degrees(
        math.atan2(
            main_tip_offset - 0.25 * (root_chord - main_tip_chord),
            main_half_span,
        )
    )
    linear_mid = 0.5 * (root_chord + main_tip_chord)
    mid_factor = mid_chord / max(linear_mid, 1e-12)
    main_area = 0.5 * main_half_span * (
        root_chord + 2.0 * mid_chord + main_tip_chord
    )
    winglet_projected_area = winglet_projection * (
        main_tip_chord + winglet_tip_chord
    )
    winglet_surface_area = winglet_length * (
        main_tip_chord + winglet_tip_chord
    )
    return {
        "span": span,
        "main_half_span": main_half_span,
        "root_chord": root_chord,
        "taper": taper,
        "sweep": sweep,
        "twist": twist,
        "mid_factor": mid_factor,
        "mid_twist": mid_twist,
        "installed_incidence": installed_incidence,
        "area": main_area + winglet_projected_area,
        "winglet_enabled": winglet_enabled,
        "winglet_height": winglet_height,
        "winglet_length": winglet_length,
        "winglet_projection": winglet_projection,
        "winglet_cant_deg": cant_deg,
        "winglet_toe_deg": winglet_toe,
        "winglet_taper": winglet_taper,
        "winglet_surface_area": winglet_surface_area,
    }


def panel_telemetry(
    geometry: dict[str, float | bool],
    *,
    chordwise_panels: int,
    half_span_panels: int,
    target_cl: float,
    sampled_cl: float,
    sampled_alpha_deg: float,
    thin_surfaces: bool,
) -> dict:
    """Geometry/Cp-shaped test data; never used as an aerodynamic model."""
    panels = []
    root_chord = float(geometry["root_chord"])
    tip_chord = root_chord * float(geometry["taper"])
    mid_chord = 0.5 * (root_chord + tip_chord) * float(geometry["mid_factor"])
    main_half_span = float(geometry["main_half_span"])
    sweep_tangent = math.tan(math.radians(float(geometry["sweep"])))

    def chord_at(eta: float) -> float:
        if eta <= 0.5:
            return root_chord + 2.0 * eta * (mid_chord - root_chord)
        return mid_chord + 2.0 * (eta - 0.5) * (tip_chord - mid_chord)

    def add_panel(
        vertices: list[list[float]],
        *,
        area: float,
        cp: float,
        side: str,
        component: str,
        surface_index: int,
        leading: bool,
        trailing: bool,
    ) -> None:
        center = [sum(vertex[axis] for vertex in vertices) / len(vertices) for axis in range(3)]
        panels.append(
            {
                "panel_index": len(panels),
                "wing_index": 0,
                "surface_index": surface_index,
                "surface": "mid" if thin_surfaces else "upper",
                "component": component,
                "side": side,
                "x_m": center[0],
                "y_m": center[1],
                "z_m": center[2],
                "nx": 0.0,
                "ny": 0.0,
                "nz": 1.0,
                "area_m2": area,
                "cp": cp,
                "leading_edge_panel": leading,
                "trailing_edge_panel": trailing,
                "vertices": vertices,
            }
        )

    for side_sign, side_name in ((-1.0, "left"), (1.0, "right")):
        for span_index in range(half_span_panels):
            eta0 = span_index / half_span_panels
            eta1 = (span_index + 1) / half_span_panels
            chord0 = chord_at(eta0)
            chord1 = chord_at(eta1)
            y0 = side_sign * eta0 * main_half_span
            y1 = side_sign * eta1 * main_half_span
            xle0 = abs(y0) * sweep_tangent
            xle1 = abs(y1) * sweep_tangent
            for chord_index in range(chordwise_panels):
                xc0 = chord_index / chordwise_panels
                xc1 = (chord_index + 1) / chordwise_panels
                xc = 0.5 * (xc0 + xc1)
                eta = 0.5 * (eta0 + eta1)
                cp = -0.18 - 1.35 * max(sampled_cl, 0.0) * (1.0 - xc) ** 0.58 * (
                    0.86 + 0.28 * eta
                )
                add_panel(
                    [
                        [xle0 + xc0 * chord0, y0, 0.0],
                        [xle0 + xc1 * chord0, y0, 0.0],
                        [xle1 + xc1 * chord1, y1, 0.0],
                        [xle1 + xc0 * chord1, y1, 0.0],
                    ],
                    area=abs(y1 - y0) * 0.5 * (chord0 + chord1) / chordwise_panels,
                    cp=cp,
                    side=side_name,
                    component="main_wing",
                    surface_index=1,
                    leading=chord_index == 0,
                    trailing=chord_index == chordwise_panels - 1,
                )

    if bool(geometry["winglet_enabled"]):
        span_panels = max(2, half_span_panels // 4)
        length = float(geometry["winglet_length"])
        projection = float(geometry["winglet_projection"])
        height = float(geometry["winglet_height"])
        winglet_tip_chord = tip_chord * float(geometry["winglet_taper"])
        for side_sign, side_name, surface_index in (
            (-1.0, "left", 0),
            (1.0, "right", 2),
        ):
            for span_index in range(span_panels):
                fraction0 = span_index / span_panels
                fraction1 = (span_index + 1) / span_panels
                chord0 = tip_chord + fraction0 * (winglet_tip_chord - tip_chord)
                chord1 = tip_chord + fraction1 * (winglet_tip_chord - tip_chord)
                y0 = side_sign * (main_half_span + fraction0 * projection)
                y1 = side_sign * (main_half_span + fraction1 * projection)
                z0 = fraction0 * height
                z1 = fraction1 * height
                for chord_index in range(chordwise_panels):
                    xc0 = chord_index / chordwise_panels
                    xc1 = (chord_index + 1) / chordwise_panels
                    xc = 0.5 * (xc0 + xc1)
                    fraction = 0.5 * (fraction0 + fraction1)
                    cp = -0.15 - 1.10 * max(sampled_cl, 0.0) * (1.0 - xc) ** 0.62 * (
                        0.90 + 0.20 * fraction
                    )
                    add_panel(
                        [
                            [xc0 * chord0, y0, z0],
                            [xc1 * chord0, y0, z0],
                            [xc1 * chord1, y1, z1],
                            [xc0 * chord1, y1, z1],
                        ],
                        area=length * 0.5 * (chord0 + chord1)
                        / (span_panels * chordwise_panels),
                        cp=cp,
                        side=side_name,
                        component="winglet",
                        surface_index=surface_index,
                        leading=chord_index == 0,
                        trailing=chord_index == chordwise_panels - 1,
                    )

    return {
        "target_cl": target_cl,
        "sampled_cl": sampled_cl,
        "sampled_alpha_deg": sampled_alpha_deg,
        "panel_count": len(panels),
        "panel_area_sum_m2": sum(float(panel["area_m2"]) for panel in panels),
        "thin_surfaces": thin_surfaces,
        "upper_lower_resolved": not thin_surfaces,
        "cp_definition": "flow5 thin-surface panel pressure coefficient",
        "panels": panels,
    }


def run_wing(request: dict) -> dict:
    section_foils = request.get("section_foils", [])
    if section_foils:
        for section in section_foils:
            assert_coordinate_count(request, Path(request["paths"][section["path_key"]]))
    else:
        assert_coordinate_count(request, Path(request["paths"]["foil.dat"]))
    geometry = plane_geometry(Path(request["paths"]["plane.xml"]))
    span = float(geometry["span"])
    root_chord = float(geometry["root_chord"])
    taper = float(geometry["taper"])
    sweep = float(geometry["sweep"])
    twist = float(geometry["twist"])
    installed_incidence = float(geometry["installed_incidence"])
    mid_factor = float(geometry["mid_factor"])
    mid_twist = float(geometry["mid_twist"])
    tip_chord = root_chord * taper
    mid_chord = 0.5 * (root_chord + tip_chord) * mid_factor
    area = float(geometry["area"])
    aspect_ratio = span * span / area
    density = float(request["fluid"]["density_kg_m3"])
    method = request["method"]
    mesh = request.get("mesh", {})
    chordwise_panels = int(mesh.get("chordwise_panels", 14))
    half_span_panels = int(mesh.get("half_span_panels", 18))
    nominal_panels = 2 * chordwise_panels * half_span_panels
    mesh_delta = 0.012 / math.sqrt(max(nominal_panels, 1))
    method_delta = 0.00015 if method in {"TRIUNIFORM", "TRILINEAR", "QUADS"} else 0.00045
    efficiency = max(
        0.56,
        0.93
        - 0.10 * (taper - 0.48) ** 2
        - 0.0007 * sweep**2
        - 0.003 * (twist + 1.5) ** 2
        - 0.025 * (mid_factor - 1.08) ** 2
        - 0.002 * (mid_twist - 0.55 * twist) ** 2,
    )
    winglet_area_ratio = float(geometry["winglet_surface_area"]) / max(area, 1.0e-12)
    winglet_quality = 0.0
    if geometry["winglet_enabled"]:
        cant = math.radians(float(geometry["winglet_cant_deg"]))
        toe = float(geometry["winglet_toe_deg"])
        winglet_taper = float(geometry["winglet_taper"])
        winglet_quality = (
            max(math.sin(cant), 0.0)
            * math.exp(-0.055 * toe**2)
            * math.exp(-1.4 * (winglet_taper - 0.55) ** 2)
        )
    induced_relief = 1.0 + 1.6 * float(geometry["winglet_height"]) / max(
        span, 1.0e-12
    ) * winglet_quality
    cases = []
    for case in request["cases"]:
        speed = float(case["speed_m_s"])
        q = 0.5 * density * speed**2
        points = []
        for alpha in alphas(request):
            cl = 0.145 * (
                alpha + installed_incidence + 0.7 + 0.11 * twist
            )
            cdi = cl * cl / (math.pi * aspect_ratio * efficiency * induced_relief**2)
            cdv = (
                0.0080
                + 0.0012 * (root_chord - 0.33) ** 2
                + method_delta
                + 0.0014 * cl**2
                + 0.0065 * winglet_area_ratio
            )
            cd = cdi + cdv + mesh_delta
            distribution = []
            for index in range(-5, 6):
                eta = index / 5.0
                eta_abs = abs(eta)
                if eta_abs <= 0.5:
                    chord = root_chord + 2.0 * eta_abs * (mid_chord - root_chord)
                else:
                    chord = mid_chord + 2.0 * (eta_abs - 0.5) * (tip_chord - mid_chord)
                local_cl = cl * math.sqrt(max(0.0, 1.0 - eta * eta))
                distribution.append(
                    {
                        "y_m": eta * span / 2.0,
                        "chord_m": chord,
                        "local_cl": local_cl,
                        "lift_n_per_m": q * chord * local_cl,
                        "reynolds": speed * chord / float(
                            request["fluid"]["kinematic_viscosity_m2_s"]
                        ),
                        "induced_angle_deg": -2.0 * abs(eta),
                        "cdi": cdi,
                        "cdv": cdv,
                        "bending_moment_nm": q * area * cl * span * (1.0 - abs(eta)) / 8.0,
                        "twist_deg": twist * abs(eta),
                        "converged": True,
                    }
                )
            points.append(
                {
                    "alpha_deg": alpha,
                    "cl": cl,
                    "cd": cd,
                    "cdi": cdi,
                    "cdv": cdv,
                    "cm": -0.04,
                    "lift_n": q * area * cl,
                    "drag_n": q * area * cd,
                    "root_bending_moment_nm": q * area * cl * span / 8.0,
                    "out_of_mesh": False,
                    "viscous_converged": True,
                    "viscous_converged_fraction": 1.0,
                    "station_count": len(distribution),
                    "panel4_count": nominal_panels,
                    "panel3_count": 2 * nominal_panels,
                    "cp_min": -0.85 - 0.65 * max(cl, 0.0),
                    "distribution": distribution,
                }
            )
        output_case = {"speed_m_s": speed, "method": method, "points": points}
        if request.get("panel_telemetry"):
            target_cl = float(request["panel_telemetry_target_lift_n"]) / max(
                q * area, 1.0e-12
            )
            sampled = min(points, key=lambda point: abs(float(point["cl"]) - target_cl))
            output_case["panel_telemetry"] = panel_telemetry(
                geometry,
                chordwise_panels=chordwise_panels,
                half_span_panels=half_span_panels,
                target_cl=target_cl,
                sampled_cl=float(sampled["cl"]),
                sampled_alpha_deg=float(sampled["alpha_deg"]),
                thin_surfaces=bool(request.get("thin_surfaces", True)),
            )
        cases.append(output_case)
    artifacts = {}
    if request.get("save_project"):
        project = Path(request["output_dir"]) / "aeropt-optimized.fl5"
        project.write_bytes(b"FLOW5_TEST_DOUBLE_PROJECT\x00")
        artifacts["project_fl5"] = str(project)
    if request.get("save_step"):
        step = Path(request["output_dir"]) / "aeropt-wing.step"
        write_step_test_double(request, step)
        artifacts["wing_step"] = str(step)
    return {
        "cases": cases,
        "artifacts": artifacts,
        "mesh": {
            "chordwise_panels": chordwise_panels,
            "half_span_panels": half_span_panels,
            "actual_panel4_count": nominal_panels,
            "actual_panel3_count": 2 * nominal_panels,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    response = {
        "protocol": PROTOCOL,
        "ok": False,
        "solver": {"name": "flow5", "version": "7.57-test-double"},
    }
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if request.get("protocol") != PROTOCOL:
            raise ValueError("protocol mismatch")
        payload = run_foil(request) if request.get("mode") == "foil" else run_wing(request)
        response.update(
            ok=True,
            mode=request.get("mode"),
            foil_coordinate_points_used=int(request.get("foil_coordinate_points", 100)),
            **payload,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        response["error"] = str(exc)
    Path(args.response).write_text(json.dumps(response), encoding="utf-8")
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
