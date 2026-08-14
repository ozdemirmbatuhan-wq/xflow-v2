from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

import numpy as np

from aeropt.airfoil import naca4_design
from aeropt.baselines import build_baseline_profile
from aeropt.checkpoint import OptimizerCheckpointStore, optimizer_fingerprint
from aeropt.convergence import BudgetEscalationController, BudgetEscalationSettings
from aeropt.diagnostics import diagnose_runtime_failure
from aeropt.exporters import flow5_plane_xml, wing_obj, wing_step_sections
from aeropt.flow5 import Flow5Mesh, Flow5Runner
from aeropt.flow5_optimization import (
    WingCandidate,
    build_pareto_analysis,
    fast_non_dominated_sort,
    nsga2_environmental_selection,
    select_wing_finalist_indices,
    wing_candidate_objectives,
    wing_constraint_violation,
    wing_objective_specs,
)
from aeropt.hydro import HydroSettings, analyze_hydro
from aeropt.models import FLUID_PRESETS, WingGeometry
from aeropt.reliability import build_multi_seed_report
from aeropt.structures import StructuralSettings, analyze_structure
from aeropt.surrogate import RBFSurrogateAdvisor, SurrogateSettings


FAKE_RUNNER = Path(__file__).with_name("fake_flow5_runner.py").resolve()


def sample_conditions(geometry: WingGeometry) -> list[dict]:
    distribution = []
    semispan = 0.5 * geometry.span
    for index in range(9):
        eta = index / 8.0
        chord = geometry.chord_at(eta)
        distribution.append(
            {
                "y_m": eta * semispan,
                "chord_m": chord,
                "local_cl": 0.8 * (1.0 - eta**2) ** 0.5,
                "lift_n_per_m": 180.0 * (1.0 - eta**2) ** 0.5,
            }
        )
    return [
        {
            "speed_m_s": 18.0,
            "target_cl": 0.6,
            "point": {"distribution": distribution, "cm": -0.04, "cp_min": -1.1},
        }
    ]


class AdvancedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.geometry = WingGeometry(2.4, 0.34, 0.45, 3.0, -2.0, 3.0, 1.08, -0.8)
        self.air = FLUID_PRESETS["air"]

    def test_three_section_geometry_preserves_piecewise_area_and_xml_mesh(self):
        self.assertNotAlmostEqual(self.geometry.mid_chord, self.geometry.linear_mid_chord)
        self.assertGreater(self.geometry.area, 0.0)
        foil = build_baseline_profile("e818").foil
        xml = flow5_plane_xml(
            foil, self.geometry, chordwise_panels=12, half_span_panels=20
        ).replace("<!DOCTYPE flow5>", "")
        root = ET.fromstring(xml)
        sections = root.findall(".//Section")
        self.assertEqual(len(sections), 3)
        self.assertEqual(
            sum(int(section.findtext("y_number_of_panels")) for section in sections), 20
        )
        self.assertTrue(
            all(section.findtext("x_number_of_panels") == "12" for section in sections)
        )

    def test_winglet_geometry_preserves_projected_span_and_exports_four_sections(self):
        geometry = WingGeometry(
            span=4.4,
            root_chord=1.1,
            taper=0.45,
            sweep_deg=4.0,
            tip_twist_deg=-2.0,
            alpha_deg=2.0,
            mid_chord_factor=1.04,
            mid_twist_deg=-0.8,
            winglet_enabled=True,
            winglet_height=0.35,
            winglet_cant_deg=78.0,
            winglet_toe_deg=-1.2,
            winglet_taper=0.55,
        )
        self.assertTrue(geometry.winglet_active)
        self.assertTrue(geometry.winglet_geometry_valid)
        self.assertLess(geometry.main_span, geometry.span)
        self.assertGreater(geometry.developed_area, geometry.area)

        foil = build_baseline_profile("e818").foil
        winglet_foil = naca4_design("0012")
        xml = flow5_plane_xml(
            foil,
            geometry,
            chordwise_panels=12,
            half_span_panels=24,
            section_foils=(foil, foil, foil, winglet_foil),
        ).replace("<!DOCTYPE flow5>", "")
        root = ET.fromstring(xml)
        sections = root.findall(".//Section")
        self.assertEqual(len(sections), 4)
        self.assertEqual(
            sum(int(section.findtext("y_number_of_panels")) for section in sections),
            24,
        )
        self.assertAlmostEqual(float(sections[2].findtext("Dihedral")), 78.0)
        self.assertEqual(
            [section.findtext("Right_Side_FoilName") for section in sections[-2:]],
            ["NACA0012", "NACA0012"],
        )
        main_tip_y = float(sections[2].findtext("y_position"))
        winglet_tip_y = float(sections[3].findtext("y_position"))
        projected_semispan = main_tip_y + (winglet_tip_y - main_tip_y) * np.cos(
            np.deg2rad(78.0)
        )
        self.assertAlmostEqual(projected_semispan, 0.5 * geometry.span, places=7)

        obj = wing_obj(foil, geometry, span_sections=17)
        z_coordinates = [
            float(line.split()[3]) for line in obj.splitlines() if line.startswith("v ")
        ]
        self.assertGreater(max(z_coordinates), 0.30)

        planar_sections = wing_step_sections(foil, self.geometry)
        winglet_sections = wing_step_sections(foil, geometry)
        self.assertEqual(len(planar_sections), 5)
        self.assertEqual(len(winglet_sections), 7)
        self.assertEqual({len(section) for section in planar_sections}, {160})
        self.assertEqual({len(section) for section in winglet_sections}, {160})
        self.assertAlmostEqual(
            min(point[1] for section in planar_sections for point in section),
            -max(point[1] for section in planar_sections for point in section),
            places=8,
        )
        self.assertGreater(
            max(point[2] for section in winglet_sections for point in section),
            0.30,
        )

    def test_structure_is_a_true_off_switch_and_runs_when_enabled(self):
        conditions = sample_conditions(self.geometry)
        disabled = analyze_structure(
            geometry=self.geometry,
            foil_thickness_ratio=0.12,
            fluid=self.air,
            conditions=conditions,
            settings=StructuralSettings(enabled=False),
        )
        self.assertFalse(disabled["performed"])
        self.assertEqual(disabled["penalty"], 0.0)
        enabled = analyze_structure(
            geometry=self.geometry,
            foil_thickness_ratio=0.12,
            fluid=self.air,
            conditions=conditions,
            settings=StructuralSettings(enabled=True),
        )
        self.assertTrue(enabled["performed"])
        self.assertGreater(enabled["estimated_wing_material_mass_kg"], 0.0)
        self.assertGreaterEqual(enabled["stress_utilization"], 0.0)

    def test_hydro_screen_uses_cp_min_and_depth(self):
        water = FLUID_PRESETS["fresh_water"]
        safe = analyze_hydro(
            geometry=self.geometry,
            fluid=water,
            conditions=sample_conditions(self.geometry),
            settings=HydroSettings(enabled=True, submergence_depth_m=2.0),
        )
        self.assertTrue(safe["performed"])
        self.assertIn("cavitation_utilization", safe)
        shallow_fast = analyze_hydro(
            geometry=self.geometry,
            fluid=water,
            conditions=sample_conditions(self.geometry),
            settings=HydroSettings(enabled=True, submergence_depth_m=0.02),
        )
        self.assertTrue(shallow_fast["free_surface_risk"])

    def test_panel_cavitation_map_reports_area_span_and_sensitivity(self):
        water = FLUID_PRESETS["fresh_water"]
        raw_panels = []
        for index, (y0, y1, cp, component) in enumerate(
            (
                (0.0, 0.6, -0.10, "main_wing"),
                (0.6, 1.2, -1.10, "main_wing"),
                (-0.6, 0.0, -0.15, "main_wing"),
                (-1.2, -0.6, -1.25, "winglet"),
            )
        ):
            raw_panels.append(
                {
                    "panel_index": index,
                    "wing_index": 0,
                    "surface_index": index,
                    "surface": "mid",
                    "component": component,
                    "side": "right" if y1 > 0 else "left",
                    "x_m": 0.2,
                    "y_m": 0.5 * (y0 + y1),
                    "z_m": 0.1 if component == "winglet" else 0.0,
                    "area_m2": 0.25,
                    "cp": cp,
                    "vertices": [
                        [0.0, y0, 0.0],
                        [0.4, y0, 0.0],
                        [0.4, y1, 0.0],
                        [0.0, y1, 0.0],
                    ],
                }
            )
        conditions = [
            {
                "speed_m_s": 18.0,
                "target_cl": 0.6,
                "point": {"cp_min": -1.25},
                "panel_telemetry": {
                    "thin_surfaces": True,
                    "upper_lower_resolved": False,
                    "cp_definition": "flow5 thin-surface panel pressure coefficient",
                    "panels": raw_panels,
                },
            }
        ]
        report = analyze_hydro(
            geometry=self.geometry,
            fluid=water,
            conditions=conditions,
            settings=HydroSettings(
                enabled=True,
                constraint_mode="report_only",
                submergence_depth_m=1.0,
            ),
        )
        self.assertTrue(report["performed"])
        self.assertTrue(report["panel_map_available"])
        self.assertFalse(report["passed"])
        self.assertTrue(report["constraint_passed"])
        self.assertEqual(report["penalty"], 0.0)
        self.assertAlmostEqual(report["risk_area_percent"], 50.0, places=8)
        self.assertGreater(report["affected_span_bins_percent"], 0.0)
        self.assertIn("winglet", report["panel_map"]["component_summary"])
        self.assertEqual(len(report["speed_sensitivity"]), 5)
        self.assertEqual(len(report["depth_sensitivity"]), 5)
        self.assertLessEqual(
            report["depth_sensitivity"][-1]["maximum_utilization"],
            report["depth_sensitivity"][0]["maximum_utilization"],
        )
        hard_report = analyze_hydro(
            geometry=self.geometry,
            fluid=water,
            conditions=conditions,
            settings=HydroSettings(
                enabled=True,
                constraint_mode="hard",
                submergence_depth_m=1.0,
            ),
        )
        self.assertFalse(hard_report["passed"])
        self.assertFalse(hard_report["constraint_passed"])
        self.assertGreater(hard_report["penalty"], 0.0)

    def test_runner_cache_reuses_identical_solver_evaluation(self):
        old = os.environ.get("AEROPT_ALLOW_TEST_DOUBLE")
        os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = "1"
        try:
            baseline = build_baseline_profile("e818")
            with tempfile.TemporaryDirectory() as cache_dir:
                runner = Flow5Runner(FAKE_RUNNER, cache_dir=cache_dir)
                kwargs = dict(
                    foil=baseline.foil,
                    geometry=self.geometry,
                    fluid=self.air,
                    speeds_m_s=[18.0],
                    method="VLM2",
                    alpha_min_deg=-2.0,
                    alpha_max_deg=8.0,
                    alpha_step_deg=2.0,
                    max_threads=4,
                    foil_dat_text=baseline.solver_dat_text,
                    mesh=Flow5Mesh(10, 14),
                )
                first = runner.analyze_wing(**kwargs)
                second = runner.analyze_wing(**kwargs)
                self.assertFalse(first["cache"]["hit"])
                self.assertTrue(second["cache"]["hit"])
                self.assertEqual(runner.cache_stats()["hits"], 1)
                point = second["cases"][0]["points"][0]
                self.assertIn("cp_min", point)
                self.assertTrue(point["distribution"])
        finally:
            if old is None:
                os.environ.pop("AEROPT_ALLOW_TEST_DOUBLE", None)
            else:
                os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = old

    def test_optimizer_checkpoint_round_trip_and_problem_fingerprint(self):
        first_key = optimizer_fingerprint("foil", {"seed": 42, "budget": 48})
        second_key = optimizer_fingerprint("foil", {"budget": 48, "seed": 42})
        self.assertEqual(first_key, second_key)
        with tempfile.TemporaryDirectory() as directory:
            store = OptimizerCheckpointStore(directory=directory)
            state = {
                "optimizer": "differential_evolution",
                "generation": 3,
                "evaluations_completed": 27,
                "population_vectors": [[0.1, 0.2], [0.3, 0.4]],
                "rng_state": {"bit_generator": "test"},
            }
            self.assertTrue(store.save(first_key, state))
            self.assertEqual(store.load(first_key), state)
            self.assertEqual(store.stats()["loads"], 1)
            self.assertTrue(store.clear(first_key))
            self.assertIsNone(store.load(first_key))

    def test_surrogate_only_ranks_proposals_and_reports_real_solver_contract(self):
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]])
        advisor = RBFSurrogateAdvisor(
            bounds,
            SurrogateSettings(enabled=True, proposals_per_real_evaluation=5),
        )
        for x in np.linspace(0.0, 1.0, 4):
            for y in np.linspace(0.0, 1.0, 4):
                advisor.record([x, y], (x - 0.25) ** 2 + (y - 0.75) ** 2)
        proposals = [
            np.asarray([0.9, 0.1]),
            np.asarray([0.3, 0.7]),
            np.asarray([0.6, 0.5]),
        ]
        selected = advisor.choose(proposals)
        self.assertIn(selected, range(len(proposals)))
        report = advisor.report(real_evaluations=16, budget=32)
        self.assertTrue(report["trained"])
        self.assertEqual(report["proposals_screened"], 2)
        self.assertTrue(report["finalists_always_solver_verified"])

    def test_pareto_front_is_built_from_solver_candidate_tradeoffs(self):
        def candidate(drag: float, bending: float, stall: float) -> WingCandidate:
            conditions = [
                {
                    "drag_n": drag,
                    "ld": 40.0 / drag,
                    "stall_ratio": stall,
                    "point": {
                        "root_bending_moment_nm": bending,
                        "out_of_mesh": False,
                        "viscous_converged": True,
                    },
                }
            ]
            return WingCandidate(self.geometry, score=drag / 40.0, response={"ok": True}, conditions=conditions)

        low_drag = candidate(1.0, 1400.0, 0.78)
        low_stall = candidate(1.4, 8.0, 0.70)
        dominated = candidate(1.8, 18.0, 0.88)
        report = build_pareto_analysis([low_drag, low_stall, dominated], low_drag)
        search_rows = {item["id"]: item for item in report["candidates"]}
        self.assertTrue(search_rows["search-1"]["on_pareto_front"])
        self.assertTrue(search_rows["search-2"]["on_pareto_front"])
        self.assertFalse(search_rows["search-3"]["on_pareto_front"])
        self.assertEqual(report["definition"], "non-dominated real flow5 candidates; all listed objectives minimized")

    def test_nsga2_keeps_non_dominated_extremes_and_removes_dominated_point(self):
        def candidate(drag: float, bending: float, stall: float) -> WingCandidate:
            return WingCandidate(
                self.geometry,
                score=drag / 40.0,
                response={"ok": True},
                conditions=[
                    {
                        "drag_n": drag,
                        "ld": 40.0 / drag,
                        "stall_ratio": stall,
                        "point": {
                            "root_bending_moment_nm": bending,
                            "out_of_mesh": False,
                            "viscous_converged": True,
                        },
                    }
                ],
            )

        candidates = [
            candidate(1.0, 14.0, 0.72),
            candidate(1.4, 8.0, 0.70),
            candidate(1.8, 18.0, 0.88),
            candidate(1.2, 11.0, 0.74),
        ]
        keys = ["mean_drag_n", "worst_stall_ratio"]
        fronts = fast_non_dominated_sort(candidates, keys)
        self.assertIn(0, fronts[0])
        self.assertIn(1, fronts[0])
        self.assertNotIn(2, fronts[0])
        selected = nsga2_environmental_selection(candidates, 2, keys)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item in candidates for item in selected))

    def test_root_moment_is_telemetry_only_and_never_changes_selection(self):
        def candidate(score: float, stall: float, moment: float) -> WingCandidate:
            return WingCandidate(
                self.geometry,
                score=score,
                response={"ok": True},
                conditions=[
                    {
                        "drag_n": 1.25,
                        "ld": 32.0,
                        "stall_ratio": stall,
                        "point": {
                            "root_bending_moment_nm": moment,
                            "out_of_mesh": False,
                            "viscous_converged": True,
                        },
                    }
                ],
            )

        low_moment = candidate(0.30, 0.80, 1.0)
        extreme_moment = candidate(0.30, 0.80, 1.0e12)
        keys = [item["key"] for item in wing_objective_specs()]
        self.assertNotIn("max_root_bending_moment_nm", keys)
        np.testing.assert_allclose(
            wing_candidate_objectives(low_moment, keys),
            wing_candidate_objectives(extreme_moment, keys),
        )
        self.assertEqual(
            wing_constraint_violation(low_moment),
            wing_constraint_violation(extreme_moment),
        )

    def test_scalar_only_winner_is_added_without_consuming_finalist_quota(self):
        def candidate(score: float, stall: float) -> WingCandidate:
            return WingCandidate(
                self.geometry,
                score=score,
                response={"ok": True},
                conditions=[
                    {
                        "drag_n": 1.0 + score,
                        "ld": 30.0,
                        "stall_ratio": stall,
                        "point": {
                            "root_bending_moment_nm": 1.0e9 * score,
                            "out_of_mesh": False,
                            "viscous_converged": True,
                        },
                    }
                ],
            )

        candidates = [candidate(0.40, 0.80), candidate(0.10, 1.20)]
        indices, report = select_wing_finalist_indices(
            candidates,
            1,
            ["mean_drag_n", "worst_stall_ratio"],
            optimizer="nsga2",
        )
        self.assertEqual(indices, [0, 1])
        self.assertEqual(report["requested_finalists"], 1)
        self.assertEqual(report["evaluated_finalists"], 2)
        self.assertTrue(report["scalar_only_diagnostic_added"])

    def test_budget_controller_escalates_then_stops_when_objectives_stabilize(self):
        controller = BudgetEscalationController(
            8,
            BudgetEscalationSettings(
                enabled=True,
                growth_factor=2.0,
                maximum_multiplier=4.0,
                convergence_tolerance_percent=0.5,
            ),
            hard_limit=100,
        )
        first_scores = [10.0, 10.1, 10.2, 10.3, 8.0, 8.2, 8.4, 8.6]
        first_objectives = [[score, score * 2.0] for score in first_scores]
        first = controller.observe(scores=first_scores, objectives=first_objectives)
        self.assertEqual(first["decision"], "escalated")
        self.assertEqual(controller.current_target, 16)
        second_scores = [*first_scores, *([8.0] * 8)]
        second_objectives = [*first_objectives, *([[8.0, 16.0]] * 8)]
        second = controller.observe(scores=second_scores, objectives=second_objectives)
        self.assertEqual(second["decision"], "converged")
        report = controller.report(evaluations=16)
        self.assertTrue(report["converged"])
        self.assertEqual(report["milestones"], [8, 16, 32])

    def test_multi_seed_report_selects_best_feasible_run_and_measures_spread(self):
        def result(objective: float, ld: float, span: float, feasible: bool = True) -> dict:
            geometry = self.geometry.to_dict()
            geometry["span"] = span
            return {
                "wing_optimization": {"objective": objective, "feasible": feasible},
                "wing": {"ld": ld, "drag_n": 40.0 / ld, "geometry": geometry},
                "airfoil": {"name": "Flow5-CST6"},
                "validation_report": {"passed": True},
                "solver_run": {"evaluation_cache": {}},
            }

        records = [
            {"seed": 10, "result": result(0.10, 20.0, 2.2)},
            {"seed": 20, "result": result(0.09, 22.0, 2.22)},
            {"seed": 30, "result": result(0.02, 5.0, 2.0, feasible=False)},
        ]
        report = build_multi_seed_report(records, objective_cv_tolerance_percent=50.0, geometry_cv_tolerance_percent=50.0)
        self.assertEqual(report["selected_seed"], 20)
        self.assertEqual(report["runs_completed"], 3)
        self.assertGreater(report["objective_cv_percent"], 0.0)

    def test_multi_seed_report_preserves_the_common_root_failure(self):
        records = [
            {
                "seed": seed,
                "result": None,
                "error": "flow5 ince profil doğrulamasında hedef CL aralığı yakınsamadı",
                "failure_diagnosis": {
                    "title": "Hedef CL polar/yakınsama aralığının dışında kaldı"
                },
            }
            for seed in (42, 100045, 200048)
        ]
        with self.assertRaisesRegex(
            RuntimeError,
            r"Ortak kök hata \(3/3 seed\).*hedef CL aralığı yakınsamadı",
        ):
            build_multi_seed_report(records)

    def test_multi_seed_report_preserves_distinct_seed_failures(self):
        records = [
            {"seed": 42, "result": None, "error": "ilk kök hata"},
            {"seed": 100045, "result": None, "error": "ikinci kök hata"},
        ]
        with self.assertRaises(RuntimeError) as captured:
            build_multi_seed_report(records)
        message = str(captured.exception)
        self.assertIn("seed 42: ilk kök hata", message)
        self.assertIn("seed 100045: ikinci kök hata", message)

    def test_runtime_failure_diagnosis_returns_actionable_runner_advice(self):
        diagnosis = diagnose_runtime_failure(
            "flow5 API sürümü uyumsuz: runner 8.0, beklenen 7.57"
        )
        self.assertEqual(diagnosis["code"], "runner_version")
        self.assertIn("7.57", diagnosis["recommendation"])


if __name__ == "__main__":
    unittest.main()
