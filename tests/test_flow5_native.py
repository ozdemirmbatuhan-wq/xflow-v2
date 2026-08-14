from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import zipfile
from xml.etree import ElementTree as ET

from aeropt.pipeline import InputError, run_design
from aeropt.flow5 import Flow5Mesh, Flow5Runner, _hidden_subprocess_kwargs
from aeropt.flow5_pipeline import _sample_speeds


FAKE_RUNNER = Path(__file__).with_name("fake_flow5_runner.py").resolve()


class Flow5NativePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_value = os.environ.get("AEROPT_ALLOW_TEST_DOUBLE")
        os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = "1"
        try:
            cls.result = run_design(
                {
                    "flow": {
                        "speed_m_s": 18.0,
                        "speed_min_m_s": 14.0,
                        "speed_max_m_s": 22.0,
                        "speed_samples": 3,
                        "target_lift_n": 40.0,
                    },
                    "airfoil": {"design_cl": 0.70},
                    "solver": {
                        "airfoil_strategy": "flow5_native",
                        "flow5_runner_path": str(FAKE_RUNNER),
                        "flow5_threads": 16,
                        "flow5_foil_candidate_budget": 8,
                        "flow5_wing_candidate_budget": 8,
                        "flow5_finalists": 1,
                        "flow5_alpha_step_search_deg": 2.0,
                        "flow5_alpha_step_final_deg": 1.0,
                        "seed": 9,
                    },
                }
            )
        finally:
            if old_value is None:
                os.environ.pop("AEROPT_ALLOW_TEST_DOUBLE", None)
            else:
                os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = old_value

    def test_all_aerodynamic_provenance_is_flow5(self):
        result = self.result
        self.assertTrue(result["flow5_native"])
        self.assertEqual(result["solver_run"]["aerodynamic_score_source"], "flow5 only")
        self.assertIn("flow5", result["polar_source"])
        self.assertEqual(result["flow5_native_analysis"]["foil_solver"], "flow5::XFoilTask")
        self.assertEqual(len(result["foil_polars"]), 3)
        self.assertEqual(len(result["wing_cases"]), 3)
        self.assertGreater(result["wing"]["ld"], 5.0)
        self.assertTrue(result["wing_optimization"]["mesh_convergence"]["passed"])
        self.assertTrue(
            result["wing_optimization"]["solver_telemetry"][
                "spanwise_distribution_available"
            ]
        )
        self.assertIn("coupled_design", result)
        self.assertFalse(result["structural_analysis"]["enabled"])
        json.dumps(result, allow_nan=False)

    def test_manual_cl_is_only_the_first_seed_and_final_pair_is_returned(self):
        coupled = self.result["coupled_design"]
        history = coupled["history"]

        self.assertEqual(coupled["initial_design_cl_source"], "user")
        self.assertTrue(coupled["manual_design_cl_is_seed_only"])
        self.assertFalse(coupled["explicit_design_cl_locked"])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["input_source"], "user_seed")
        self.assertEqual(history[1]["input_source"], "previous_wing")
        self.assertAlmostEqual(history[0]["target_cls"][1], 0.70, places=12)
        self.assertEqual(history[1]["target_cls"], history[0]["next_target_cls"])
        self.assertEqual(history[1]["reynolds"], history[0]["next_reynolds"])
        self.assertNotAlmostEqual(
            history[1]["target_cls"][1], history[0]["target_cls"][1], places=4
        )
        self.assertFalse(history[0]["selected"])
        self.assertTrue(history[-1]["selected"])
        self.assertEqual(coupled["selected_iteration"], history[-1]["iteration"])
        self.assertEqual(self.result["solver_run"]["selected_coupled_iteration"], 2)
        self.assertAlmostEqual(
            self.result["airfoil_optimization"]["design_cl_at_reference"],
            history[-1]["target_cls"][1],
            places=12,
        )

    def test_real_project_payload_is_required_and_packaged(self):
        exports = self.result["exports"]
        self.assertEqual(
            base64.b64decode(exports["flow5_project_base64"]),
            b"FLOW5_TEST_DOUBLE_PROJECT\x00",
        )
        step = base64.b64decode(exports["wing_step_base64"])
        self.assertTrue(step.startswith(b"ISO-10303-21;"))
        self.assertIn(b"END-ISO-10303-21;", step)
        self.assertIn(b"SI_UNIT($,.METRE.)", step.replace(b" ", b""))
        self.assertIn(b"5 sections x 160 points", step)
        bundle = base64.b64decode(exports["flow5_bundle_base64"])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertIn("aeropt-optimized.fl5", archive.namelist())
            self.assertIn("aeropt-analysis.xml", archive.namelist())
            self.assertIn("aeropt-validation.json", archive.namelist())
            self.assertIn("aeropt-pareto.json", archive.namelist())
            self.assertIn("aeropt-diagnostics.json", archive.namelist())
            self.assertIn("aeropt-wing.step", archive.namelist())
            self.assertNotIn("aeropt-cavitation.json", archive.namelist())
            self.assertEqual(
                archive.read("aeropt-optimized.fl5"), b"FLOW5_TEST_DOUBLE_PROJECT\x00"
            )
            self.assertEqual(archive.read("aeropt-wing.step"), step)
        self.assertNotIn("cavitation_json", exports)

    def test_feasibility_first_and_scalar_only_selections_are_both_reported(self):
        comparison = self.result["wing_optimization"]["selection_comparison"]
        self.assertIn("feasibility_first", comparison)
        self.assertIn("scalar_only", comparison)
        self.assertEqual(
            comparison["feasibility_first"]["wing"]["geometry"],
            self.result["wing"]["geometry"],
        )
        self.assertIn(
            comparison["scalar_only"]["same_as_feasibility_first"],
            {True, False},
        )
        self.assertEqual(
            self.result["wing_optimization"]["finalists_requested"], 1
        )
        objective_keys = {
            item["key"]
            for item in self.result["wing_optimization"]["multi_objective"][
                "objective_specs"
            ]
        }
        self.assertNotIn("max_root_bending_moment_nm", objective_keys)
        scalar_exports = self.result["exports"]["scalar_only_alternative"]
        if not comparison["scalar_only"]["same_as_feasibility_first"]:
            self.assertTrue(scalar_exports["wing_step_base64"])
            self.assertTrue(scalar_exports["wing_obj"])

    def test_sixteen_core_budget_avoids_outer_inner_oversubscription(self):
        foil_meta = self.result["airfoil_optimization"]
        self.assertLessEqual(
            foil_meta["outer_parallel_runners"] * foil_meta["threads_per_runner"], 16
        )
        self.assertEqual(self.result["wing_optimization"]["threads_inside_flow5"], 16)
        self.assertTrue(self.result["wing_optimization"]["oversubscription_prevented"])

    def test_validation_pareto_diagnostics_and_resume_metadata_are_exported(self):
        validation = self.result["validation_report"]
        self.assertTrue(validation["enabled"])
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["checks_total"], 9)
        self.assertEqual(len(validation["regression_signature_sha256"]), 64)
        pareto = self.result["pareto_analysis"]
        self.assertGreaterEqual(pareto["candidate_count"], 2)
        self.assertGreaterEqual(pareto["frontier_count"], 1)
        self.assertTrue(all("fidelity" in row for row in pareto["candidates"]))
        self.assertEqual(pareto["provenance"], "optimizer_generated")
        self.assertEqual(self.result["wing_optimization"]["optimizer"], "nsga2")
        self.assertTrue(self.result["wing_optimization"]["multi_objective"]["enabled"])
        self.assertIn(
            self.result["wing_optimization"]["budget_convergence"]["status"],
            {"converged", "budget_exhausted"},
        )
        self.assertIn(
            self.result["diagnostic_report"]["status"],
            {"clear", "warning", "critical"},
        )
        stability = self.result["multi_seed_stability"]
        self.assertEqual(stability["status"], "single_run")
        self.assertFalse(stability["enabled"])
        self.assertIn("checkpoint", self.result["airfoil_optimization"])
        self.assertIn("surrogate", self.result["wing_optimization"])
        exports = self.result["exports"]
        for key in (
            "validation_json",
            "pareto_json",
            "multi_seed_json",
            "diagnostics_json",
        ):
            self.assertIn(key, exports)

    def test_e818_baseline_is_compared_before_the_same_100_point_foil_enters_wing_search(self):
        foil_meta = self.result["airfoil_optimization"]
        initial_baseline = self.result["initial_baseline_airfoil"]
        self.assertEqual(initial_baseline["identifier"], "e818")
        self.assertEqual(initial_baseline["cst_order"], 6)
        self.assertEqual(
            self.result["coupled_design"]["history"][0]["baseline_identifier"],
            "e818",
        )
        self.assertTrue(foil_meta["baseline"]["search_converged"])
        self.assertEqual(foil_meta["solver_coordinate_points"], 100)
        self.assertEqual(self.result["wing_optimization"]["foil_coordinate_points"], 100)
        self.assertEqual(foil_meta["selection"]["selected_name"], self.result["airfoil"]["name"])
        self.assertGreaterEqual(foil_meta["selection"]["finalists_evaluated"], 1)
        self.assertEqual(len(self.result["exports"]["airfoil_dat"].splitlines()) - 1, 100)

    def test_missing_native_runner_is_rejected(self):
        with self.assertRaisesRegex(InputError, "runner"):
            run_design({"solver": {"airfoil_strategy": "flow5_native", "flow5_runner_path": ""}})

    def test_speed_mesh_contains_bounds_and_exact_reference(self):
        self.assertEqual(_sample_speeds((13.0, 22.0), 18.0, 4), [13.0, 16.0, 18.0, 22.0])
        self.assertEqual(_sample_speeds((12.5, 12.5), 12.5, 5), [12.5])
        self.assertEqual(_sample_speeds((1.0, 12.5), 12.5, 1), [12.5])
        self.assertEqual(_sample_speeds((1.0, 12.5), 12.5, 2), [1.0, 12.5])

    def test_windows_flow5_children_are_started_without_a_console_window(self):
        class DummyStartupInfo:
            def __init__(self):
                self.dwFlags = 0
                self.wShowWindow = -1

        with (
            patch("aeropt.flow5.os.name", "nt"),
            patch.object(subprocess, "STARTUPINFO", DummyStartupInfo, create=True),
            patch.object(subprocess, "STARTF_USESHOWWINDOW", 1, create=True),
            patch.object(subprocess, "SW_HIDE", 0, create=True),
            patch.object(subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
        ):
            options = _hidden_subprocess_kwargs()

        self.assertEqual(options["creationflags"], 0x08000000)
        self.assertEqual(options["startupinfo"].dwFlags & 1, 1)
        self.assertEqual(options["startupinfo"].wShowWindow, 0)

    def test_bridge_uses_the_pinned_flow5_757_cp_storage_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "flow5_bridge" / "main.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "std::min(wing->nPanel4(), wingOpp.m_nPanel4)", source
        )
        self.assertIn("wingOpp.m_dCp[local]", source)
        self.assertNotIn("wingOpp.m_dCp.size()", source)
        self.assertIn("panel.index() * 3", source)
        self.assertIn('"panel_telemetry"', source)
        self.assertIn("gmsh::model::occ::addThruSections", source)
        self.assertIn("OpenCascade STEP loft did not produce a closed solid", source)
        self.assertIn("normalizeStepMetreUnit", source)
        self.assertIn('"wing_step"', source)

    def test_foil_only_result_can_feed_a_fixed_foil_wing_optimization(self):
        common = {
            "flow": {
                "speed_m_s": 18.0,
                "speed_min_m_s": 14.0,
                "speed_max_m_s": 22.0,
                "speed_samples": 3,
                "target_lift_n": 40.0,
            },
            "solver": {
                "airfoil_strategy": "flow5_native",
                "flow5_runner_path": str(FAKE_RUNNER),
                "flow5_threads": 16,
                "flow5_foil_candidate_budget": 8,
                "flow5_wing_candidate_budget": 8,
                "flow5_finalists": 1,
                "flow5_alpha_step_search_deg": 2.0,
                "flow5_alpha_step_final_deg": 1.0,
                "flow5_budget_escalation_enabled": False,
                "flow5_surrogate_enabled": False,
                "flow5_mesh_convergence_enabled": False,
                "seed": 19,
            },
        }
        old_value = os.environ.get("AEROPT_ALLOW_TEST_DOUBLE")
        os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = "1"
        try:
            foil_result = run_design({**common, "workflow": {"mode": "foil_only"}})
            self.assertEqual(foil_result["workflow_mode"], "foil_only")
            self.assertNotIn("wing", foil_result)
            self.assertIn("airfoil_dat", foil_result["exports"])
            self.assertEqual(foil_result["solver_run"]["wing_optimizer"], "skipped")

            wing_request = {
                **common,
                "workflow": {"mode": "wing_only"},
                "airfoil": {
                    "baseline_profile": "custom_dat",
                    "baseline_dat": foil_result["exports"]["airfoil_dat"],
                },
            }
            wing_result = run_design(wing_request)
        finally:
            if old_value is None:
                os.environ.pop("AEROPT_ALLOW_TEST_DOUBLE", None)
            else:
                os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = old_value

        self.assertEqual(wing_result["workflow_mode"], "wing_only")
        self.assertIn("wing", wing_result)
        self.assertEqual(
            wing_result["airfoil_optimization"]["optimizer"],
            "skipped_fixed_airfoil",
        )
        self.assertEqual(wing_result["airfoil_optimization"]["candidates_evaluated"], 1)
        self.assertEqual(wing_result["solver_run"]["foil_optimizer"], "skipped_fixed_airfoil")
        self.assertFalse(wing_result["coupled_design"]["enabled"])

    def test_optional_winglet_stage_compares_against_planar_optimum(self):
        request = {
            "workflow": {"mode": "coupled"},
            "flow": {
                "speed_m_s": 18.0,
                "speed_min_m_s": 18.0,
                "speed_max_m_s": 18.0,
                "speed_samples": 1,
                "target_lift_n": 40.0,
            },
            "airfoil": {"baseline_profile": "e818", "design_cl": 0.65},
            "wing": {
                "span_min_m": 2.0,
                "span_max_m": 2.4,
                "root_chord_min_m": 0.25,
                "root_chord_max_m": 0.55,
                "winglet_optimization_enabled": True,
                "winglet_naca_code": "2412",
                "winglet_height_min_m": 0.16,
                "winglet_height_max_m": 0.32,
                "winglet_cant_min_deg": 72.0,
                "winglet_cant_max_deg": 90.0,
                "winglet_toe_min_deg": -2.0,
                "winglet_toe_max_deg": 2.0,
                "winglet_taper_min": 0.40,
                "winglet_taper_max": 0.75,
            },
            "solver": {
                "airfoil_strategy": "flow5_native",
                "flow5_runner_path": str(FAKE_RUNNER),
                "flow5_threads": 16,
                "flow5_foil_candidate_budget": 8,
                "flow5_wing_candidate_budget": 8,
                "flow5_winglet_candidate_budget": 8,
                "flow5_finalists": 1,
                "flow5_alpha_step_search_deg": 2.0,
                "flow5_alpha_step_final_deg": 1.0,
                "flow5_search_half_span_panels": 10,
                "flow5_final_half_span_panels": 14,
                "flow5_convergence_half_span_panels": 18,
                "flow5_budget_escalation_enabled": False,
                "flow5_surrogate_enabled": False,
                "flow5_mesh_convergence_enabled": False,
                "flow5_checkpoint_enabled": False,
                "flow5_coupled_iterations": 2,
                "seed": 31,
            },
            "hydro": {"enabled": False},
        }
        old_value = os.environ.get("AEROPT_ALLOW_TEST_DOUBLE")
        os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = "1"
        try:
            result = run_design(request)
        finally:
            if old_value is None:
                os.environ.pop("AEROPT_ALLOW_TEST_DOUBLE", None)
            else:
                os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = old_value

        comparison = result["winglet_comparison"]
        self.assertTrue(comparison["enabled"])
        self.assertTrue(comparison["performed"])
        self.assertEqual(comparison["airfoil_name"], "NACA2412")
        self.assertEqual(comparison["execution_policy"], "post_coupling_once")
        self.assertEqual(comparison["execution_count"], 1)
        self.assertFalse(comparison["included_in_coupled_loop"])
        self.assertFalse(result["coupled_design"]["winglet_included"])
        self.assertEqual(
            [item["winglet_selection"] for item in result["coupled_design"]["history"]],
            ["not_run_in_coupled_loop", "not_run_in_coupled_loop"],
        )
        self.assertEqual(
            comparison["selection"], result["solver_run"]["winglet_selection"]
        )
        self.assertFalse(comparison["planar"]["geometry"]["winglet_active"])
        self.assertTrue(comparison["winglet"]["geometry"]["winglet_active"])
        self.assertTrue(comparison["same_projected_span_and_target_lift"])
        self.assertTrue(comparison["frozen_planar_geometry"])
        self.assertEqual(
            comparison["optimized_variables"], ["height", "cant", "toe", "taper"]
        )
        for key in (
            "span",
            "root_chord",
            "taper",
            "sweep_deg",
            "tip_twist_deg",
            "mid_chord_factor",
            "mid_twist_deg",
        ):
            self.assertAlmostEqual(
                comparison["planar"]["geometry"][key],
                comparison["winglet"]["geometry"][key],
                places=10,
            )
        self.assertIn("drag_percent", comparison["delta_winglet_vs_planar"])
        self.assertEqual(
            result["wing"]["geometry"]["winglet_active"],
            comparison["selection"] == "winglet",
        )
        exported = ET.fromstring(
            result["exports"]["plane_xml"].replace("<!DOCTYPE flow5>", "")
        )
        expected_sections = 4 if comparison["selection"] == "winglet" else 3
        self.assertEqual(len(exported.findall(".//Section")), expected_sections)
        if comparison["selection"] == "winglet":
            names = [
                section.findtext("Right_Side_FoilName")
                for section in exported.findall(".//Section")
            ]
            self.assertEqual(names[-2:], ["NACA2412", "NACA2412"])
        json.dumps(result, allow_nan=False)

    def test_invalid_fine_mesh_keeps_verified_final_mesh_as_review_result(self):
        original_analyze_wing = Flow5Runner.analyze_wing

        def reject_only_the_fine_mesh(runner, *args, **kwargs):
            response = original_analyze_wing(runner, *args, **kwargs)
            mesh = kwargs.get("mesh")
            if isinstance(mesh, Flow5Mesh) and mesh == Flow5Mesh(20, 32):
                response = deepcopy(response)
                for case in response.get("cases", []):
                    case["points"] = []
            return response

        request = {
            "workflow": {"mode": "coupled"},
            "flow": {
                "fluid": "sea_water",
                "speed_m_s": 7.5,
                "speed_min_m_s": 7.5,
                "speed_max_m_s": 7.5,
                "speed_samples": 1,
                "target_lift_n": 1500.0,
            },
            "wing": {
                "span_min_m": 0.40,
                "span_max_m": 0.45,
                "root_chord_min_m": 0.10,
                "root_chord_max_m": 0.20,
                "taper_min": 0.50,
                "taper_max": 1.0,
            },
            "solver": {
                "airfoil_strategy": "flow5_native",
                "flow5_runner_path": str(FAKE_RUNNER),
                "flow5_threads": 16,
                "flow5_foil_candidate_budget": 8,
                "flow5_wing_candidate_budget": 8,
                "flow5_finalists": 1,
                "flow5_coupled_iterations": 1,
                "flow5_budget_escalation_enabled": False,
                "flow5_surrogate_enabled": False,
                "flow5_cache_enabled": False,
                "flow5_checkpoint_enabled": False,
                "flow5_mesh_convergence_enabled": True,
                "seed": 42,
            },
            "hydro": {"enabled": True, "constraint_mode": "report_only"},
        }
        old_value = os.environ.get("AEROPT_ALLOW_TEST_DOUBLE")
        os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = "1"
        try:
            with patch.object(
                Flow5Runner, "analyze_wing", new=reject_only_the_fine_mesh
            ):
                result = run_design(request)
        finally:
            if old_value is None:
                os.environ.pop("AEROPT_ALLOW_TEST_DOUBLE", None)
            else:
                os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = old_value

        convergence = result["wing_optimization"]["mesh_convergence"]
        self.assertEqual(result["status"], "review")
        self.assertFalse(convergence["passed"])
        self.assertFalse(convergence["fine_mesh_valid"])
        self.assertTrue(convergence["fallback_used"])
        self.assertEqual(
            convergence["status"], "fine_mesh_invalid_fallback_to_final_mesh"
        )
        self.assertIn("yakınsayan kanat polar noktası", convergence["reason"])
        self.assertEqual(
            result["wing_optimization"]["output_mesh"], Flow5Mesh(14, 22).to_dict()
        )
        self.assertTrue(result["hydro_analysis"]["panel_map_available"])
        self.assertTrue(result["exports"]["flow5_project_base64"])
        self.assertTrue(result["exports"]["wing_step_base64"])
        json.dumps(result, allow_nan=False)

    def test_finalist_panel_cavitation_map_is_exported_without_blocking_ld(self):
        request = {
            "workflow": {"mode": "wing_only"},
            "flow": {
                "fluid": "fresh_water",
                "speed_m_s": 18.0,
                "speed_min_m_s": 18.0,
                "speed_max_m_s": 18.0,
                "speed_samples": 1,
                "target_lift_n": 400.0,
            },
            "airfoil": {"baseline_profile": "e818", "design_cl": 0.65},
            "solver": {
                "airfoil_strategy": "flow5_native",
                "flow5_runner_path": str(FAKE_RUNNER),
                "flow5_threads": 4,
                "flow5_wing_candidate_budget": 8,
                "flow5_finalists": 1,
                "flow5_alpha_step_search_deg": 2.0,
                "flow5_alpha_step_final_deg": 1.0,
                "flow5_budget_escalation_enabled": False,
                "flow5_surrogate_enabled": False,
                "flow5_mesh_convergence_enabled": False,
                "flow5_checkpoint_enabled": False,
                "seed": 41,
            },
            "hydro": {
                "enabled": True,
                "constraint_mode": "report_only",
                "submergence_depth_m": 1.0,
            },
        }
        old_value = os.environ.get("AEROPT_ALLOW_TEST_DOUBLE")
        os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = "1"
        try:
            result = run_design(request)
            disabled_request = deepcopy(request)
            disabled_request["hydro"]["enabled"] = False
            without_hydro = run_design(disabled_request)
        finally:
            if old_value is None:
                os.environ.pop("AEROPT_ALLOW_TEST_DOUBLE", None)
            else:
                os.environ["AEROPT_ALLOW_TEST_DOUBLE"] = old_value

        hydro = result["hydro_analysis"]
        self.assertTrue(hydro["performed"])
        self.assertTrue(hydro["panel_map_available"])
        self.assertEqual(hydro["constraint_mode"], "report_only")
        self.assertTrue(hydro["constraint_passed"])
        self.assertEqual(hydro["penalty"], 0.0)
        self.assertEqual(result["wing"]["geometry"], without_hydro["wing"]["geometry"])
        self.assertAlmostEqual(result["wing"]["ld"], without_hydro["wing"]["ld"], places=12)
        self.assertAlmostEqual(
            result["wing_optimization"]["objective"],
            without_hydro["wing_optimization"]["objective"],
            places=12,
        )
        self.assertGreater(hydro["panel_map"]["panel_count"], 0)
        self.assertTrue(hydro["panel_map"]["panels"][0]["vertices"])
        self.assertEqual(len(hydro["speed_sensitivity"]), 5)
        self.assertTrue(
            result["wing_optimization"]["solver_telemetry"]["panel_map_available"]
        )
        exports = result["exports"]
        exported = json.loads(exports["cavitation_json"])
        self.assertTrue(exported["panel_map_available"])
        bundle = base64.b64decode(exports["flow5_bundle_base64"])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertIn("aeropt-cavitation.json", archive.namelist())
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
