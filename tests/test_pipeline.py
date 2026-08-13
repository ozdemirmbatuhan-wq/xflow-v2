import json
import base64
import io
import unittest
from unittest.mock import patch
import zipfile
from xml.etree import ElementTree as ET

from aeropt.pipeline import (
    DEFAULT_REQUEST,
    InputError,
    _fluid_from_input,
    _merge_defaults,
    run_design,
)
from aeropt.exporters import flow5_bundle_bytes


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_design(
            {
                "solver": {
                    "quality": "quick",
                    "seed": 7,
                    "lifting_line_modes": 8,
                    "airfoil_strategy": "internal",
                }
            }
        )

    def test_end_to_end_design_is_serializable_and_hits_lift(self):
        result = self.result
        self.assertIn(result["status"], {"feasible", "review"})
        self.assertAlmostEqual(result["wing"]["lift_n"], 120.0, delta=2.4)
        self.assertGreater(result["wing"]["ld"], 5.0)
        json.dumps(result, allow_nan=False)

    def test_exports_are_consistent(self):
        exports = self.result["exports"]
        first_line = exports["airfoil_dat"].splitlines()[0]
        self.assertEqual(first_line, self.result["airfoil"]["name"])
        xml_text = exports["plane_xml"].replace("<!DOCTYPE flow5>", "")
        root = ET.fromstring(xml_text)
        self.assertEqual(root.tag, "xflplane")
        foil_names = [node.text for node in root.findall(".//Left_Side_FoilName")]
        self.assertEqual(foil_names, [first_line, first_line, first_line])
        self.assertIn("group,parameter,value,unit", exports["results_csv"])
        self.assertIn("\nv ", exports["wing_obj"])
        self.assertIn("\nf ", exports["wing_obj"])
        bundle = base64.b64decode(exports["flow5_bundle_base64"])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertIn("aeropt-airfoil.dat", archive.namelist())
            self.assertIn("aeropt-wing.xml", archive.namelist())
            self.assertIn("aeropt-wing.obj", archive.namelist())

    def test_bundle_can_carry_distinct_scalar_only_cad_outputs(self):
        bundle = flow5_bundle_bytes(
            foil_dat_text="foil\n",
            plane_xml_text="<plane/>",
            wing_obj_text="o primary\n",
            results_csv_text="primary\n",
            project_json_text="{}",
            polar_csv_text=None,
            wing_step_bytes=b"PRIMARY STEP",
            scalar_only_wing_obj_text="o scalar\n",
            scalar_only_wing_step_bytes=b"SCALAR STEP",
            scalar_only_results_csv_text="scalar\n",
            scalar_only_flow5_project_bytes=b"SCALAR FL5",
            scalar_only_same_as_primary=False,
        )
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertEqual(archive.read("aeropt-wing.step"), b"PRIMARY STEP")
            self.assertEqual(
                archive.read("aeropt-scalar-only-wing.step"), b"SCALAR STEP"
            )
            self.assertEqual(
                archive.read("aeropt-scalar-only-optimized.fl5"), b"SCALAR FL5"
            )

    def test_rejects_supersonic_or_transonic_request(self):
        with self.assertRaises(InputError):
            run_design(
                {
                    "flow": {"speed_m_s": 300.0, "speed_max_m_s": 300.0},
                    "solver": {"airfoil_strategy": "internal"},
                }
            )

    def test_hybrid_mode_requires_existing_xfoil_binary(self):
        with self.assertRaises(InputError):
            run_design(
                {
                    "solver": {
                        "airfoil_strategy": "xfoil_closed_loop",
                        "xfoil_path": "/does/not/exist/xfoil",
                    }
                }
            )

    def test_default_request_not_mutated(self):
        self.assertEqual(DEFAULT_REQUEST["workflow"]["mode"], "coupled")
        self.assertEqual(DEFAULT_REQUEST["solver"]["quality"], "balanced")
        self.assertEqual(DEFAULT_REQUEST["solver"]["parallel_workers"], 16)
        self.assertEqual(DEFAULT_REQUEST["solver"]["airfoil_strategy"], "flow5_native")
        self.assertEqual(DEFAULT_REQUEST["airfoil"]["baseline_profile"], "e818")
        self.assertEqual(DEFAULT_REQUEST["airfoil"]["cst_order"], 6)
        self.assertEqual(DEFAULT_REQUEST["airfoil"]["solver_coordinate_points"], 100)
        self.assertEqual(DEFAULT_REQUEST["solver"]["flow5_threads"], 16)
        self.assertTrue(DEFAULT_REQUEST["solver"]["flow5_surrogate_enabled"])
        self.assertTrue(DEFAULT_REQUEST["solver"]["flow5_checkpoint_enabled"])
        self.assertEqual(DEFAULT_REQUEST["solver"]["flow5_wing_optimizer"], "nsga2")
        self.assertFalse(DEFAULT_REQUEST["wing"]["winglet_optimization_enabled"])
        self.assertNotIn("max_root_bending_moment_nm", DEFAULT_REQUEST["wing"])
        self.assertEqual(DEFAULT_REQUEST["solver"]["flow5_winglet_candidate_budget"], 48)
        self.assertTrue(DEFAULT_REQUEST["solver"]["flow5_budget_escalation_enabled"])
        self.assertEqual(DEFAULT_REQUEST["solver"]["flow5_budget_maximum_multiplier"], 4.0)
        self.assertEqual(
            DEFAULT_REQUEST["solver"]["flow5_budget_convergence_tolerance_percent"],
            3.0,
        )
        self.assertEqual(DEFAULT_REQUEST["solver"]["flow5_multi_seed_runs"], 1)
        self.assertTrue(DEFAULT_REQUEST["validation"]["enabled"])
        self.assertFalse(DEFAULT_REQUEST["structure"]["enabled"])
        self.assertEqual(DEFAULT_REQUEST["hydro"]["constraint_mode"], "hard")

        migrated = _merge_defaults(
            {"wing": {"max_root_bending_moment_nm": 123.0}}
        )
        self.assertNotIn("max_root_bending_moment_nm", migrated["wing"])

    def test_flow5_candidate_timeout_accepts_six_hours_and_rejects_more(self):
        with (
            patch(
                "aeropt.pipeline.resolve_flow5_runner_path",
                return_value=__file__,
            ),
            patch(
                "aeropt.pipeline.run_flow5_native_design",
                return_value={},
            ) as native_run,
        ):
            run_design(
                {
                    "workflow": {"mode": "foil_only"},
                    "solver": {"flow5_timeout_seconds": 21600},
                }
            )
        self.assertEqual(native_run.call_args.kwargs["settings"].timeout_seconds, 21600)
        with self.assertRaises(InputError):
            run_design(
                {
                    "workflow": {"mode": "foil_only"},
                    "solver": {"flow5_timeout_seconds": 21601},
                }
            )

    def test_fluid_preset_properties_are_used_when_json_omits_editable_fields(self):
        request = _merge_defaults({"flow": {"fluid": "sea_water"}})
        fluid = _fluid_from_input(request["flow"])
        self.assertEqual(fluid.density, 1025.0)
        self.assertEqual(fluid.dynamic_viscosity, 1.188e-3)
        self.assertEqual(fluid.speed_of_sound, 1500.0)

        overridden = _merge_defaults(
            {"flow": {"fluid": "sea_water", "density_kg_m3": 1030.0}}
        )
        self.assertEqual(overridden["flow"]["density_kg_m3"], 1030.0)
        self.assertEqual(overridden["flow"]["dynamic_viscosity_pa_s"], 1.188e-3)


if __name__ == "__main__":
    unittest.main()
