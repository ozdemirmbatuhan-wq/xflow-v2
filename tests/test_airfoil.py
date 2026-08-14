import math
import unittest

import numpy as np

from aeropt.airfoil import (
    airfoil_coordinates,
    alpha_for_cl,
    cst_geometry_is_valid,
    fit_naca_to_cst,
    generate_polar,
    naca4_coordinates,
    naca4_design,
    polar_point,
    thin_airfoil_properties,
)
from aeropt.baselines import EPPLER_E818_DAT, build_baseline_profile
from aeropt.exporters import airfoil_dat
from aeropt.models import AirfoilDesign


class AirfoilTests(unittest.TestCase):
    def test_four_digit_naca_code_is_normalized_and_validated(self):
        foil = naca4_design("NACA 2412")
        self.assertEqual(foil.name, "NACA2412")
        self.assertAlmostEqual(foil.max_camber, 0.02)
        self.assertAlmostEqual(foil.camber_position, 0.4)
        self.assertAlmostEqual(foil.thickness, 0.12)
        with self.assertRaises(ValueError):
            naca4_design("23012")
        with self.assertRaises(ValueError):
            naca4_design("0010x")

    def test_eppler_e818_cst6_fit_and_solver_contour_use_exactly_100_points(self):
        baseline = build_baseline_profile("e818", cst_order=6, solver_point_count=100)
        self.assertEqual(baseline.identifier, "e818")
        self.assertEqual(baseline.source_point_count, 68)
        self.assertEqual(baseline.foil.family, "CST6")
        self.assertLess(baseline.fit_rms_over_c, 0.0004)
        self.assertLess(baseline.fit_max_over_c, 0.0010)
        x, y = airfoil_coordinates(baseline.foil, total_points=100)
        self.assertEqual(len(x), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(airfoil_dat(baseline.foil).splitlines()) - 1, 100)

    def test_eppler_data_can_follow_the_custom_dat_path(self):
        baseline = build_baseline_profile(
            "custom_dat", custom_dat=EPPLER_E818_DAT, cst_order=5, solver_point_count=100
        )
        self.assertEqual(baseline.identifier, "custom_dat")
        self.assertEqual(baseline.foil.family, "CST5")
        self.assertEqual(baseline.solver_point_count, 100)

    def test_coordinate_order_and_closed_trailing_edge(self):
        foil = AirfoilDesign(0.02, 0.4, 0.12, "test")
        x, y = naca4_coordinates(foil, 101)
        self.assertEqual(len(x), 201)
        self.assertAlmostEqual(x[0], 1.0, places=6)
        self.assertAlmostEqual(x[-1], 1.0, places=6)
        self.assertAlmostEqual(y[0], y[-1], places=6)
        self.assertLess(float(np.min(x)), 0.001)
        self.assertGreater(float(np.max(y) - np.min(y)), 0.115)

    def test_symmetric_foil_has_zero_lift_near_zero_alpha(self):
        foil = AirfoilDesign(0.0, 0.4, 0.12)
        _, alpha_l0, cm = thin_airfoil_properties(foil)
        point = polar_point(foil, 0.0, 400_000.0)
        self.assertAlmostEqual(alpha_l0, 0.0, places=7)
        self.assertAlmostEqual(cm, 0.0, places=7)
        self.assertAlmostEqual(point.cl, 0.0, places=7)

    def test_cambered_foil_has_negative_zero_lift_angle(self):
        foil = AirfoilDesign(0.02, 0.4, 0.12)
        _, alpha_l0, cm = thin_airfoil_properties(foil)
        self.assertLess(math.degrees(alpha_l0), -1.0)
        self.assertLess(cm, 0.0)

    def test_inverse_design_alpha_reaches_requested_cl(self):
        foil = AirfoilDesign(0.03, 0.4, 0.12)
        alpha = alpha_for_cl(foil, 0.65, 500_000.0)
        point = polar_point(foil, alpha, 500_000.0)
        self.assertAlmostEqual(point.cl, 0.65, places=6)
        self.assertGreater(point.cd, 0.0)

    def test_polar_is_monotonic_in_attached_range(self):
        foil = AirfoilDesign(0.02, 0.4, 0.12)
        points = generate_polar(foil, 500_000.0, 0.0, [-4, -2, 0, 2, 4, 6])
        cls = [p.cl for p in points]
        self.assertTrue(all(b > a for a, b in zip(cls, cls[1:])))

    def test_cst_fit_reproduces_naca_and_respects_envelope(self):
        foil = AirfoilDesign(0.03, 0.4, 0.12)
        cst = fit_naca_to_cst(foil, order=3)
        _, naca_y = naca4_coordinates(foil, 201)
        _, cst_y = naca4_coordinates(cst, 201)
        self.assertLess(float(np.sqrt(np.mean((naca_y - cst_y) ** 2))), 0.001)
        self.assertTrue(
            cst_geometry_is_valid(
                cst,
                camber_bounds=(0.0, 0.06),
                camber_position_bounds=(0.25, 0.65),
                thickness_bounds=(0.10, 0.16),
            )
        )
        self.assertEqual(cst.family, "CST3")


if __name__ == "__main__":
    unittest.main()
