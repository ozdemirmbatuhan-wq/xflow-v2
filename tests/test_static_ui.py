from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StaticUiContractTests(unittest.TestCase):
    def test_javascript_element_references_exist_and_ids_are_unique(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        element_ids = re.findall(r'\bid="([^"]+)"', html)
        referenced = set(re.findall(r'\$\("([A-Za-z][A-Za-z0-9_-]*)"\)', javascript))
        self.assertEqual(len(element_ids), len(set(element_ids)))
        self.assertFalse(referenced.difference(element_ids))

    def test_reliability_controls_and_result_panels_are_present(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        for identifier in (
            "surrogateEnabled",
            "optimizerCheckpoint",
            "multiSeedRuns",
            "validationPanel",
            "diagnosticPanel",
            "paretoPanel",
            "stabilityPanel",
            "historyPanel",
            "budgetEscalation",
            "budgetPanel",
            "flow5WingOptimizer",
            "wingletOptimization",
            "wingletFields",
            "wingletNacaCode",
            "wingletHeightMin",
            "wingletHeightMax",
            "wingletCantMin",
            "wingletCantMax",
            "wingletToeMin",
            "wingletToeMax",
            "wingletTaperMin",
            "wingletTaperMax",
            "flow5WingletBudget",
            "hydroConstraintMode",
            "cavitationPanel",
            "cavitationMap",
            "cavitationSpanChart",
            "cavitationSpeedChart",
            "cavitationDepthChart",
        ):
            self.assertIn(f'id="{identifier}"', html)

    def test_invalid_fine_mesh_fallback_is_explained_in_results(self):
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("mesh.fine_mesh_valid === false", javascript)
        self.assertIn("ince ağ çözülemedi · final ağ sonucu korundu", javascript)

    def test_scalar_downloads_are_defined_inside_the_wing_result_renderer(self):
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        render_result = javascript.split("function renderResult(result) {", 1)[1].split(
            "async function optimize", 1
        )[0]
        declaration = render_result.index("const scalarDownloads")
        usage = render_result.index("...scalarDownloads")
        self.assertLess(declaration, usage)

    def test_highest_ld_comparison_and_downloads_are_wired(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        render_result = javascript.split("function renderResult(result) {", 1)[1].split(
            "async function optimize", 1
        )[0]
        declaration = render_result.index("const highestLdDownloads")
        usage = render_result.index("...highestLdDownloads")
        self.assertLess(declaration, usage)
        self.assertIn("result.highest_ld_comparison", javascript)
        self.assertIn("En yüksek L/D finalisti", javascript)
        self.assertIn("En yüksek L/D · STEP", javascript)
        self.assertIn("highest-ld/", html)

    def test_delivered_geometry_zero_aoa_contract_is_visible(self):
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("installed_geometry_validation", javascript)
        self.assertIn("Montaj açısı", javascript)
        self.assertIn("Dosyada 0° taşıma", javascript)
        self.assertIn("teslim geometrisi AoA 0°", javascript)
        self.assertIn("installedIncidence + twistAt", javascript)

    def test_winglet_controls_are_serialized_and_comparison_is_rendered(self):
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        for field in (
            "winglet_optimization_enabled",
            "winglet_naca_code",
            "winglet_height_min_m",
            "winglet_height_max_m",
            "winglet_cant_min_deg",
            "winglet_cant_max_deg",
            "winglet_toe_min_deg",
            "winglet_toe_max_deg",
            "winglet_taper_min",
            "winglet_taper_max",
            "flow5_winglet_candidate_budget",
        ):
            self.assertIn(field, javascript)
        self.assertIn("result.winglet_comparison", javascript)
        self.assertIn("Wingletli sonuç", javascript)

    def test_root_moment_constraint_is_removed_and_step_outputs_are_wired(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn('id="maxBending"', html)
        self.assertNotIn("max_root_bending_moment_nm", javascript)
        self.assertIn("selection_comparison", javascript)
        self.assertIn("Fizibilite önceliği olmasaydı", javascript)
        self.assertIn("wing_step_base64", javascript)
        self.assertIn("3B CAD · STEP", javascript)
        self.assertIn("Skaler alternatif · STEP", javascript)

    def test_cavitation_map_controls_and_unbounded_timeout_are_wired(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        timeout = re.search(r'<input\b[^>]*id="flow5Timeout"[^>]*>', html)
        self.assertIsNotNone(timeout)
        self.assertNotIn('max=', timeout.group(0))
        self.assertIn("üst sınır yok", html)
        self.assertIn('value="report_only"', html)
        self.assertIn("constraint_mode", javascript)
        self.assertIn("function renderCavitation", javascript)
        self.assertIn("result.hydro_analysis", javascript)
        self.assertIn("Kavitasyon haritası · JSON", javascript)

    def test_foil_and_wing_workflows_are_selectable_and_reusable(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        for identifier in (
            "optimizationMode",
            "wingAirfoilSource",
            "wingAirfoilDatFile",
            "savedAirfoilStatus",
        ):
            self.assertIn(f'id="{identifier}"', html)
        for mode in ("coupled", "foil_only", "wing_only"):
            self.assertIn(f'value="{mode}"', html)
        self.assertIn('workflow: { mode: workflowMode }', javascript)
        self.assertIn('aeropt.savedAirfoil.v1', javascript)
        self.assertIn('result.workflow_mode === "foil_only"', javascript)

    def test_completion_auto_downloads_one_bundle_and_cancel_shows_best_snapshot(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function autoDownloadResult", javascript)
        self.assertIn("ex.flow5_bundle_base64 || ex.foil_bundle_base64", javascript)
        self.assertIn("renderResult(result); autoDownloadResult(result)", javascript)
        self.assertIn("renderCancelledBest(state.best_so_far || {})", javascript)
        self.assertIn("aeropt-best-so-far.json", javascript)
        self.assertIn('id="resultEyebrow"', html)

    def test_equal_range_guidance_and_total_cpu_budget_are_visible(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("Min ve Max alanlarına aynı değeri girin", html)
        self.assertIn("Toplam CPU bütçesi", html)
        self.assertIn("NumPy/SciPy", html)
        self.assertIn('"Toplam CPU bütçesi"', javascript)
        self.assertIn('"İç sayısal havuzlar"', javascript)

    def test_coupled_round_control_explains_cl_and_re_feedback(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        control = re.search(
            r'<input\b[^>]*id="flow5CoupledIterations"[^>]*>', html
        )
        self.assertIsNotNone(control)
        self.assertIn('min="1"', control.group(0))
        self.assertIn('max="8"', control.group(0))
        self.assertIn("kanadın gerçek kesit C<sub>L</sub> dağılımı ve MAC/Re", html)

    def test_design_form_uses_application_validation(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        form = re.search(r'<form\b[^>]*\bid="designForm"[^>]*>', html)
        self.assertIsNotNone(form)
        self.assertIn("novalidate", form.group(0))
        self.assertIn('if (raw === "")', javascript)
        self.assertIn('if (input.min !== "")', javascript)
        self.assertIn('if (input.max !== "")', javascript)

    def test_single_reference_speed_point_is_selectable(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        control = re.search(r'<input\b[^>]*id="speedSamples"[^>]*>', html)
        self.assertIsNotNone(control)
        self.assertIn('min="1"', control.group(0))
        self.assertIn("1 = yalnız referans", html)

    def test_number_inputs_accept_free_continuous_values_or_plain_integers(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        number_inputs = re.findall(r'<input\b[^>]*type="number"[^>]*>', html)
        target_lift = next(tag for tag in number_inputs if 'id="targetLift"' in tag)
        self.assertIn('step="any"', target_lift)
        for tag in number_inputs:
            step_match = re.search(r'\bstep="([^"]+)"', tag)
            if step_match is None:
                continue
            step = step_match.group(1)
            self.assertIn(step, {"any", "1"}, tag)
            if step == "1":
                minimum = re.search(r'\bmin="([^"]+)"', tag)
                if minimum is not None:
                    self.assertTrue(float(minimum.group(1)).is_integer(), tag)

    def test_packaged_ui_smoke_is_part_of_windows_workflow(self):
        workflow = (ROOT / ".github" / "workflows" / "build-windows-flow5.yml").read_text(
            encoding="utf-8"
        )
        smoke = (ROOT / "ci" / "smoke_packaged_ui.py").read_text(encoding="utf-8")
        self.assertIn("verify-windows-artifact:", workflow)
        self.assertIn("actions/download-artifact@v5", workflow)
        self.assertIn("python ci/smoke_packaged_ui.py", workflow)
        self.assertIn('page.locator("#runButton").click()', smoke)
        self.assertIn('form => form.noValidate', smoke)


if __name__ == "__main__":
    unittest.main()
