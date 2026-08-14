const $ = (id) => document.getElementById(id);

const presets = {
  air: { density: 1.225, viscosity: 1.7894e-5, sound: 340.3 },
  fresh_water: { density: 999.1, viscosity: 1.138e-3, sound: 1481.0 },
  sea_water: { density: 1025.0, viscosity: 1.188e-3, sound: 1500.0 },
};

let lastResult = null;
let activeJobId = null;
const SAVED_AIRFOIL_KEY = "aeropt.savedAirfoil.v1";

function readSavedAirfoil() {
  try {
    const value = JSON.parse(localStorage.getItem(SAVED_AIRFOIL_KEY) || "null");
    return value && typeof value.dat === "string" && value.dat.trim() ? value : null;
  } catch { return null; }
}

async function readDatFile(inputId, missingMessage) {
  const file = $(inputId).files?.[0];
  if (!file) throw new Error(missingMessage);
  if (file.size > 500000) throw new Error("DAT dosyası 500 kB sınırını aşıyor.");
  return file.text();
}

function numberValue(id) {
  const input = $(id);
  const label = input.labels?.[0]?.textContent || id;
  const raw = String(input.value ?? "").trim();
  if (raw === "") throw new Error(`${label} için sayı girin.`);
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${label} için geçerli bir sayı girin.`);
  if (input.min !== "") {
    const minimum = Number(input.min);
    if (Number.isFinite(minimum) && value < minimum) {
      throw new Error(`${label} en az ${input.min} olmalı.`);
    }
  }
  if (input.max !== "") {
    const maximum = Number(input.max);
    if (Number.isFinite(maximum) && value > maximum) {
      throw new Error(`${label} en fazla ${input.max} olmalı.`);
    }
  }
  return value;
}

async function collectRequest() {
  const optionalCl = $("designCl").value.trim();
  const workflowMode = $("optimizationMode").value;
  let baselineProfile = $("baselineAirfoil").value;
  let baselineDat = "";
  if (workflowMode === "wing_only") {
    if ($("wingAirfoilSource").value === "saved") {
      const saved = readSavedAirfoil();
      if (!saved) throw new Error("Önce yalnız profil optimizasyonu çalıştırın veya bir DAT dosyası seçin.");
      baselineDat = saved.dat;
    } else {
      baselineDat = await readDatFile("wingAirfoilDatFile", "Kanat optimizasyonu için bir DAT dosyası seçin.");
    }
    baselineProfile = "custom_dat";
  } else if (baselineProfile === "custom_dat") {
    baselineDat = await readDatFile("baselineDatFile", "Özel başlangıç profili için bir DAT dosyası seçin.");
  }
  return {
    workflow: { mode: workflowMode },
    flow: {
      fluid: $("fluid").value,
      density_kg_m3: numberValue("density"),
      dynamic_viscosity_pa_s: numberValue("viscosity"),
      speed_of_sound_m_s: numberValue("soundSpeed"),
      speed_m_s: numberValue("speed"),
      speed_min_m_s: numberValue("speedMin"),
      speed_max_m_s: numberValue("speedMax"),
      speed_samples: numberValue("speedSamples"),
      target_lift_n: numberValue("targetLift"),
    },
    airfoil: {
      baseline_profile: baselineProfile,
      baseline_dat: baselineDat,
      cst_order: numberValue("cstOrder"),
      solver_coordinate_points: numberValue("solverCoordinatePoints"),
      camber_min_percent: numberValue("camberMin"),
      camber_max_percent: numberValue("camberMax"),
      camber_position_min_percent: numberValue("camberPosMin"),
      camber_position_max_percent: numberValue("camberPosMax"),
      thickness_min_percent: numberValue("thicknessMin"),
      thickness_max_percent: numberValue("thicknessMax"),
      design_cl: optionalCl === "" ? null : numberValue("designCl"),
    },
    wing: {
      span_min_m: numberValue("spanMin"), span_max_m: numberValue("spanMax"),
      root_chord_min_m: numberValue("chordMin"), root_chord_max_m: numberValue("chordMax"),
      taper_min: numberValue("taperMin"), taper_max: numberValue("taperMax"),
      sweep_min_deg: numberValue("sweepMin"), sweep_max_deg: numberValue("sweepMax"),
      tip_twist_min_deg: numberValue("twistMin"), tip_twist_max_deg: numberValue("twistMax"),
      alpha_min_deg: numberValue("alphaMin"), alpha_max_deg: numberValue("alphaMax"),
      multi_section_geometry_enabled: $("multiSectionGeometry").checked,
      mid_chord_factor_min: numberValue("midChordFactorMin"),
      mid_chord_factor_max: numberValue("midChordFactorMax"),
      mid_twist_min_deg: numberValue("midTwistMin"),
      mid_twist_max_deg: numberValue("midTwistMax"),
      winglet_optimization_enabled: $("wingletOptimization").checked,
      winglet_naca_code: $("wingletNacaCode").value.trim(),
      winglet_height_min_m: numberValue("wingletHeightMin"),
      winglet_height_max_m: numberValue("wingletHeightMax"),
      winglet_cant_min_deg: numberValue("wingletCantMin"),
      winglet_cant_max_deg: numberValue("wingletCantMax"),
      winglet_toe_min_deg: numberValue("wingletToeMin"),
      winglet_toe_max_deg: numberValue("wingletToeMax"),
      winglet_taper_min: numberValue("wingletTaperMin"),
      winglet_taper_max: numberValue("wingletTaperMax"),
    },
    solver: {
      quality: $("quality").value,
      seed: numberValue("seed"),
      lifting_line_modes: numberValue("modes"),
      airfoil_strategy: $("airfoilStrategy").value,
      flow5_runner_path: $("flow5RunnerPath").value.trim(),
      flow5_threads: numberValue("flow5Threads"),
      flow5_timeout_seconds: numberValue("flow5Timeout"),
      flow5_foil_candidate_budget: numberValue("flow5FoilBudget"),
      flow5_wing_candidate_budget: numberValue("flow5WingBudget"),
      flow5_winglet_candidate_budget: numberValue("flow5WingletBudget"),
      flow5_finalists: numberValue("flow5Finalists"),
      flow5_search_method: $("flow5SearchMethod").value,
      flow5_final_method: $("flow5FinalMethod").value,
      flow5_alpha_step_search_deg: numberValue("flow5SearchStep"),
      flow5_alpha_step_final_deg: numberValue("flow5FinalStep"),
      flow5_ncrit: numberValue("flow5Ncrit"),
      flow5_xtr_top: numberValue("flow5XtrTop"),
      flow5_xtr_bottom: numberValue("flow5XtrBottom"),
      flow5_foil_minimum_improvement_percent: numberValue("flow5MinImprovement"),
      flow5_foil_optimizer: $("flow5FoilOptimizer").value,
      flow5_wing_optimizer: $("flow5WingOptimizer").value,
      flow5_coupled_iterations: numberValue("flow5CoupledIterations"),
      flow5_coupling_cl_tolerance_percent: numberValue("flow5CouplingClTolerance"),
      flow5_coupling_objective_tolerance_percent: numberValue("flow5CouplingObjectiveTolerance"),
      flow5_spanwise_airfoil_optimization_enabled: $("spanwiseFoilOptimization").checked,
      flow5_spanwise_foil_budget_fraction: numberValue("spanwiseFoilBudgetFraction"),
      flow5_spanwise_foil_acceptance_tolerance_percent: numberValue("spanwiseFoilAcceptance"),
      flow5_search_chordwise_panels: numberValue("searchChordPanels"),
      flow5_search_half_span_panels: numberValue("searchSpanPanels"),
      flow5_final_chordwise_panels: numberValue("finalChordPanels"),
      flow5_final_half_span_panels: numberValue("finalSpanPanels"),
      flow5_convergence_chordwise_panels: numberValue("convergenceChordPanels"),
      flow5_convergence_half_span_panels: numberValue("convergenceSpanPanels"),
      flow5_mesh_convergence_enabled: $("meshConvergence").checked,
      flow5_mesh_cd_tolerance_percent: numberValue("meshCdTolerance"),
      flow5_mesh_alpha_tolerance_deg: numberValue("meshAlphaTolerance"),
      flow5_cache_enabled: $("evaluationCache").checked,
      flow5_cache_dir: "",
      flow5_checkpoint_enabled: $("optimizerCheckpoint").checked,
      flow5_checkpoint_dir: "",
      flow5_surrogate_enabled: $("surrogateEnabled").checked,
      flow5_surrogate_proposals_per_evaluation: numberValue("surrogateProposals"),
      flow5_surrogate_minimum_real_fraction: numberValue("surrogateMinFraction"),
      flow5_surrogate_maximum_error_percent: numberValue("surrogateMaxError"),
      flow5_surrogate_early_stop_improvement_percent: numberValue("surrogateEarlyStop"),
      flow5_budget_escalation_enabled: $("budgetEscalation").checked,
      flow5_budget_growth_factor: numberValue("budgetGrowthFactor"),
      flow5_budget_maximum_multiplier: numberValue("budgetMaxMultiplier"),
      flow5_budget_convergence_tolerance_percent: numberValue("budgetConvergenceTolerance"),
      flow5_budget_stable_checkpoints: numberValue("budgetStableChecks"),
      flow5_multi_seed_runs: numberValue("multiSeedRuns"),
      flow5_multi_seed_stride: 100003,
      flow5_multi_seed_objective_cv_tolerance_percent: numberValue("multiSeedObjectiveCv"),
      flow5_multi_seed_geometry_cv_tolerance_percent: numberValue("multiSeedGeometryCv"),
      xfoil_path: $("xfoilPath").value.trim(),
      xfoil_cl_tolerance_percent: numberValue("clTolerance"),
      xfoil_cd_tolerance_percent: numberValue("cdTolerance"),
      cst_candidate_budget: numberValue("candidateBudget"),
      parallel_workers: numberValue("parallelWorkers"),
      xfoil_timeout_seconds: numberValue("xfoilTimeout"),
    },
    structure: {
      enabled: $("structureEnabled").checked,
      youngs_modulus_gpa: numberValue("youngsModulus"),
      material_density_kg_m3: numberValue("materialDensity"),
      allowable_stress_mpa: numberValue("allowableStress"),
      safety_factor: numberValue("structuralSafetyFactor"),
      spar_height_fraction_of_foil: numberValue("sparHeightFraction"),
      spar_cap_width_fraction_chord: numberValue("sparCapWidth"),
      spar_cap_thickness_mm: numberValue("sparCapThickness"),
      skin_thickness_mm: numberValue("skinThickness"),
      torsion_box_width_fraction_chord: numberValue("torsionBoxWidth"),
      poisson_ratio: numberValue("poissonRatio"),
      max_tip_deflection_percent_semispan: numberValue("maxTipDeflection"),
      max_elastic_twist_deg: numberValue("maxElasticTwist"),
    },
    hydro: {
      enabled: $("hydroEnabled").checked,
      constraint_mode: $("hydroConstraintMode").value,
      submergence_depth_m: numberValue("submergenceDepth"),
      ambient_pressure_pa: numberValue("ambientPressure"),
      vapor_pressure_pa: numberValue("vaporPressure"),
      cavitation_safety_factor: numberValue("cavitationSafetyFactor"),
      minimum_submergence_chords: numberValue("minimumSubmergenceChords"),
      free_surface_screen_enabled: $("freeSurfaceScreen").checked,
    },
    validation: {
      enabled: $("validationEnabled").checked,
      force_closure_tolerance_percent: numberValue("validationForceTolerance"),
      drag_decomposition_tolerance_percent: numberValue("validationDragTolerance"),
      minimum_span_efficiency: 0.20,
      maximum_span_efficiency: 1.20,
    },
  };
}

function applyDefaults(data) {
  const map = {
    optimizationMode: data.workflow?.mode || "coupled",
    fluid: data.flow.fluid, density: data.flow.density_kg_m3,
    viscosity: data.flow.dynamic_viscosity_pa_s, soundSpeed: data.flow.speed_of_sound_m_s,
    speed: data.flow.speed_m_s, speedMin: data.flow.speed_min_m_s,
    speedMax: data.flow.speed_max_m_s, speedSamples: data.flow.speed_samples,
    targetLift: data.flow.target_lift_n,
    baselineAirfoil: data.airfoil.baseline_profile,
    cstOrder: data.airfoil.cst_order,
    solverCoordinatePoints: data.airfoil.solver_coordinate_points,
    camberMin: data.airfoil.camber_min_percent, camberMax: data.airfoil.camber_max_percent,
    camberPosMin: data.airfoil.camber_position_min_percent, camberPosMax: data.airfoil.camber_position_max_percent,
    thicknessMin: data.airfoil.thickness_min_percent, thicknessMax: data.airfoil.thickness_max_percent,
    designCl: data.airfoil.design_cl ?? "", spanMin: data.wing.span_min_m, spanMax: data.wing.span_max_m,
    chordMin: data.wing.root_chord_min_m, chordMax: data.wing.root_chord_max_m,
    taperMin: data.wing.taper_min, taperMax: data.wing.taper_max,
    sweepMin: data.wing.sweep_min_deg, sweepMax: data.wing.sweep_max_deg,
    twistMin: data.wing.tip_twist_min_deg, twistMax: data.wing.tip_twist_max_deg,
    alphaMin: data.wing.alpha_min_deg, alphaMax: data.wing.alpha_max_deg,
    quality: data.solver.quality,
    midChordFactorMin: data.wing.mid_chord_factor_min, midChordFactorMax: data.wing.mid_chord_factor_max,
    midTwistMin: data.wing.mid_twist_min_deg, midTwistMax: data.wing.mid_twist_max_deg,
    wingletHeightMin: data.wing.winglet_height_min_m,
    wingletHeightMax: data.wing.winglet_height_max_m,
    wingletCantMin: data.wing.winglet_cant_min_deg,
    wingletCantMax: data.wing.winglet_cant_max_deg,
    wingletToeMin: data.wing.winglet_toe_min_deg,
    wingletToeMax: data.wing.winglet_toe_max_deg,
    wingletTaperMin: data.wing.winglet_taper_min,
    wingletTaperMax: data.wing.winglet_taper_max,
    seed: data.solver.seed, modes: data.solver.lifting_line_modes,
    airfoilStrategy: data.solver.airfoil_strategy, xfoilPath: data.solver.xfoil_path,
    flow5RunnerPath: data.solver.flow5_runner_path,
    flow5Threads: data.solver.flow5_threads,
    flow5Timeout: data.solver.flow5_timeout_seconds,
    flow5FoilBudget: data.solver.flow5_foil_candidate_budget,
    flow5WingBudget: data.solver.flow5_wing_candidate_budget,
    flow5WingletBudget: data.solver.flow5_winglet_candidate_budget,
    flow5Finalists: data.solver.flow5_finalists,
    flow5SearchMethod: data.solver.flow5_search_method,
    flow5FinalMethod: data.solver.flow5_final_method,
    flow5SearchStep: data.solver.flow5_alpha_step_search_deg,
    flow5FinalStep: data.solver.flow5_alpha_step_final_deg,
    flow5Ncrit: data.solver.flow5_ncrit,
    flow5XtrTop: data.solver.flow5_xtr_top,
    flow5XtrBottom: data.solver.flow5_xtr_bottom,
    flow5MinImprovement: data.solver.flow5_foil_minimum_improvement_percent,
    flow5FoilOptimizer: data.solver.flow5_foil_optimizer,
    flow5WingOptimizer: data.solver.flow5_wing_optimizer,
    flow5CoupledIterations: data.solver.flow5_coupled_iterations,
    flow5CouplingClTolerance: data.solver.flow5_coupling_cl_tolerance_percent,
    flow5CouplingObjectiveTolerance: data.solver.flow5_coupling_objective_tolerance_percent,
    spanwiseFoilBudgetFraction: data.solver.flow5_spanwise_foil_budget_fraction,
    spanwiseFoilAcceptance: data.solver.flow5_spanwise_foil_acceptance_tolerance_percent,
    searchChordPanels: data.solver.flow5_search_chordwise_panels,
    searchSpanPanels: data.solver.flow5_search_half_span_panels,
    finalChordPanels: data.solver.flow5_final_chordwise_panels,
    finalSpanPanels: data.solver.flow5_final_half_span_panels,
    convergenceChordPanels: data.solver.flow5_convergence_chordwise_panels,
    convergenceSpanPanels: data.solver.flow5_convergence_half_span_panels,
    meshCdTolerance: data.solver.flow5_mesh_cd_tolerance_percent,
    meshAlphaTolerance: data.solver.flow5_mesh_alpha_tolerance_deg,
    surrogateProposals: data.solver.flow5_surrogate_proposals_per_evaluation,
    surrogateMinFraction: data.solver.flow5_surrogate_minimum_real_fraction,
    surrogateMaxError: data.solver.flow5_surrogate_maximum_error_percent,
    surrogateEarlyStop: data.solver.flow5_surrogate_early_stop_improvement_percent,
    budgetGrowthFactor: data.solver.flow5_budget_growth_factor,
    budgetMaxMultiplier: data.solver.flow5_budget_maximum_multiplier,
    budgetConvergenceTolerance: data.solver.flow5_budget_convergence_tolerance_percent,
    budgetStableChecks: data.solver.flow5_budget_stable_checkpoints,
    multiSeedRuns: data.solver.flow5_multi_seed_runs,
    multiSeedObjectiveCv: data.solver.flow5_multi_seed_objective_cv_tolerance_percent,
    multiSeedGeometryCv: data.solver.flow5_multi_seed_geometry_cv_tolerance_percent,
    validationForceTolerance: data.validation.force_closure_tolerance_percent,
    validationDragTolerance: data.validation.drag_decomposition_tolerance_percent,
    wingletNacaCode: data.wing.winglet_naca_code,
    youngsModulus: data.structure.youngs_modulus_gpa,
    materialDensity: data.structure.material_density_kg_m3,
    allowableStress: data.structure.allowable_stress_mpa,
    structuralSafetyFactor: data.structure.safety_factor,
    sparHeightFraction: data.structure.spar_height_fraction_of_foil,
    sparCapWidth: data.structure.spar_cap_width_fraction_chord,
    sparCapThickness: data.structure.spar_cap_thickness_mm,
    skinThickness: data.structure.skin_thickness_mm,
    torsionBoxWidth: data.structure.torsion_box_width_fraction_chord,
    poissonRatio: data.structure.poisson_ratio,
    maxTipDeflection: data.structure.max_tip_deflection_percent_semispan,
    maxElasticTwist: data.structure.max_elastic_twist_deg,
    submergenceDepth: data.hydro.submergence_depth_m,
    hydroConstraintMode: data.hydro.constraint_mode,
    ambientPressure: data.hydro.ambient_pressure_pa,
    vaporPressure: data.hydro.vapor_pressure_pa,
    cavitationSafetyFactor: data.hydro.cavitation_safety_factor,
    minimumSubmergenceChords: data.hydro.minimum_submergence_chords,
    clTolerance: data.solver.xfoil_cl_tolerance_percent,
    cdTolerance: data.solver.xfoil_cd_tolerance_percent,
    candidateBudget: data.solver.cst_candidate_budget,
    parallelWorkers: data.solver.parallel_workers,
    xfoilTimeout: data.solver.xfoil_timeout_seconds,
  };
  Object.entries(map).forEach(([id, value]) => { $(id).value = value; });
  const checks = {
    multiSectionGeometry: data.wing.multi_section_geometry_enabled,
    wingletOptimization: data.wing.winglet_optimization_enabled,
    spanwiseFoilOptimization: data.solver.flow5_spanwise_airfoil_optimization_enabled,
    meshConvergence: data.solver.flow5_mesh_convergence_enabled,
    evaluationCache: data.solver.flow5_cache_enabled,
    optimizerCheckpoint: data.solver.flow5_checkpoint_enabled,
    surrogateEnabled: data.solver.flow5_surrogate_enabled,
    budgetEscalation: data.solver.flow5_budget_escalation_enabled,
    validationEnabled: data.validation.enabled,
    structureEnabled: data.structure.enabled,
    hydroEnabled: data.hydro.enabled,
    freeSurfaceScreen: data.hydro.free_surface_screen_enabled,
  };
  Object.entries(checks).forEach(([id, value]) => { $(id).checked = Boolean(value); });
  toggleSolverSettings();
  toggleBaselineInput();
  toggleOptionalPanels();
  toggleWorkflowMode();
}

function showState(name) {
  ["emptyState", "loadingState", "errorState", "resultState"].forEach((id) => $(id).classList.add("hidden"));
  $(name).classList.remove("hidden");
}

function startLoading() {
  $("loadingMessage").textContent = "Optimizasyon işi hazırlanıyor…";
  $("progressBar").style.width = "0%";
  $("progressPercent").textContent = "%0.0";
  $("progressCount").textContent = "";
  $("cancelButton").disabled = false;
  $("runButton").disabled = true;
  showState("loadingState");
}

function stopLoading() {
  $("runButton").disabled = false;
  activeJobId = null;
}

function updateProgress(progress = {}) {
  const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
  $("progressBar").style.width = `${percent}%`;
  $("progressPercent").textContent = `%${fmt(percent,1)}`;
  $("loadingMessage").textContent = progress.message || "flow5 çözücüsü çalışıyor…";
  const iteration = progress.iteration_total > 1 ? ` · döngü ${progress.iteration}/${progress.iteration_total}` : "";
  const seedRun = progress.seed_run_total > 1 ? ` · seed ${progress.seed_run}/${progress.seed_run_total}` : "";
  $("progressCount").textContent = progress.total ? `${progress.current}/${progress.total}${iteration}${seedRun}` : `${iteration}${seedRun}`;
}

function pause(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }

function fmt(value, digits = 2) {
  if (!Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("tr-TR", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(Number(value));
}

function sci(value) {
  if (!Number.isFinite(Number(value))) return "—";
  return Number(value).toExponential(2).replace("e+", "e");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function svgLineChart(container, points, xKey, yKey, options = {}) {
  if (!Array.isArray(points) || points.length < 2) {
    container.innerHTML = `<div class="chart-empty">${escapeHtml(options.emptyText || "Bu sonuç için dağılım verisi runner tarafından dışa aktarılmadı; .fl5 projesinde inceleyin.")}</div>`;
    return;
  }
  const width = 520, height = 200, pad = { l: 42, r: 15, t: 12, b: 30 };
  const xs = points.map((p) => Number(p[xKey])); const ys = points.map((p) => Number(p[yKey]));
  let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (Math.abs(xmax - xmin) < 1e-12) { xmin -= 1; xmax += 1; }
  if (Math.abs(ymax - ymin) < 1e-12) { ymin -= 1; ymax += 1; }
  const yMargin = (ymax - ymin) * 0.12; ymin -= yMargin; ymax += yMargin;
  const sx = (x) => pad.l + (x - xmin) / (xmax - xmin) * (width - pad.l - pad.r);
  const sy = (y) => height - pad.b - (y - ymin) / (ymax - ymin) * (height - pad.t - pad.b);
  const path = points.map((p, i) => `${i ? "L" : "M"}${sx(Number(p[xKey])).toFixed(1)},${sy(Number(p[yKey])).toFixed(1)}`).join(" ");
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const gy = pad.t + i * (height - pad.t - pad.b) / 4;
    const value = ymax - i * (ymax - ymin) / 4;
    grid += `<line class="grid-line" x1="${pad.l}" x2="${width-pad.r}" y1="${gy}" y2="${gy}"/><text class="axis-label" x="${pad.l-7}" y="${gy+3}" text-anchor="end">${fmt(value, options.yDigits ?? 2)}</text>`;
  }
  for (let i = 0; i <= 4; i++) {
    const gx = pad.l + i * (width - pad.l - pad.r) / 4;
    const value = xmin + i * (xmax - xmin) / 4;
    grid += `<text class="axis-label" x="${gx}" y="${height-10}" text-anchor="middle">${fmt(value, options.xDigits ?? 1)}</text>`;
  }
  let reference = "";
  if (Number.isFinite(options.referenceX)) reference += `<line class="reference-line" x1="${sx(options.referenceX)}" x2="${sx(options.referenceX)}" y1="${pad.t}" y2="${height-pad.b}"/>`;
  if (Number.isFinite(options.referenceY) && options.referenceY >= ymin && options.referenceY <= ymax) reference += `<line class="reference-line" x1="${pad.l}" x2="${width-pad.r}" y1="${sy(options.referenceY)}" y2="${sy(options.referenceY)}"/>`;
  const area = `${path} L${sx(xs[xs.length-1])},${height-pad.b} L${sx(xs[0])},${height-pad.b} Z`;
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(options.label || "çizgi grafik")}"><defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ed693e" stop-opacity=".32"/><stop offset="1" stop-color="#ed693e" stop-opacity="0"/></linearGradient></defs>${grid}${reference}<path class="data-area" d="${area}"/><path class="data-line" d="${path}"/></svg>`;
}

function renderFoil(result) {
  const points = result.airfoil_coordinates; const width = 520, height = 190, pad = 24;
  const ys = points.map((p) => p.y_over_c); const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const scaleX = (x) => pad + x * (width - 2 * pad);
  const scaleY = (y) => height / 2 - y / Math.max(ymax - ymin, .1) * 115;
  const path = points.map((p, i) => `${i ? "L" : "M"}${scaleX(p.x_over_c).toFixed(1)},${scaleY(p.y_over_c).toFixed(1)}`).join(" ") + " Z";
  $("foilChart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Optimize profil kesiti"><line class="grid-line" x1="${pad}" x2="${width-pad}" y1="${height/2}" y2="${height/2}"/><path class="foil-shape" d="${path}"/><text class="axis-label" x="${pad}" y="${height-4}">0</text><text class="axis-label" x="${width-pad}" y="${height-4}" text-anchor="end">x/c = 1</text></svg>`;
  const f = result.airfoil;
  $("foilStats").innerHTML = `<div><small>Camber</small><strong>%${fmt(f.max_camber*100,1)}</strong></div><div><small>Konum</small><strong>%${fmt(f.camber_position*100,0)} c</strong></div><div><small>Kalınlık</small><strong>%${fmt(f.thickness*100,1)}</strong></div>`;
}

function renderPlanform(result) {
  const g = result.wing.geometry; const foil = result.airfoil_coordinates; const width = 520, height = 190;
  const half = g.span / 2; const root = g.root_chord; const mainHalf = Number(g.main_semispan ?? half);
  const winglet = Boolean(g.winglet_active && Number(g.winglet_height) > 0);
  const midChord = Number(g.mid_chord ?? .5*(g.root_chord+g.tip_chord));
  const midTwist = Number(g.effective_mid_twist_deg ?? .5*g.tip_twist_deg);
  const chordAt = (eta) => eta <= .5 ? root + 2*eta*(midChord-root) : midChord + 2*(eta-.5)*(g.tip_chord-midChord);
  const twistAt = (eta) => eta <= .5 ? 2*eta*midTwist : midTwist + 2*(eta-.5)*(g.tip_twist_deg-midTwist);
  const mainPoint = (side, eta, xc, zc) => {
    const chord = chordAt(eta);
    const xOffset = .25*root + eta*mainHalf*Math.tan(g.sweep_deg*Math.PI/180) - .25*chord;
    const twist = twistAt(eta) * Math.PI / 180;
    const xq = (xc-.25)*chord, z = zc*chord;
    return [xOffset + .25*chord + xq*Math.cos(twist) + z*Math.sin(twist), side*eta*mainHalf, -xq*Math.sin(twist) + z*Math.cos(twist)];
  };
  const wingletPoint = (side, fraction, xc, zc) => {
    const cant = Number(g.winglet_cant_deg) * Math.PI / 180;
    const length = Number(g.winglet_developed_length); const distance = fraction*length;
    const rootChord = Number(g.winglet_root_chord ?? g.tip_chord);
    const tipChord = Number(g.winglet_tip_chord ?? rootChord);
    const chord = rootChord + fraction*(tipChord-rootChord);
    const rootOffset = Number(g.tip_le_offset); const tipOffset = Number(g.winglet_tip_le_offset ?? rootOffset);
    const xOffset = rootOffset + fraction*(tipOffset-rootOffset);
    const twist = (Number(g.tip_twist_deg) + fraction*Number(g.winglet_toe_deg || 0))*Math.PI/180;
    const xq = (xc-.25)*chord, z = zc*chord;
    const xr = xq*Math.cos(twist) + z*Math.sin(twist);
    const normal = -xq*Math.sin(twist) + z*Math.cos(twist);
    return [
      xOffset + .25*chord + xr,
      side*(mainHalf + distance*Math.cos(cant)) - side*Math.sin(cant)*normal,
      distance*Math.sin(cant) + Math.cos(cant)*normal,
    ];
  };
  const stationPoint = (station, xc, zc) => station.kind === "winglet"
    ? wingletPoint(station.side, station.fraction, xc, zc)
    : mainPoint(station.side, station.fraction, xc, zc);
  const zScale = Math.max(root, Number(g.winglet_height || 0)*1.35, .05);
  const project = ([x,y,z]) => [width*.48 + y/half*width*.39 + x/root*width*.105, height*.62 + x/root*height*.18 - z/zScale*height*.42];
  const linePath = (points) => points.map((point,index) => { const [u,v]=project(point); return `${index?"L":"M"}${u.toFixed(1)},${v.toFixed(1)}`; }).join(" ");
  const mainStations = [-1,-.75,-.5,-.25,0,.25,.5,.75,1].map((fraction) => ({kind:"main",side:fraction<0?-1:1,fraction:Math.abs(fraction)}));
  const sectionStations = [...mainStations];
  if (winglet) sectionStations.push(
    {kind:"winglet",side:-1,fraction:.5},{kind:"winglet",side:-1,fraction:1},
    {kind:"winglet",side:1,fraction:.5},{kind:"winglet",side:1,fraction:1},
  );
  const mainOutlineWorld = [
    mainPoint(-1,1,0,0),mainPoint(1,0,0,0),mainPoint(1,1,0,0),
    mainPoint(1,1,1,0),mainPoint(1,0,1,0),mainPoint(-1,1,1,0),
  ];
  const polygon = (points, css="wing-surface") => `<polygon class="${css}" points="${points.map((point) => project(point).map((value)=>value.toFixed(1)).join(",")).join(" ")}"/>`;
  let surfaces = polygon(mainOutlineWorld);
  if (winglet) {
    [-1,1].forEach((side) => { surfaces += polygon([
      wingletPoint(side,0,0,0),wingletPoint(side,1,0,0),wingletPoint(side,1,1,0),wingletPoint(side,0,1,0),
    ],"wing-surface winglet-surface"); });
  }
  const sampledFoil = foil.filter((_,index)=>index%3===0);
  const sections = sectionStations.map((station) => `<path class="wing-section" d="${linePath(sampledFoil.map((point)=>stationPoint(station,point.x_over_c,point.y_over_c)))} Z"/>`).join("");
  let mesh = [0,.25,.5,.75,1].map((xc) => `<path class="wing-mesh" d="${linePath(mainStations.map((station)=>stationPoint(station,xc,0)))}"/>`).join("");
  if (winglet) [-1,1].forEach((side) => { [0,.5,1].forEach((xc) => { mesh += `<path class="wing-mesh" d="${linePath([0,.5,1].map((fraction)=>wingletPoint(side,fraction,xc,0)))}"/>`; }); });
  $("planformChart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Optimize kanadın ${winglet?"wingletli ":""}üç boyutlu izometrik görünüşü">${surfaces}${mesh}${sections}</svg>`;
  const lastStat = winglet
    ? `<div><small>Winglet</small><strong>${fmt(g.winglet_height,3)} m · ${fmt(g.winglet_cant_deg,1)}°</strong></div>`
    : `<div><small>Alan</small><strong>${fmt(g.area,3)} m²</strong></div>`;
  $("planformStats").innerHTML = `<div><small>Kök chord</small><strong>${fmt(g.root_chord,3)} m</strong></div><div><small>Orta chord</small><strong>${fmt(midChord,3)} m</strong></div><div><small>Uç chord</small><strong>${fmt(g.tip_chord,3)} m</strong></div>${lastStat}`;
}

function metric(label, value, unit) { return `<div class="metric"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong><em>${escapeHtml(unit || "")}</em></div>`; }

function renderComparison(result) {
  const winglet = result.winglet_comparison || {};
  const policy = result.wing_optimization?.selection_comparison || {};
  const scalar = policy.scalar_only || {};
  const feasibility = policy.feasibility_first || {};
  const policyWing = feasibility.wing || result.wing;
  const scalarWing = scalar.wing || scalar.coarse_wing;
  let policyHtml = "";
  if (scalar.same_as_feasibility_first) {
    policyHtml = `<div class="solver-help"><b>Fizibilite önceliği olmasaydı da aynı kanat seçilecekti.</b> Skaler toplam amaç ve sert-kısıt önceliği bu aday havuzunda ayrışmadı. Kök momenti seçim koşulu değildir; yalnız telemetridir.</div>`;
  } else if (scalarWing) {
    const policyRow = (label, selectedValue, scalarValue, digits=2, unit="") => `<tr><td>${escapeHtml(label)}</td><td class="best">${fmt(selectedValue,digits)} ${unit}</td><td>${fmt(scalarValue,digits)} ${unit}</td></tr>`;
    const selectedState = feasibility.feasible ? "fizibil" : `ihlal ${fmt(feasibility.constraint_violation,4)}`;
    const scalarState = scalar.feasible ? "fizibil" : `ihlal ${fmt(scalar.constraint_violation,4)}`;
    policyHtml = `<div class="solver-help"><b>Fizibilite önceliği seçim karşılaştırması</b> · seçilen: ${escapeHtml(selectedState)} · yalnız skaler amaç: ${escapeHtml(scalarState)}${scalar.available ? "" : " · yüksek çözünürlüklü çıktı üretilemedi"}</div><table class="comparison-table"><thead><tr><th>Gösterge</th><th>Fizibilite öncelikli</th><th>Yalnız skaler amaç</th></tr></thead><tbody>${policyRow("Toplam amaç",feasibility.objective,scalar.objective,6)}${policyRow("Toplam sürükleme",policyWing.drag_n,scalarWing.drag_n,2,"N")}${policyRow("L/D",policyWing.ld,scalarWing.ld,1)}${policyRow("Açıklık",policyWing.geometry.span,scalarWing.geometry.span,3,"m")}${policyRow("Alan",policyWing.geometry.area,scalarWing.geometry.area,3,"m²")}${policyRow("Stall kullanımı",policyWing.stall_ratio,scalarWing.stall_ratio,3)}</tbody></table><div class="solver-help">Kök momenti her iki kanatta da yalnız rapor telemetrisidir; amaç, finalist sırası veya fizibilite kararına girmez.</div>`;
  } else if (policy.scalar_only) {
    policyHtml = `<div class="solver-help"><b>Skaler-puan alternatifi teslim edilemedi.</b> ${escapeHtml(scalar.error || "Yüksek çözünürlüklü son çözüm yakınsamadı.")}</div>`;
  }
  let primaryHtml;
  if (winglet.performed) {
    const p = winglet.planar, w = winglet.winglet;
    const selected = winglet.selection === "winglet" ? "Winglet seçildi" : "Planar seçildi";
    const row = (label, planar, withWinglet, digits=2, unit="") => {
      const delta = Number(withWinglet)-Number(planar);
      return `<tr><td>${label}</td><td class="${winglet.selection==="planar"?"best":""}">${fmt(planar,digits)} ${unit}</td><td class="${winglet.selection==="winglet"?"best":""}">${fmt(withWinglet,digits)} ${unit}</td><td>${delta>=0?"+":""}${fmt(delta,digits)} ${unit}</td></tr>`;
    };
    const geometryRows = `${row("Winglet yüksekliği",0,w.geometry.winglet_height,3,"m")}${row("Winglet cant",0,w.geometry.winglet_cant_deg,1,"°")}${row("Winglet toe",0,w.geometry.winglet_toe_deg,1,"°")}${row("Winglet taper",0,w.geometry.winglet_taper,3)}`;
    primaryHtml = `<p class="solver-help"><b>${selected}</b> · profil–kanat döngüsü dışında tek son işlem · sabit ${escapeHtml(winglet.airfoil_name || "NACA")} · aynı izdüşüm açıklığı ve taşıma hedefi · ${escapeHtml(winglet.selection_reason || "kısıtlı amaç karşılaştırması")}</p><table class="comparison-table"><thead><tr><th>Gösterge</th><th>Wingletsiz optimum</th><th>Wingletli sonuç</th><th>Wingletli − wingletsiz</th></tr></thead><tbody>${row("Toplam sürükleme",p.drag_n,w.drag_n,2,"N")}${row("L/D",p.ld,w.ld,1)}${row("Profil Cᴅ",p.cd_profile,w.cd_profile,5)}${row("İndüklenmiş Cᴅ",p.cd_induced,w.cd_induced,5)}${row("İndüklenmiş pay",p.induced_drag_fraction_percent,w.induced_drag_fraction_percent,1,"%")}${geometryRows}${row("Kök eğilme momenti (telemetri)",p.root_bending_moment_nm,w.root_bending_moment_nm,1,"N·m")}</tbody></table>`;
  } else {
    const o = result.wing, b = result.rectangular_baseline;
    const row = (label, optimum, baseline, digits=2, unit="") => `<tr><td>${label}</td><td class="best">${fmt(optimum,digits)} ${unit}</td><td>${fmt(baseline,digits)} ${unit}</td></tr>`;
    primaryHtml = `<table class="comparison-table"><thead><tr><th>Gösterge</th><th>Optimize</th><th>Dikdörtgen</th></tr></thead><tbody>${row("Toplam sürükleme",o.drag_n,b.drag_n,2,"N")}${row("L/D",o.ld,b.ld,1)}${row("Profil Cᴅ",o.cd_profile,b.cd_profile,4)}${row("İndüklenmiş Cᴅ",o.cd_induced,b.cd_induced,4)}${row("Span verimi",o.span_efficiency,b.span_efficiency,3)}${row("Kök eğilme momenti (telemetri)",o.root_bending_moment_nm,b.root_bending_moment_nm,1,"N·m")}</tbody></table>`;
  }
  $("comparisonTable").innerHTML = `${primaryHtml}${policyHtml}`;
}

function downloadLink(filename, contents, type, label) {
  const blob = new Blob([contents], { type }); const url = URL.createObjectURL(blob);
  return `<a class="download-button" href="${url}" download="${escapeHtml(filename)}"><span>${escapeHtml(label)}</span><b>↓</b></a>`;
}

function base64DownloadLink(filename, encoded, type, label) {
  const raw = atob(encoded); const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  return `<a class="download-button bundle-button" href="${url}" download="${escapeHtml(filename)}"><span>${escapeHtml(label)}</span><b>↓</b></a>`;
}

function renderXfoil(result) {
  const panel = $("xfoilPanel");
  if (result.flow5_native_analysis) {
    const analysis = result.flow5_native_analysis;
    const cruise = result.airfoil_optimization.final_cruise_point || {};
    const baseline = result.airfoil_optimization.baseline || {};
    const selection = result.airfoil_optimization.selection || {};
    const solver = analysis.solver || {};
    const row = (label, value, unit="") => `<tr><td>${escapeHtml(label)}</td><td class="best">${escapeHtml(value)} ${escapeHtml(unit)}</td></tr>`;
    panel.classList.remove("hidden");
    $("solverSummaryTitle").textContent = "flow5-native çözüm zinciri";
    $("xfoilStatusTag").textContent = `flow5 ${solver.version || "API"} · ${analysis.wing_solver_final}`;
    const selectionText = selection.selected_baseline ? `${baseline.display_name || "Baseline"} korundu` : `${selection.selected_family || "CST"} optimize profil`;
    const improvement = selection.selected_improvement_vs_baseline_percent;
    const mesh = analysis.mesh_convergence || {};
    const cache = analysis.evaluation_cache || {};
    const coupled = result.coupled_design || {};
    const spanwise = result.spanwise_airfoil_optimization || {};
    const foilSurrogate = result.airfoil_optimization.surrogate || {};
    const wingSurrogate = result.wing_optimization.surrogate || {};
    const selectionComparison = result.wing_optimization.selection_comparison || {};
    const scalarSelection = selectionComparison.scalar_only || {};
    const foilCheckpoint = result.airfoil_optimization.checkpoint || {};
    const wingCheckpoint = result.wing_optimization.checkpoint || {};
    const foilBudget = result.airfoil_optimization.budget_convergence || {};
    const wingBudget = result.wing_optimization.budget_convergence || {};
    const multiObjective = result.wing_optimization.multi_objective || {};
    const stability = result.multi_seed_stability || {};
    const winglet = result.winglet_comparison || {};
    const summaryRows = [
      row("Başlangıç profili", baseline.display_name || "Eppler E818"),
      row("DAT → CST uyumu", `RMS ${fmt(Number(baseline.fit_rms_over_c)*100,4)} %c`),
      row("Profil temsili", `CST${result.airfoil_optimization.cst_order}`),
      row("Solver koordinatı", String(result.airfoil_optimization.solver_coordinate_points), "nokta"),
      row("Profil seçimi", selectionText), row("Baseline iyileşmesi", improvement == null ? "—" : fmt(improvement,2), "%"),
      row("Foil optimizeri", result.solver_run.foil_optimizer || "differential_evolution"),
      row("Kanat optimizeri", result.solver_run.wing_optimizer || "differential_evolution"),
      row("Winglet karşılaştırması", winglet.performed ? `${winglet.airfoil_name || "NACA"} · döngü sonrası 1 kez · ${winglet.selection === "winglet" ? "winglet seçildi" : "planar seçildi"} · ΔD %${fmt(winglet.delta_winglet_vs_planar?.drag_percent,2)}` : winglet.enabled ? "döngü sonrası aşama tamamlanamadı" : "kapalı · hiç çalıştırılmadı"),
      row("Fizibilite önceliği olmasaydı", scalarSelection.same_as_feasibility_first ? "aynı kanat" : scalarSelection.available ? `${scalarSelection.stage || "ayrı"} alternatif ayrıca verildi` : "alternatif son çözüm üretilemedi"),
      row("Kanat amaçları", multiObjective.enabled ? `${(multiObjective.objective_specs || []).length} amaç · Pareto rank + crowding` : "skaler amaç"),
      row(
        "Bağlı foil–kanat",
        `${coupled.iterations_completed || 1}/${coupled.iterations_requested || 1} tur · final ${coupled.selected_iteration || 1}${coupled.converged ? " · yakınsadı" : ""}`,
      ),
      row("Spanwise profil", spanwise.selected ? "kök / orta / uç" : "tek profil"),
      row("Foil çözücüsü", analysis.foil_solver), row("Akış noktası", String(result.flow.speed_samples)),
      row("CST foil adayı", String(result.airfoil_optimization.candidates_evaluated)),
      row("Kanat adayı", String(result.wing_optimization.candidates_evaluated)),
      row("Winglet adayı", winglet.performed ? String(winglet.winglet?.candidates_evaluated || 0) : "—"),
      row("Bütçe · foil", foilBudget.converged === true ? `${foilBudget.evaluations_completed}/${foilBudget.maximum_budget} · yeterli` : foilBudget.converged === false ? `${foilBudget.evaluations_completed}/${foilBudget.maximum_budget} · artır` : `${foilBudget.evaluations_completed || result.airfoil_optimization.candidates_evaluated} · sabit`),
      row("Bütçe · kanat", wingBudget.converged === true ? `${wingBudget.evaluations_completed}/${wingBudget.maximum_budget} · yeterli` : wingBudget.converged === false ? `${wingBudget.evaluations_completed}/${wingBudget.maximum_budget} · artır` : `${wingBudget.evaluations_completed || result.wing_optimization.candidates_evaluated} · sabit`),
      row("Kanat taraması", analysis.wing_solver_search), row("Son doğrulama", analysis.wing_solver_final),
      row("Mesh yakınsaması", mesh.enabled ? (mesh.passed ? `geçti · ΔCD %${fmt(mesh.max_cd_change_percent,2)}` : "tolerans dışı") : "kapalı"),
      row("Önbellekten alınan", String(cache.hits || 0), "aday"),
      row("Surrogate · foil", foilSurrogate.enabled ? `${foilSurrogate.proposals_screened || 0} öneri elendi · ${foilSurrogate.real_solver_evaluations || 0} gerçek` : "kapalı"),
      row("Surrogate · kanat", wingSurrogate.enabled ? `${wingSurrogate.proposals_screened || 0} öneri elendi · ${wingSurrogate.real_solver_evaluations || 0} gerçek` : "kapalı"),
      row("Checkpoint · foil", foilCheckpoint.resumed ? `${foilCheckpoint.evaluations_restored} değerlendirmeden sürdü` : foilCheckpoint.enabled ? "etkin · yeni koşu" : "kapalı"),
      row("Checkpoint · kanat", wingCheckpoint.resumed ? `${wingCheckpoint.evaluations_restored} değerlendirmeden sürdü` : wingCheckpoint.enabled ? "etkin · yeni koşu" : "kapalı"),
      row("Multi-seed", stability.enabled ? `${stability.runs_completed}/${stability.runs_requested} · CV %${fmt(stability.objective_cv_percent,2)}` : "1 koşu"),
      row("flow5 içi çekirdek", String(result.solver_run.flow5_threads)), row("Referans α", fmt(cruise.alpha_deg,2), "°"),
      row("Referans CL", fmt(cruise.cl,4)), row("Referans CD", fmt(cruise.cd,5)), row("2B L/D", fmt(cruise.ld,1)),
    ];
    $("xfoilSummary").innerHTML = `<table class="comparison-table validation-table"><tbody>${summaryRows.join("")}</tbody></table>`;
    return;
  }
  const validation = result.airfoil_validation;
  if (!validation) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $("solverSummaryTitle").textContent = "XFOIL kabul ve seyir noktası";
  const initial = validation.initial_check || {}; const cruise = validation.final_xfoil_point || {};
  const cst = validation.cst_optimization;
  $("xfoilStatusTag").textContent = validation.escalated_to_cst ? "NACA → CST / XFOIL" : "NACA KABUL / XFOIL";
  const row = (label, value, unit="") => `<tr><td>${escapeHtml(label)}</td><td class="best">${escapeHtml(value)} ${escapeHtml(unit)}</td></tr>`;
  const initialStatus = initial.accepted ? "Tolerans içinde" : "Tolerans dışı → yeniden arandı";
  const workerText = cst ? `${cst.parallel_workers_used} / ${cst.candidates_evaluated} aday` : "CST gerekmedi";
  $("xfoilSummary").innerHTML = `<table class="comparison-table validation-table"><tbody>${row("İlk NACA kararı", initialStatus)}${row("CL farkı · aynı α", fmt(initial.cl_error_percent,2), "%")}${row("CD farkı · aynı CL", fmt(initial.cd_error_percent,2), "%")}${row("Son profil ailesi", result.airfoil.family)}${row("XFOIL seyir α", fmt(cruise.alpha_deg,2), "°")}${row("XFOIL seyir CL", fmt(cruise.cl,4))}${row("XFOIL seyir CD", fmt(cruise.cd,5))}${row("XFOIL 2B L/D", fmt(cruise.ld,1))}${row("Paralel CST işi", workerText)}</tbody></table>`;
}

function renderEngineering(result) {
  const structure = result.structural_analysis || { enabled:false };
  const hydro = result.hydro_analysis || { enabled:false };
  const panel = $("engineeringPanel");
  const row = (label, value, unit="") => `<tr><td>${escapeHtml(label)}</td><td class="best">${escapeHtml(value)} ${escapeHtml(unit)}</td></tr>`;
  const rows = [];
  rows.push(row("Yapısal denetim", structure.enabled ? (structure.performed ? (structure.passed ? "geçti" : "sınır dışı") : "veri yok") : "kapalı"));
  if (structure.enabled && structure.performed) {
    rows.push(row("Gerilme kullanımı", fmt(structure.stress_utilization,3)));
    rows.push(row("Sehim kullanımı", fmt(structure.deflection_utilization,3)));
    rows.push(row("Burulma kullanımı", fmt(structure.twist_utilization,3)));
    rows.push(row("Maks. uç sehmi", fmt(structure.max_tip_deflection_m,4), "m"));
    rows.push(row("Elastik uç twist", fmt(structure.max_elastic_twist_tip_deg,3), "°"));
    rows.push(row("Tahmini kanat malzeme kütlesi", fmt(structure.estimated_wing_material_mass_kg,3), "kg"));
  }
  const hydroMode = hydro.constraint_mode === "report_only" ? "yalnız rapor" : "sert kısıt";
  rows.push(row("Hidrofoil taraması", hydro.enabled ? (hydro.performed ? `${hydro.passed ? "kavitasyon marjı var" : "kavitasyon riski"} · ${hydroMode}` : `Cp_min yok · ${hydroMode}`) : "uygulanmadı / kapalı"));
  if (hydro.enabled && hydro.performed) {
    rows.push(row("Kavitasyon kullanımı", fmt(hydro.cavitation_utilization,3)));
    rows.push(row("Minimum marj oranı", fmt(hydro.minimum_cavitation_margin_ratio,3)));
    if (hydro.panel_map_available) rows.push(row("Riskli finalist alanı", fmt(hydro.risk_area_percent,2), "%"));
    rows.push(row("Serbest yüzey risk bayrağı", hydro.free_surface_risk ? "evet" : "hayır"));
  }
  $("engineeringSummary").innerHTML = `<table class="comparison-table validation-table"><tbody>${rows.join("")}</tbody></table>`;
  $("engineeringStatusTag").textContent = structure.enabled || hydro.enabled ? "ÖN TARAMA" : "KAPALI";
  panel.classList.remove("hidden");
}

function cavitationColor(utilization) {
  const value = Number(utilization);
  if (!Number.isFinite(value)) return "#aab5b0";
  if (value >= 1) return value >= 1.25 ? "#9f3428" : "#d84d37";
  if (value >= .8) return "#d8993b";
  return value >= .55 ? "#55a99f" : "#0d786e";
}

function renderCavitationMap(container, map) {
  const panels = (map?.panels || []).filter((panel) => Array.isArray(panel.vertices) && panel.vertices.length >= 3);
  if (!panels.length) {
    container.innerHTML = `<div class="chart-empty">Finalist panel köşe koordinatları runner tarafından döndürülmedi.</div>`;
    return;
  }
  const width = 560, height = 285, pad = 18;
  const projectRaw = (vertex) => {
    const x = Number(vertex[0]), y = Number(vertex[1]), z = Number(vertex[2]);
    return [y + .34*x, -.92*z + .22*x];
  };
  const projectedPanels = panels.map((panel) => ({ panel, points: panel.vertices.map(projectRaw) }));
  const all = projectedPanels.flatMap((item) => item.points);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  all.forEach(([u,v]) => {
    xmin = Math.min(xmin,u); xmax = Math.max(xmax,u);
    ymin = Math.min(ymin,v); ymax = Math.max(ymax,v);
  });
  if (Math.abs(xmax-xmin) < 1e-12) { xmin -= 1; xmax += 1; }
  if (Math.abs(ymax-ymin) < 1e-12) { ymin -= 1; ymax += 1; }
  const scale = Math.min((width-2*pad)/(xmax-xmin), (height-2*pad)/(ymax-ymin));
  const ox = (width - scale*(xmax-xmin))/2, oy = (height - scale*(ymax-ymin))/2;
  const screen = ([u,v]) => [ox + (u-xmin)*scale, height - oy - (v-ymin)*scale];
  projectedPanels.sort((a,b) => Number(a.panel.z_m)-Number(b.panel.z_m));
  const shapes = projectedPanels.map(({panel,points}) => {
    const vertices = points.map((point) => screen(point).map((value) => value.toFixed(1)).join(",")).join(" ");
    const title = `${panel.component || "kanat"} · ${panel.surface || "yüzey"} · U=${fmt(panel.cavitation_utilization,3)} · Cp=${fmt(panel.cp,3)}`;
    return `<polygon class="cavitation-cell ${escapeHtml(panel.risk_state || "safe")}" fill="${cavitationColor(panel.cavitation_utilization)}" points="${vertices}"><title>${escapeHtml(title)}</title></polygon>`;
  }).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Kanadın panel bazlı kavitasyon risk haritası">${shapes}</svg><div class="cavitation-legend"><span><i class="safe"></i>Güvenli · U&lt;0,8</span><span><i class="near"></i>Sınıra yakın</span><span><i class="risk"></i>Risk · U≥1</span></div>`;
}

function renderCavitation(result) {
  const hydro = result.hydro_analysis || {};
  const panel = $("cavitationPanel");
  const map = hydro.panel_map;
  if (!hydro.enabled || !hydro.panel_map_available || !map?.available) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const mode = hydro.constraint_mode === "report_only" ? "YALNIZ RAPOR" : "SERT KISIT";
  $("cavitationStatusTag").textContent = `${hydro.passed ? "MARJ VAR" : "RİSK"} · ${mode}`;
  const item = (label,value,unit="") => `<div><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong><em>${escapeHtml(unit)}</em></div>`;
  $("cavitationMetrics").innerHTML = [
    item("Riskli alan",fmt(hydro.risk_area_percent,2),"%"),
    item("Sınıra yakın alan",fmt(hydro.near_risk_area_percent,2),"%"),
    item("Fiziksel başlangıç alanı",fmt(hydro.physical_onset_area_percent,2),"%"),
    item("Alan-ağırlıklı şiddet",fmt(hydro.cavitation_severity_index,4)),
    item("Etkilenen izdüşümsel span",fmt(hydro.affected_projected_span_percent,1),"%"),
    item("En kötü kullanım",fmt(map.maximum_utilization,3)),
    item("Basınç açığı integrali",fmt(hydro.pressure_deficit_area_integral_n,2),"N"),
    item("Kavitasyon sayısı",fmt(map.cavitation_number_sigma,3),"σ"),
  ].join("");
  renderCavitationMap($("cavitationMap"), map);
  svgLineChart($("cavitationSpanChart"), map.spanwise_distribution || [], "span_fraction", "maximum_utilization", { referenceY:1, xDigits:2, yDigits:2, label:"Yarı açıklık boyunca kavitasyon kullanımı", emptyText:"Açıklık dağılımı üretilemedi." });
  svgLineChart($("cavitationSpeedChart"), hydro.speed_sensitivity || [], "speed_m_s", "risk_area_percent", { xDigits:1, yDigits:1, label:"Hıza göre riskli alan yüzdesi", emptyText:"Hız duyarlılığı üretilemedi." });
  svgLineChart($("cavitationDepthChart"), hydro.depth_sensitivity || [], "submergence_depth_m", "risk_area_percent", { xDigits:2, yDigits:1, label:"Batma derinliğine göre riskli alan yüzdesi", emptyText:"Derinlik duyarlılığı üretilemedi." });
  const resolution = map.upper_lower_resolved ? "üst ve alt yüzey ayrı çözüldü" : "flow5 ince-yüzey Cp panelleri kullanıldı; üst/alt yüzey ayrı değildir";
  $("cavitationModelNote").textContent = `${map.panel_count} finalist paneli · ${fmt(map.speed_m_s,2)} m/s · ${fmt(map.submergence_depth_m,2)} m derinlik · ${resolution}. Hız/derinlik eğrileri donmuş-Cp ön taramasıdır; gerçek kavite boyu, hacmi veya L/D kaybı çok-fazlı CFD olmadan hesaplanmaz.`;
}

function compactValue(value) {
  if (value == null) return "—";
  if (typeof value === "number") return fmt(value, Math.abs(value) < 0.1 ? 4 : 2);
  if (typeof value === "object") {
    return Object.entries(value).map(([key, item]) => `${key}: ${compactValue(item)}`).join(" · ");
  }
  return String(value);
}

function renderValidation(result) {
  const report = result.validation_report;
  const panel = $("validationPanel");
  if (!report) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $("validationStatusTag").textContent = !report.enabled ? "KAPALI" : report.passed ? `${report.checks_passed}/${report.checks_total} GEÇTİ` : "İNCELE";
  if (!report.enabled) {
    $("validationSummary").innerHTML = `<div class="empty-report">Doğrulama kullanıcı ayarıyla kapatıldı.</div>`;
    return;
  }
  const checks = (report.checks || []).map((check) => {
    const state = check.passed ? "pass" : check.blocking ? "fail" : "warn";
    const detail = check.detail ? `<small>${escapeHtml(check.detail)}</small>` : "";
    return `<div class="check-row ${state}"><i>${check.passed ? "✓" : "!"}</i><div><strong>${escapeHtml(check.title)}</strong><span>${escapeHtml(compactValue(check.measured))} · sınır ${escapeHtml(check.limit)}</span>${detail}</div></div>`;
  }).join("");
  const signature = String(report.regression_signature_sha256 || "");
  $("validationSummary").innerHTML = `${checks}<div class="report-foot"><span>${escapeHtml(report.fidelity || "")}</span><code title="${escapeHtml(signature)}">imza ${escapeHtml(signature.slice(0,16))}</code></div>`;
}

function renderDiagnostics(result) {
  const report = result.diagnostic_report;
  const panel = $("diagnosticPanel");
  if (!report) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $("diagnosticStatusTag").textContent = report.status === "clear" ? "TEMİZ" : report.status === "critical" ? "KRİTİK" : "UYARI";
  if (!(report.diagnoses || []).length) {
    $("diagnosticSummary").innerHTML = `<div class="diagnostic-item clear"><i>✓</i><div><strong>Belirgin bir sınırlayıcı neden bulunmadı</strong><p>Solver, mesh, geometri sınırları ve optimizer telemetrisi kuralları temiz geçti.</p></div></div>`;
    return;
  }
  $("diagnosticSummary").innerHTML = report.diagnoses.map((item) => `<div class="diagnostic-item ${escapeHtml(item.severity)}"><i>${item.severity === "critical" ? "!" : "•"}</i><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.evidence)}</p><small>${escapeHtml(item.recommendation)}</small></div></div>`).join("");
}

function scatterChart(container, points, xKey, yKey, xSpec, ySpec) {
  const usable = (points || []).filter((point) => Number.isFinite(Number(point[xKey])) && Number.isFinite(Number(point[yKey])));
  if (!usable.length) { container.innerHTML = `<div class="chart-empty">Bu eksen çifti için sayısal Pareto adayı yok.</div>`; return; }
  const width = 760, height = 300, pad = { l: 62, r: 25, t: 22, b: 48 };
  const xs = usable.map((point) => Number(point[xKey]));
  const ys = usable.map((point) => Number(point[yKey]));
  let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xmargin = Math.max((xmax-xmin)*.09, Math.abs(xmax)*.015, 1e-9);
  const ymargin = Math.max((ymax-ymin)*.09, Math.abs(ymax)*.015, 1e-9);
  xmin -= xmargin; xmax += xmargin; ymin -= ymargin; ymax += ymargin;
  const sx = (value) => pad.l + (value-xmin)/(xmax-xmin)*(width-pad.l-pad.r);
  const sy = (value) => height-pad.b-(value-ymin)/(ymax-ymin)*(height-pad.t-pad.b);
  let grid = "";
  for (let index=0; index<=5; index++) {
    const x = pad.l + index*(width-pad.l-pad.r)/5;
    const y = pad.t + index*(height-pad.t-pad.b)/5;
    grid += `<line class="grid-line" x1="${x}" x2="${x}" y1="${pad.t}" y2="${height-pad.b}"/><text class="axis-label" x="${x}" y="${height-25}" text-anchor="middle">${fmt(xmin+index*(xmax-xmin)/5,2)}</text>`;
    grid += `<line class="grid-line" x1="${pad.l}" x2="${width-pad.r}" y1="${y}" y2="${y}"/><text class="axis-label" x="${pad.l-8}" y="${y+3}" text-anchor="end">${fmt(ymax-index*(ymax-ymin)/5,2)}</text>`;
  }
  const frontier = usable.filter((point) => point.on_pareto_front).sort((a,b) => Number(a[xKey])-Number(b[xKey]));
  const frontPath = frontier.length > 1 ? `<path class="pareto-front-line" d="${frontier.map((point,index)=>`${index?"L":"M"}${sx(Number(point[xKey])).toFixed(1)},${sy(Number(point[yKey])).toFixed(1)}`).join(" ")}"/>` : "";
  const dots = usable.map((point) => {
    const classes = ["pareto-dot", point.on_pareto_front ? "front" : "", point.selected ? "selected" : "", point.feasible ? "" : "infeasible"].filter(Boolean).join(" ");
    const radius = point.selected ? 7 : point.on_pareto_front ? 5 : 3.5;
    return `<circle class="${classes}" cx="${sx(Number(point[xKey])).toFixed(1)}" cy="${sy(Number(point[yKey])).toFixed(1)}" r="${radius}"><title>${escapeHtml(point.id)} · ${escapeHtml(xSpec.label)} ${fmt(point[xKey],3)} ${escapeHtml(xSpec.unit)} · ${escapeHtml(ySpec.label)} ${fmt(point[yKey],3)} ${escapeHtml(ySpec.unit)}</title></circle>`;
  }).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Pareto aday dağılımı">${grid}${frontPath}${dots}<text class="axis-title" x="${(pad.l+width-pad.r)/2}" y="${height-6}" text-anchor="middle">${escapeHtml(xSpec.label)} · ${escapeHtml(xSpec.unit)}</text><text class="axis-title" transform="translate(13 ${(pad.t+height-pad.b)/2}) rotate(-90)" text-anchor="middle">${escapeHtml(ySpec.label)} · ${escapeHtml(ySpec.unit)}</text></svg>`;
}

function renderPareto(result) {
  const report = result.pareto_analysis;
  const panel = $("paretoPanel");
  if (!report?.enabled || !(report.objective_specs || []).length) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const specs = report.objective_specs;
  const previousX = $("paretoX").value;
  const previousY = $("paretoY").value;
  const options = specs.map((spec) => `<option value="${escapeHtml(spec.key)}">${escapeHtml(spec.label)}</option>`).join("");
  $("paretoX").innerHTML = options; $("paretoY").innerHTML = options;
  $("paretoX").value = specs.some((spec)=>spec.key===previousX) ? previousX : specs[0].key;
  $("paretoY").value = specs.some((spec)=>spec.key===previousY) ? previousY : (specs[1] || specs[0]).key;
  const xSpec = specs.find((spec)=>spec.key===$("paretoX").value) || specs[0];
  const ySpec = specs.find((spec)=>spec.key===$("paretoY").value) || specs[1] || specs[0];
  scatterChart($("paretoChart"), report.candidates, xSpec.key, ySpec.key, xSpec, ySpec);
  $("paretoStatusTag").textContent = report.optimizer_generated_frontier
    ? `NSGA-II · ${report.frontier_count}/${report.candidate_count}`
    : `${report.frontier_count}/${report.candidate_count} SONRADAN`;
  $("paretoNote").textContent = "Turuncu: seçilen · yeşil: non-dominated · sol-alt daha iyi";
  $("paretoSummary").innerHTML = `<span><b>${report.candidate_count}</b> gerçek çözücü adayı</span><span><b>${report.frontier_count}</b> Pareto çözümü</span><span><b>${report.selected_on_front ? "Evet" : "Hayır"}</b> seçilen cephede</span><small>${escapeHtml(report.optimizer_generated_frontier ? "Cephe NSGA-II seçim baskısıyla doğrudan üretildi. " : "Cephe tek amaçlı arama sonrasında çıkarıldı. ")}${escapeHtml(report.fidelity_note || "")}</small>`;
}

function renderStability(result) {
  const report = result.multi_seed_stability;
  const panel = $("stabilityPanel");
  if (!report) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $("stabilityStatusTag").textContent = !report.enabled ? "1 KOŞU" : report.stable ? "KARARLI" : "İNCELE";
  const geometryMax = Math.max(0, ...Object.values(report.geometry_cv_percent || {}).map(Number));
  const rows = (report.runs || []).map((run) => run.completed
    ? `<tr><td>${run.selected ? "★ " : ""}${escapeHtml(run.seed)}</td><td>${run.feasible ? "evet" : "hayır"}</td><td>${fmt(run.objective,5)}</td><td>${fmt(run.ld,1)}</td><td>${fmt(run.drag_n,2)} N</td></tr>`
    : `<tr class="failed-row"><td>${escapeHtml(run.seed)}</td><td colspan="4">Başarısız · ${escapeHtml(run.diagnosis?.title || run.error || "bilinmeyen hata")}${run.diagnosis?.recommendation ? `<br><small>${escapeHtml(run.diagnosis.recommendation)}</small>` : ""}</td></tr>`).join("");
  $("stabilitySummary").innerHTML = `<div class="stability-metrics"><span><small>Tamamlanan</small><b>${report.runs_completed}/${report.runs_requested}</b></span><span><small>Amaç CV</small><b>%${fmt(report.objective_cv_percent,2)}</b></span><span><small>Maks. geometri CV</small><b>%${fmt(geometryMax,2)}</b></span><span><small>Seçilen seed</small><b>${escapeHtml(report.selected_seed)}</b></span></div><div class="table-scroll"><table class="comparison-table"><thead><tr><th>Seed</th><th>Fizibil</th><th>Amaç</th><th>L/D</th><th>Drag</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderBudgetConvergence(result) {
  const foil = result.airfoil_optimization?.budget_convergence;
  const wing = result.wing_optimization?.budget_convergence;
  const winglet = result.winglet_comparison?.winglet?.budget_convergence;
  const panel = $("budgetPanel");
  if (!foil && !wing) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const reports = [foil, wing, winglet].filter(Boolean);
  const exhausted = reports.some((report) => report.status === "budget_exhausted");
  const allConverged = reports.length > 0 && reports.every((report) => report.converged === true);
  $("budgetStatusTag").textContent = exhausted ? "BÜTÇE-SINIRLI" : allConverged ? "YETERLİ" : "SABİT BÜTÇE";
  const row = (label, report) => {
    const last = (report.checkpoints || []).at(-1) || {};
    const route = (report.milestones || [report.base_budget]).join(" → ");
    const decision = report.converged === true ? "yakınsadı" : report.converged === false ? "azami bütçede hareketli" : "otomatik karar kapalı";
    return `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(route)}</td><td class="best">${report.evaluations_completed}/${report.maximum_budget}</td><td>%${fmt(last.controlling_change_percent,3)}</td><td>${escapeHtml(decision)}</td></tr>`;
  };
  $("budgetSummary").innerHTML = `<div class="table-scroll"><table class="comparison-table"><thead><tr><th>Aşama</th><th>Kontrol bütçeleri</th><th>Gerçek / azami</th><th>Son değişim</th><th>Karar</th></tr></thead><tbody>${foil ? row("Airfoil",foil) : ""}${wing ? row("Kanat",wing) : ""}${winglet ? row("Winglet",winglet) : ""}</tbody></table></div><p class="report-note">${escapeHtml(exhausted ? "Azami bütçede Pareto/amaç hareketi toleransı aşmış; bütçeyi veya multi-seed sayısını artırın." : allConverged ? "Tüm etkin aramalar ayarlanan tolerans içinde kararlı; ayrılmamış azami bütçe kullanılmadı." : "Otomatik bütçe denetimi kapalı olduğu için yalnız girilen sabit bütçe tamamlandı.")}</p>`;
}

const HISTORY_KEY = "aeropt.projectHistory.v1";
function readHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch { return []; }
}
function writeHistory(items) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0,12))); } catch { /* private mode */ }
}
function historySummary(result) {
  const geometry = result.wing.geometry;
  const signature = result.validation_report?.regression_signature_sha256 || "";
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    created_at: new Date().toISOString(), signature,
    name: result.airfoil.name, fluid: result.flow.name,
    speed_m_s: result.flow.speed_m_s, target_lift_n: result.flow.target_lift_n,
    status: result.status, selected_seed: result.multi_seed_stability?.selected_seed ?? result.solver_run?.selected_seed ?? "—",
    validation: result.validation_report?.status || "—",
    wing: { lift_n:result.wing.lift_n, drag_n:result.wing.drag_n, ld:result.wing.ld, root_bending_moment_nm:result.wing.root_bending_moment_nm },
    geometry: { span:geometry.span, root_chord:geometry.root_chord, taper:geometry.taper, sweep_deg:geometry.sweep_deg, tip_twist_deg:geometry.tip_twist_deg, area:geometry.area, aspect_ratio:geometry.aspect_ratio, winglet_enabled:geometry.winglet_enabled, winglet_height:geometry.winglet_height, winglet_cant_deg:geometry.winglet_cant_deg, winglet_toe_deg:geometry.winglet_toe_deg, winglet_taper:geometry.winglet_taper },
    airfoil: { max_camber:result.airfoil.max_camber, thickness:result.airfoil.thickness },
  };
}
function rememberResult(result) {
  const items = readHistory();
  items.unshift(historySummary(result));
  writeHistory(items);
}
function historyLabel(item) {
  const date = new Date(item.created_at);
  const stamp = Number.isNaN(date.getTime()) ? "" : date.toLocaleString("tr-TR", {day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
  return `${stamp} · ${item.name} · L/D ${fmt(item.wing?.ld,1)}`;
}
function renderHistory() {
  const items = readHistory();
  const priorA = $("historyA").value, priorB = $("historyB").value;
  const options = items.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(historyLabel(item))}</option>`).join("");
  $("historyA").innerHTML = options; $("historyB").innerHTML = options;
  if (!items.length) { $("historyComparison").innerHTML = `<div class="empty-report">Henüz kaydedilmiş tasarım yok.</div>`; return; }
  $("historyA").value = items.some((item)=>item.id===priorA) ? priorA : items[0].id;
  $("historyB").value = items.some((item)=>item.id===priorB) ? priorB : (items[1] || items[0]).id;
  const a = items.find((item)=>item.id===$("historyA").value) || items[0];
  const b = items.find((item)=>item.id===$("historyB").value) || items[0];
  const row = (label, av, bv, digits=2, unit="") => `<tr><td>${escapeHtml(label)}</td><td class="best">${fmt(av,digits)} ${escapeHtml(unit)}</td><td>${fmt(bv,digits)} ${escapeHtml(unit)}</td><td>${fmt(Number(av)-Number(bv),digits)} ${escapeHtml(unit)}</td></tr>`;
  $("historyComparison").innerHTML = `<div class="table-scroll"><table class="comparison-table"><thead><tr><th>Gösterge</th><th>Tasarım A</th><th>Tasarım B</th><th>A − B</th></tr></thead><tbody>${row("L/D",a.wing.ld,b.wing.ld,1)}${row("Sürükleme",a.wing.drag_n,b.wing.drag_n,2,"N")}${row("Taşıma",a.wing.lift_n,b.wing.lift_n,1,"N")}${row("Açıklık",a.geometry.span,b.geometry.span,3,"m")}${row("Alan",a.geometry.area,b.geometry.area,3,"m²")}${row("Açıklık oranı",a.geometry.aspect_ratio,b.geometry.aspect_ratio,2)}${row("Taper",a.geometry.taper,b.geometry.taper,3)}${row("Sweep",a.geometry.sweep_deg,b.geometry.sweep_deg,2,"°")}${row("Uç twist",a.geometry.tip_twist_deg,b.geometry.tip_twist_deg,2,"°")}${row("Winglet yüksekliği",a.geometry.winglet_height || 0,b.geometry.winglet_height || 0,3,"m")}${row("Winglet cant",a.geometry.winglet_cant_deg || 90,b.geometry.winglet_cant_deg || 90,1,"°")}${row("Kök momenti (telemetri)",a.wing.root_bending_moment_nm,b.wing.root_bending_moment_nm,1,"N·m")}</tbody></table></div>`;
}

function rememberAirfoilResult(result) {
  const dat = result.exports?.airfoil_dat;
  if (!dat) return;
  try {
    localStorage.setItem(SAVED_AIRFOIL_KEY, JSON.stringify({
      name: result.airfoil.name,
      family: result.airfoil.family,
      dat,
      created_at: new Date().toISOString(),
    }));
  } catch { /* private mode */ }
  updateSavedAirfoilStatus();
}

function setWingResultVisibility(visible) {
  ["planformPanel", "loadPanel", "engineeringPanel", "cavitationPanel", "validationPanel", "diagnosticPanel", "paretoPanel", "stabilityPanel", "comparisonPanel", "historyPanel"]
    .forEach((id) => $(id).classList.toggle("hidden", !visible));
  $("visualGrid").classList.toggle("single-column", !visible);
  $("chartGrid").classList.toggle("single-column", !visible);
}

function configureExportPanel(foilOnly) {
  const panel = document.querySelector(".export-panel");
  panel.querySelector("h3").textContent = foilOnly ? "Optimize profil dosyaları" : "Çözümlenmiş flow5 paketi";
  panel.querySelector("p").textContent = foilOnly
    ? "DAT, flow5/XFoil poları ve yeniden kullanılabilir proje girdisi dışa aktarılır."
    : "ZIP; DAT, plane/analysis XML, OBJ, polarlar, sonuçlar ve gerçek çözümlenmiş .fl5 projesini birlikte taşır.";
  panel.querySelector("ol").innerHTML = foilOnly
    ? "<li><b>Airfoil · DAT</b> dosyasını indirin veya doğrudan Yalnız kanat modunu seçin</li><li>Son optimize profil bu tarayıcıda otomatik saklanır</li><li>Kanat aşamasında profil değişmeden flow5/XFoil ile yeniden doğrulanır</li>"
    : "<li><b>aeropt-optimized.fl5</b> dosyasını flow5 7.57'de açın</li><li>Foil ve 3B polarları Data ağacında inceleyin</li><li>DAT/XML/OBJ dosyalarını bağımsız geometri aktarımı için kullanın</li>";
}

function renderFoilOnlyResult(result) {
  lastResult = result;
  setWingResultVisibility(false);
  configureExportPanel(true);
  const foil = result.airfoil;
  const meta = result.airfoil_optimization;
  const cruise = meta.final_cruise_point || {};
  const baseline = meta.baseline || {};
  const selection = meta.selection || {};
  $("resultTitle").textContent = foil.name;
  $("resultSubtitle").textContent = `${result.flow.name} · ${fmt(result.flow.speed_min_m_s,1)}–${fmt(result.flow.speed_max_m_s,1)} m/s · yalnız profil`;
  $("feasibilityBadge").textContent = result.status === "feasible" ? "Profil hazır" : "Kontrol gerekli";
  $("feasibilityBadge").classList.toggle("review", result.status !== "feasible");
  $("metricGrid").innerHTML = [
    metric("Hedef CL",fmt(meta.target_cl,4),""), metric("Referans CL",fmt(cruise.cl,4),""),
    metric("Referans CD",fmt(cruise.cd,5),""), metric("2B L / D",fmt(cruise.ld,1),""),
    metric("Reynolds",sci(meta.reynolds),""),
  ].join("");
  $("foilTag").textContent = `${foil.family} · 100 nokta`;
  $("reTag").textContent = `${result.polar_source} · M ${fmt(result.flow.mach,3)}`;
  renderFoil(result);
  svgLineChart($("polarChart"), result.polar, "alpha_deg", "cl", { referenceX: cruise.alpha_deg, label: "Taşıma katsayısı polar grafiği" });
  const row = (label, value, unit="") => `<tr><td>${escapeHtml(label)}</td><td class="best">${escapeHtml(value)} ${escapeHtml(unit)}</td></tr>`;
  $("xfoilPanel").classList.remove("hidden");
  $("solverSummaryTitle").textContent = "flow5-native profil çözüm zinciri";
  $("xfoilStatusTag").textContent = "flow5 XFoilTask · KANAT ATLANDI";
  $("xfoilSummary").innerHTML = `<table class="comparison-table validation-table"><tbody>${[
    row("Başlangıç profili", baseline.display_name || "Eppler E818"),
    row("Profil seçimi", selection.selected_baseline ? "Baseline korundu" : `${selection.selected_family || foil.family} optimize profil`),
    row("Baseline iyileşmesi", selection.selected_improvement_vs_baseline_percent == null ? "—" : fmt(selection.selected_improvement_vs_baseline_percent,2), "%"),
    row("Foil optimizeri", result.solver_run.foil_optimizer),
    row("Gerçek foil adayı", String(meta.candidates_evaluated)),
    row("Akış noktası", String(result.flow.speed_samples)),
    row("Referans α", fmt(cruise.alpha_deg,2), "°"),
    row("Referans CL", fmt(cruise.cl,4)),
    row("Referans CD", fmt(cruise.cd,5)),
  ].join("")}</tbody></table>`;
  renderBudgetConvergence(result);
  $("insights").innerHTML = (result.insights || []).map((item) => `<div class="insight ${escapeHtml(item.level)}"><i></i><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.text)}</p></div></div>`).join("");
  const ex = result.exports;
  const scalarEx = ex.scalar_only_alternative || {};
  const scalarDownloads = !scalarEx.same_as_feasibility_first ? [
    scalarEx.wing_obj ? downloadLink(scalarEx.wing_obj_filename, scalarEx.wing_obj, "model/obj", "Skaler alternatif · OBJ") : "",
    scalarEx.wing_step_base64 ? base64DownloadLink(scalarEx.wing_step_filename, scalarEx.wing_step_base64, "model/step", "Skaler alternatif · STEP") : "",
    scalarEx.results_csv ? downloadLink(scalarEx.results_filename, scalarEx.results_csv, "text/csv", "Skaler alternatif · CSV") : "",
    scalarEx.flow5_project_base64 ? base64DownloadLink(scalarEx.flow5_project_filename, scalarEx.flow5_project_base64, "application/octet-stream", "Skaler alternatif · FL5") : "",
  ] : [];
  $("downloads").innerHTML = [
    ex.airfoil_dat ? downloadLink(ex.airfoil_filename, ex.airfoil_dat, "text/plain", "Airfoil · DAT") : "",
    ex.xfoil_polar_csv ? downloadLink(ex.xfoil_polar_filename, ex.xfoil_polar_csv, "text/csv", "flow5/XFoil polar · CSV") : "",
    ex.project_json ? downloadLink(ex.project_filename, ex.project_json, "application/json", "Profil projesi · JSON") : "",
  ].filter(Boolean).join("");
  $("methodText").textContent = `${result.model.airfoil}. Kanat geometrisi, yapı ve hidrofoil kontrolleri bu çalışmada yürütülmedi.`;
  rememberAirfoilResult(result);
  showState("resultState");
}

function renderResult(result) {
  if (result.workflow_mode === "foil_only") { renderFoilOnlyResult(result); return; }
  setWingResultVisibility(true);
  configureExportPanel(false);
  lastResult = result; const wing = result.wing, g = wing.geometry, f = result.airfoil;
  $("resultTitle").textContent = f.name;
  const speedText = result.flow5_native ? `${fmt(result.flow.speed_min_m_s,1)}–${fmt(result.flow.speed_max_m_s,1)} m/s · ref ${fmt(result.flow.speed_m_s,1)}` : `${fmt(result.flow.speed_m_s,1)} m/s`;
  $("resultSubtitle").textContent = `${result.flow.name} · ${speedText} · ${result.polar_source} · hedef ${fmt(result.flow.target_lift_n,1)} N`;
  $("feasibilityBadge").textContent = result.status === "feasible" ? "Fizibil" : "Kontrol gerekli";
  $("feasibilityBadge").classList.toggle("review", result.status !== "feasible");
  $("metricGrid").innerHTML = [
    metric("Gerçekleşen taşıma", fmt(wing.lift_n,1), "N"), metric("Toplam sürükleme",fmt(wing.drag_n,2),"N"),
    metric("L / D",fmt(wing.ld,1),""), metric("İndüklenmiş pay",fmt(wing.induced_drag_fraction_percent ?? 100*wing.cd_induced/Math.max(wing.cd_total,1e-12),1),"%"), metric("Açıklık oranı",fmt(g.aspect_ratio,2),""), metric("Tasarım α",fmt(g.alpha_deg,2),"°")
  ].join("");
  $("foilTag").textContent = `${f.family} · Re ${sci(result.airfoil_optimization.reynolds)}`;
  $("reTag").textContent = `${result.polar_source} · M ${fmt(result.flow.mach,3)}`;
  $("savingTag").textContent = result.winglet_comparison?.performed
    ? `${result.winglet_comparison.selection === "winglet" ? "WINGLET" : "PLANAR"} · ΔD %${fmt(result.winglet_comparison.delta_winglet_vs_planar?.drag_percent,1)}`
    : `%${fmt(result.wing_optimization.drag_reduction_vs_rectangular_percent,1)} sürükleme farkı`;
  renderFoil(result); renderPlanform(result);
  svgLineChart($("polarChart"), result.polar, "alpha_deg", "cl", { referenceX: g.alpha_deg, label: "Taşıma katsayısı polar grafiği" });
  svgLineChart($("loadChart"), wing.distribution, "y_m", "lift_n_per_m", { yDigits: 1, xDigits: 2, label: "Kanat açıklığı boyunca yük dağılımı", emptyText: "Span dağılımı JSON köprüsünde yok; gerçek dağılım çözümlenmiş .fl5 projesindedir." });
  renderComparison(result);
  renderXfoil(result);
  renderEngineering(result);
  renderCavitation(result);
  renderValidation(result);
  renderDiagnostics(result);
  renderPareto(result);
  renderStability(result);
  renderBudgetConvergence(result);
  rememberResult(result);
  renderHistory();
  $("insights").innerHTML = result.insights.map((item) => `<div class="insight ${escapeHtml(item.level)}"><i></i><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.text)}</p></div></div>`).join("");
  const ex = result.exports;
  $("downloads").innerHTML = [
    downloadLink(ex.airfoil_filename, ex.airfoil_dat, "text/plain", "Airfoil · DAT"),
    downloadLink(ex.plane_filename, ex.plane_xml, "application/xml", "Plane · XML"),
    ex.analysis_xml ? downloadLink(ex.analysis_filename, ex.analysis_xml, "application/xml", "Analiz · XML") : "",
    downloadLink(ex.wing_obj_filename, ex.wing_obj, "model/obj", "3B kanat · OBJ"),
    ex.wing_step_base64 ? base64DownloadLink(ex.wing_step_filename, ex.wing_step_base64, "model/step", "3B CAD · STEP") : "",
    downloadLink(ex.results_filename, ex.results_csv, "text/csv", "Sonuçlar · CSV"),
    downloadLink(ex.project_filename, ex.project_json, "application/json", "Proje · JSON"),
    ex.xfoil_polar_csv ? downloadLink(ex.xfoil_polar_filename, ex.xfoil_polar_csv, "text/csv", result.flow5_native ? "flow5/XFoil polar · CSV" : "XFOIL polar · CSV") : "",
    ex.validation_json ? downloadLink(ex.validation_filename, ex.validation_json, "application/json", "Doğrulama · JSON") : "",
    ex.pareto_json ? downloadLink(ex.pareto_filename, ex.pareto_json, "application/json", "Pareto · JSON") : "",
    ex.multi_seed_json ? downloadLink(ex.multi_seed_filename, ex.multi_seed_json, "application/json", "Multi-seed · JSON") : "",
    ex.diagnostics_json ? downloadLink(ex.diagnostics_filename, ex.diagnostics_json, "application/json", "Teşhis · JSON") : "",
    ex.cavitation_json ? downloadLink(ex.cavitation_filename, ex.cavitation_json, "application/json", "Kavitasyon haritası · JSON") : "",
    ex.flow5_project_base64 ? base64DownloadLink(ex.flow5_project_filename, ex.flow5_project_base64, "application/octet-stream", "Çözümlenmiş flow5 · FL5") : "",
    ...scalarDownloads,
    ...(ex.section_airfoils || []).map((item) => downloadLink(item.filename, item.airfoil_dat, "text/plain", `${item.station} airfoil · DAT`)),
    base64DownloadLink(ex.flow5_bundle_filename, ex.flow5_bundle_base64, "application/zip", "Tüm flow5 paketi · ZIP"),
  ].filter(Boolean).join("");
  const structureText = result.structural_analysis?.enabled ? "Yapısal sonuçlar ön boyutlandırma taramasıdır; FEA değildir. " : "Yapısal denetim kapalıdır. ";
  const hydroText = result.hydro_analysis?.enabled ? "Kavitasyon/serbest-yüzey sonuçları ön taramadır; çok-fazlı CFD değildir." : "Hidrofoil fiziği uygulanmadı.";
  $("methodText").textContent = `${result.model.airfoil}; ${result.model.wing}. ${result.flow5_native ? "AeroOpt aerodinamik korelasyonu amaç fonksiyonuna girmez. " : ""}${structureText}${hydroText}`;
  rememberAirfoilResult(result);
  showState("resultState");
}

async function optimize(event) {
  event.preventDefault();
  try {
    const request = await collectRequest(); startLoading();
    const response = await fetch("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    let job; try { job = await response.json(); } catch { job = { error: `Sunucu geçersiz yanıt verdi (${response.status}).` }; }
    if (!response.ok) throw new Error(job.detail || job.error || `HTTP ${response.status}`);
    activeJobId = job.id;
    while (activeJobId) {
      await pause(500);
      const poll = await fetch(`/api/jobs/${activeJobId}`, { cache:"no-store" });
      let state; try { state = await poll.json(); } catch { throw new Error("İlerleme yanıtı okunamadı."); }
      if (!poll.ok) throw new Error(state.error || `HTTP ${poll.status}`);
      updateProgress(state.progress);
      if (state.status === "completed") {
        const result = state.result;
        stopLoading(); renderResult(result); return;
      }
      if (state.status === "failed") {
        const detail = state.error?.detail || state.error?.message || "Optimizasyon tamamlanamadı.";
        const diagnosis = state.error?.diagnosis;
        throw new Error(diagnosis ? `${detail} · Teşhis: ${diagnosis.title}. ${diagnosis.recommendation}` : detail);
      }
      if (state.status === "cancelled") throw new Error("Optimizasyon durduruldu. Tamamlanan adaylar önbellekte; aynı ayarlarla yeniden başlatırsanız kaldığı işi tekrar kullanır.");
    }
  } catch (error) {
    stopLoading(); $("errorMessage").textContent = error.message || String(error); showState("errorState");
  }
}

$("fluid").addEventListener("change", () => {
  const preset = presets[$("fluid").value];
  if (preset) { $("density").value = preset.density; $("viscosity").value = preset.viscosity; $("soundSpeed").value = preset.sound; }
});
function toggleSolverSettings() {
  const native = $("airfoilStrategy").value === "flow5_native";
  $("flow5Settings").classList.toggle("hidden", !native);
  $("legacySettings").classList.toggle("hidden", native);
}
function toggleBaselineInput() {
  $("customBaselineField").classList.toggle(
    "hidden",
    $("optimizationMode").value === "wing_only" || $("baselineAirfoil").value !== "custom_dat"
  );
}
function updateSavedAirfoilStatus() {
  const saved = readSavedAirfoil();
  const option = $("wingAirfoilSource").querySelector('option[value="saved"]');
  option.disabled = !saved;
  option.textContent = saved ? `Son optimize profil · ${saved.name}` : "Son optimize edilen profil · henüz yok";
  $("savedAirfoilStatus").textContent = saved
    ? `${saved.name} hazır · ${new Date(saved.created_at).toLocaleString("tr-TR")}`
    : "Henüz bu tarayıcıda kaydedilmiş optimize profil yok.";
  if (!saved && $("wingAirfoilSource").value === "saved") $("wingAirfoilSource").value = "file";
}
function toggleWingAirfoilSource() {
  $("wingAirfoilFileField").classList.toggle("hidden", $("wingAirfoilSource").value !== "file");
}
function toggleWorkflowMode() {
  const mode = $("optimizationMode").value;
  $("wingAirfoilFields").classList.toggle("hidden", mode !== "wing_only");
  $("wingEnvelopeSection").classList.toggle("workflow-disabled", mode === "foil_only");
  $("profileEnvelopeSection").classList.toggle("workflow-disabled", mode === "wing_only");
  $("flow5CoupledIterations").disabled = mode !== "coupled";
  $("spanwiseFoilOptimization").disabled = mode !== "coupled";
  $("runButton").querySelector("span").textContent = mode === "foil_only"
    ? "Profil optimizasyonunu başlat"
    : mode === "wing_only"
      ? "Kanat optimizasyonunu başlat"
      : "Optimizasyonu başlat";
  updateSavedAirfoilStatus();
  toggleWingAirfoilSource();
  toggleBaselineInput();
}
function toggleOptionalPanels() {
  $("structureFields").classList.toggle("disabled-fields", !$("structureEnabled").checked);
  $("hydroFields").classList.toggle("disabled-fields", !$("hydroEnabled").checked);
  $("multiSectionFields").classList.toggle("disabled-fields", !$("multiSectionGeometry").checked);
  $("wingletFields").classList.toggle("disabled-fields", !$("wingletOptimization").checked);
  $("spanwiseFoilFields").classList.toggle("disabled-fields", !$("spanwiseFoilOptimization").checked);
  $("meshConvergenceFields").classList.toggle("disabled-fields", !$("meshConvergence").checked);
  $("surrogateFields").classList.toggle("disabled-fields", !$("surrogateEnabled").checked);
  $("budgetEscalationFields").classList.toggle("disabled-fields", !$("budgetEscalation").checked);
  $("validationFields").classList.toggle("disabled-fields", !$("validationEnabled").checked);
}
$("airfoilStrategy").addEventListener("change", toggleSolverSettings);
$("baselineAirfoil").addEventListener("change", toggleBaselineInput);
$("optimizationMode").addEventListener("change", toggleWorkflowMode);
$("wingAirfoilSource").addEventListener("change", toggleWingAirfoilSource);
["structureEnabled","hydroEnabled","multiSectionGeometry","wingletOptimization","spanwiseFoilOptimization","meshConvergence","surrogateEnabled","budgetEscalation","validationEnabled"].forEach((id) => $(id).addEventListener("change", toggleOptionalPanels));
[$("paretoX"), $("paretoY")].forEach((element) => element.addEventListener("change", () => { if (lastResult) renderPareto(lastResult); }));
[$("historyA"), $("historyB")].forEach((element) => element.addEventListener("change", renderHistory));
$("clearHistory").addEventListener("click", () => {
  if (!window.confirm("Kaydedilmiş tasarım özetlerinin tümü silinsin mi?")) return;
  try { localStorage.removeItem(HISTORY_KEY); } catch { /* private mode */ }
  renderHistory();
});
$("cancelButton").addEventListener("click", async () => {
  if (!activeJobId) return;
  $("cancelButton").disabled = true;
  $("loadingMessage").textContent = "Çalışan flow5 süreci durduruluyor…";
  try { await fetch(`/api/jobs/${activeJobId}/cancel`, { method:"POST" }); } catch { /* polling reports final state */ }
});
toggleSolverSettings();
toggleBaselineInput();
toggleOptionalPanels();
toggleWorkflowMode();
$("designForm").addEventListener("submit", optimize);
$("dismissError").addEventListener("click", () => { showState(lastResult ? "resultState" : "emptyState"); window.scrollTo({ top: 0, behavior: "smooth" }); });
$("resetButton").addEventListener("click", async () => {
  try { const response = await fetch("/api/defaults"); applyDefaults(await response.json()); } catch { window.location.reload(); }
});
