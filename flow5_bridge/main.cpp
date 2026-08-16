// SPDX-License-Identifier: GPL-3.0-or-later
// AeroOpt flow5 API runner. This program links to GPL-licensed flow5 libraries.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#ifdef _MSC_VER
// The binary Gmsh SDK recommends this C-API-backed wrapper when the SDK's C++
// compiler ABI is not guaranteed to match.  It must be included before
// flow5's api.h, which otherwise includes gmsh.h with the same include guard.
#include <gmsh.h_cwrap>
#else
#include <gmsh.h>
#endif

#include <api.h>
#include <flow5-io.h>
#include <foil.h>
#include <objects2d.h>
#include <objects2d_globals.h>
#include <objects3d.h>
#include <panelanalysis.h>
#include <panel.h>
#include <panel3.h>
#include <panel4.h>
#include <planepolar.h>
#include <planepolarnamemaker.h>
#include <planetask.h>
#include <planeopp.h>
#include <planexfl.h>
#include <polar.h>
#include <polarnamemaker.h>
#include <xfoiltask.h>
#include <wingopp.h>
#include <wingxfl.h>

#ifndef AEROPT_FLOW5_VERSION
#define AEROPT_FLOW5_VERSION "7.57"
#endif

namespace fs = std::filesystem;

static constexpr const char *kProtocol = "aeropt-flow5-v1";

static double number(const QJsonObject &object, const char *key)
{
    const QJsonValue value = object.value(QString::fromUtf8(key));
    if (!value.isDouble()) throw std::runtime_error(std::string("Missing numeric field: ") + key);
    const double result = value.toDouble();
    if (!std::isfinite(result)) throw std::runtime_error(std::string("Non-finite field: ") + key);
    return result;
}

static double numberOr(const QJsonObject &object, const char *key, double fallback)
{
    const QJsonValue value = object.value(QString::fromUtf8(key));
    return value.isDouble() && std::isfinite(value.toDouble()) ? value.toDouble() : fallback;
}

static int integerOr(const QJsonObject &object, const char *key, int fallback)
{
    return std::max(1, int(std::lround(numberOr(object, key, fallback))));
}

static std::string stringValue(const QJsonObject &object, const char *key)
{
    const QJsonValue value = object.value(QString::fromUtf8(key));
    if (!value.isString() || value.toString().isEmpty()) {
        throw std::runtime_error(std::string("Missing string field: ") + key);
    }
    return value.toString().toStdString();
}

static QJsonObject solverInfo()
{
    return {
        {"name", "flow5"},
        {"version", AEROPT_FLOW5_VERSION},
        {"api", "experimental"},
        {"runner", "aeropt-flow5-runner"},
    };
}

static bool writeJson(const fs::path &path, const QJsonObject &object)
{
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

static QJsonObject readRequest(const fs::path &path)
{
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot open request JSON");
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        throw std::runtime_error("Invalid request JSON: " + error.errorString().toStdString());
    }
    QJsonObject request = document.object();
    if (request.value("protocol").toString().toStdString() != kProtocol) {
        throw std::runtime_error("Protocol mismatch");
    }
    return request;
}

static Foil *loadFoil(
    const QJsonObject &request,
    const std::string &pathKey,
    const std::string &foilName
)
{
    const QJsonObject paths = request.value("paths").toObject();
    const std::string foilPath = stringValue(paths, pathKey.c_str());
    auto *foil = new Foil;
    int lineError = -1;
    if (!foil::readFoilFile(foilPath, foil, lineError)) {
        delete foil;
        throw std::runtime_error("Could not import foil DAT; line " + std::to_string(lineError));
    }
    const int expectedPoints = integerOr(request, "foil_coordinate_points", 100);
    if (expectedPoints != 100) {
        delete foil;
        throw std::runtime_error("AeroOpt requires exactly 100 foil coordinate points");
    }
    if (foil->nBaseNodes() != expectedPoints || foil->nNodes() != expectedPoints) {
        const int baseCount = foil->nBaseNodes();
        const int activeCount = foil->nNodes();
        delete foil;
        throw std::runtime_error(
            "flow5 imported " + std::to_string(baseCount) + " base / " +
            std::to_string(activeCount) + " active foil points; expected 100"
        );
    }
    foil->setName(foilName);
    Objects2d::insertThisFoil(foil);
    return foil;
}

static fs::path saveProject(const QJsonObject &request, const std::string &filename)
{
    const fs::path outputDir(stringValue(request, "output_dir"));
    const fs::path projectPath = outputDir / filename;
    std::string log;
    if (!io::saveProject(projectPath.string(), log)) {
        throw std::runtime_error("flow5 could not save project: " + log);
    }
    return projectPath;
}

static void normalizeStepMetreUnit(const fs::path &stepPath)
{
    std::ifstream input(stepPath, std::ios::binary);
    if (!input) throw std::runtime_error("Could not reopen STEP file for unit validation");
    std::string contents{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}
    };
    input.close();
    const std::string metre = "SI_UNIT($,.METRE.)";
    const std::string millimetre = "SI_UNIT(.MILLI.,.METRE.)";
    bool changed = false;
    std::size_t position = 0;
    while ((position = contents.find(millimetre, position)) != std::string::npos) {
        contents.replace(position, millimetre.size(), metre);
        position += metre.size();
        changed = true;
    }
    if (contents.find(metre) == std::string::npos) {
        throw std::runtime_error("STEP writer did not expose a supported metre/millimetre length unit");
    }
    if (changed) {
        std::ofstream output(stepPath, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Could not normalize STEP length unit to metres");
        output.write(contents.data(), std::streamsize(contents.size()));
        if (!output) throw std::runtime_error("Could not finish STEP metre-unit normalization");
    }
}

static fs::path saveStepWing(const QJsonObject &request)
{
    const QJsonArray sections = request.value("step_sections").toArray();
    if (sections.size() < 3) {
        throw std::runtime_error("STEP loft requires at least three wing sections");
    }
    const int sectionPointCount = sections.at(0).toArray().size();
    if (sectionPointCount < 40) {
        throw std::runtime_error("STEP wing sections require at least 40 contour points");
    }

    gmsh::model::add("AeroOpt STEP Wing");
    std::vector<int> wireTags;
    wireTags.reserve(std::size_t(sections.size()));
    for (int sectionIndex = 0; sectionIndex < sections.size(); ++sectionIndex) {
        const QJsonArray section = sections.at(sectionIndex).toArray();
        if (section.size() != sectionPointCount) {
            throw std::runtime_error("STEP wing sections must have matched contour sizes");
        }
        std::vector<int> pointTags;
        pointTags.reserve(std::size_t(sectionPointCount + 1));
        for (int pointIndex = 0; pointIndex < section.size(); ++pointIndex) {
            const QJsonArray coordinates = section.at(pointIndex).toArray();
            if (coordinates.size() != 3) {
                throw std::runtime_error("STEP wing contour point must contain x, y and z");
            }
            const double x = coordinates.at(0).toDouble(std::numeric_limits<double>::quiet_NaN());
            const double y = coordinates.at(1).toDouble(std::numeric_limits<double>::quiet_NaN());
            const double z = coordinates.at(2).toDouble(std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                throw std::runtime_error("STEP wing contour contains a non-finite coordinate");
            }
            pointTags.push_back(gmsh::model::occ::addPoint(x, y, z));
        }
        pointTags.push_back(pointTags.front());
        const int curveTag = gmsh::model::occ::addSpline(pointTags);
        wireTags.push_back(gmsh::model::occ::addWire({curveTag}, -1, true));
    }

    std::vector<std::pair<int, int>> loftEntities;
    gmsh::model::occ::addThruSections(
        wireTags,
        loftEntities,
        -1,
        true,
        true
    );
    gmsh::model::occ::synchronize();
    std::vector<std::pair<int, int>> volumes;
    gmsh::model::getEntities(volumes, 3);
    if (volumes.empty()) {
        throw std::runtime_error("OpenCascade STEP loft did not produce a closed solid");
    }

    gmsh::option::setString("Geometry.OCCSTEPModelName", "AeroOpt optimized wing");
    gmsh::option::setString("Geometry.OCCSTEPAuthor", "AeroOpt");
    const fs::path outputDir(stringValue(request, "output_dir"));
    const fs::path stepPath = outputDir / "aeropt-wing.step";
    gmsh::write(stepPath.string());
    normalizeStepMetreUnit(stepPath);
    if (!fs::is_regular_file(stepPath) || fs::file_size(stepPath) < 256) {
        throw std::runtime_error("OpenCascade reported success but no usable STEP file was written");
    }
    return stepPath;
}

static QJsonObject runFoil(const QJsonObject &request)
{
    Foil *foil = loadFoil(request, "foil.dat", stringValue(request, "foil_name"));
    const QJsonArray cases = request.value("cases").toArray();
    if (cases.isEmpty()) throw std::runtime_error("No foil cases supplied");
    const QJsonObject alpha = request.value("alpha").toObject();
    const QJsonObject transition = request.value("transition").toObject();
    const double alphaMin = number(alpha, "min_deg");
    const double alphaMax = number(alpha, "max_deg");
    const double alphaStep = number(alpha, "step_deg");
    const double ncrit = numberOr(transition, "ncrit", 9.0);
    const double xtrTop = numberOr(transition, "xtr_top", 1.0);
    const double xtrBottom = numberOr(transition, "xtr_bottom", 1.0);
    const int maxThreads = std::max(1, integerOr(request, "max_threads", 1));

    std::vector<Polar *> polars;
    std::vector<std::unique_ptr<XFoilTask>> tasks;
    polars.reserve(cases.size());
    tasks.reserve(cases.size());
    for (int index = 0; index < cases.size(); ++index) {
        const QJsonObject item = cases.at(index).toObject();
        auto *polar = Objects2d::createPolar(
            foil,
            xfl::T1POLAR,
            number(item, "reynolds"),
            numberOr(item, "mach", 0.0),
            ncrit,
            xtrTop,
            xtrBottom
        );
        polar->setName("AeroOpt-XFoil-" + std::to_string(index));
        Objects2d::insertPolar(polar);
        polars.push_back(polar);
        auto task = std::make_unique<XFoilTask>();
        task->initialize(*foil, polar, false);
        if (alphaMin < 0.0 && alphaMax > 0.0) {
            task->appendRange({true, 0.0, alphaMax, alphaStep});
            task->appendRange({true, 0.0, alphaMin, alphaStep});
        } else {
            task->appendRange({true, alphaMin, alphaMax, alphaStep});
        }
        tasks.push_back(std::move(task));
    }

    for (std::size_t begin = 0; begin < tasks.size(); begin += std::size_t(maxThreads)) {
        const std::size_t end = std::min(tasks.size(), begin + std::size_t(maxThreads));
        std::vector<std::thread> workers;
        for (std::size_t index = begin; index < end; ++index) {
            workers.emplace_back(&XFoilTask::run, tasks[index].get());
        }
        for (auto &worker : workers) worker.join();
    }

    QJsonArray outputPolars;
    for (int caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const QJsonObject inputCase = cases.at(caseIndex).toObject();
        Polar *polar = polars.at(std::size_t(caseIndex));
        QJsonArray points;
        const std::size_t count = std::min({polar->m_Alpha.size(), polar->m_Cl.size(), polar->m_Cd.size()});
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(polar->m_Alpha[index]) || !std::isfinite(polar->m_Cl[index]) ||
                !std::isfinite(polar->m_Cd[index]) || polar->m_Cd[index] <= 0.0) continue;
            QJsonObject point{
                {"alpha_deg", polar->m_Alpha[index]},
                {"cl", polar->m_Cl[index]},
                {"cd", polar->m_Cd[index]},
            };
            if (index < polar->m_Cdp.size()) point.insert("cdp", polar->m_Cdp[index]);
            if (index < polar->m_Cm.size()) point.insert("cm_c4", polar->m_Cm[index]);
            points.append(point);
        }
        outputPolars.append(QJsonObject{
            {"speed_m_s", number(inputCase, "speed_m_s")},
            {"reynolds", polar->Reynolds()},
            {"mach", polar->Mach()},
            {"points", points},
        });
    }

    QJsonObject artifacts;
    if (request.value("save_project").toBool(false)) {
        artifacts.insert("project_fl5", QString::fromStdString(saveProject(request, "aeropt-foil.fl5").string()));
    }
    QJsonObject response{
        {"protocol", kProtocol},
        {"ok", true},
        {"mode", "foil"},
        {"solver", solverInfo()},
        {"threading", QJsonObject{
            {"requested_cpu_budget", maxThreads},
            {"case_workers", std::min(maxThreads, int(cases.size()))},
            {"blas_threads_per_process", 1},
            {"nested_parallelism_disabled", true},
        }},
        {"foil_coordinate_points_used", 100},
        {"polars", outputPolars},
        {"artifacts", artifacts},
    };
    globals::deleteObjects();
    return response;
}

static xfl::enumAnalysisMethod analysisMethod(const std::string &method)
{
    if (method == "LLT") return xfl::LLT;
    if (method == "VLM1") return xfl::VLM1;
    if (method == "VLM2") return xfl::VLM2;
    if (method == "QUADS") return xfl::QUADS;
    if (method == "TRIUNIFORM") return xfl::TRIUNIFORM;
    if (method == "TRILINEAR") return xfl::TRILINEAR;
    throw std::runtime_error("Unsupported flow5 method: " + method);
}

static std::vector<double> alphaList(double minimum, double maximum, double step)
{
    if (!(step > 0.0) || maximum < minimum) throw std::runtime_error("Invalid alpha range");
    std::vector<double> values;
    for (int index = 0; ; ++index) {
        const double value = minimum + double(index) * step;
        if (value > maximum + 1e-9) break;
        values.push_back(std::min(value, maximum));
    }
    if (values.empty() || values.back() < maximum - 1e-9) values.push_back(maximum);
    return values;
}

static const PlaneOpp *nearestPlaneOpp(const PlaneTask &task, double alpha)
{
    const PlaneOpp *nearest = nullptr;
    double best = std::numeric_limits<double>::infinity();
    for (const PlaneOpp *opp : task.planeOppList()) {
        if (!opp || !std::isfinite(opp->alpha())) continue;
        const double distance = std::abs(opp->alpha() - alpha);
        if (distance < best) {
            nearest = opp;
            best = distance;
        }
    }
    return best <= 1.0e-5 ? nearest : nullptr;
}

static QJsonArray spanDistribution(const SpanDistribs &span, double dynamicPressure)
{
    QJsonArray distribution;
    const std::size_t count = std::min({
        span.m_Cl.size(), span.m_Chord.size(), span.m_StripPos.size()
    });
    for (std::size_t station = 0; station < count; ++station) {
        const double cl = span.m_Cl[station];
        const double chord = span.m_Chord[station];
        const double y = span.m_StripPos[station];
        if (!std::isfinite(cl) || !std::isfinite(chord) || !std::isfinite(y)) continue;
        QJsonObject item{
            {"y_m", y},
            {"chord_m", chord},
            {"local_cl", cl},
            {"lift_n_per_m", dynamicPressure * cl * chord},
            {"converged", station >= span.m_bConverged.size() || span.m_bConverged[station]},
        };
        if (station < span.m_Re.size() && std::isfinite(span.m_Re[station]))
            item.insert("reynolds", span.m_Re[station]);
        if (station < span.m_Ai.size() && std::isfinite(span.m_Ai[station]))
            item.insert("induced_angle_deg", span.m_Ai[station]);
        if (station < span.m_ICd.size() && std::isfinite(span.m_ICd[station]))
            item.insert("cdi", span.m_ICd[station]);
        if (station < span.m_PCd.size() && std::isfinite(span.m_PCd[station]))
            item.insert("cdv", span.m_PCd[station]);
        if (station < span.m_BendingMoment.size() && std::isfinite(span.m_BendingMoment[station]))
            item.insert("bending_moment_nm", span.m_BendingMoment[station]);
        if (station < span.m_Twist.size() && std::isfinite(span.m_Twist[station]))
            item.insert("twist_deg", span.m_Twist[station]);
        distribution.append(item);
    }
    return distribution;
}

static QString surfaceName(const Panel &panel)
{
    if (panel.isTopPanel()) return "upper";
    if (panel.isBotPanel()) return "lower";
    if (panel.isMidPanel()) return "mid";
    if (panel.isSidePanel()) return "side";
    return "other";
}

static QJsonArray vectorArray(const Vector3d &point)
{
    return {point.x, point.y, point.z};
}

static QJsonObject panelRecord(
    const Panel &panel,
    int wingIndex,
    int surfaceCount,
    bool wingletActive,
    double cp,
    const QJsonArray &vertices
)
{
    const Vector3d &center = panel.CoG();
    const Vector3d &normal = panel.normal();
    const int surfaceIndex = panel.surfaceIndex();
    const bool wingletPanel = wingletActive && surfaceCount > 1 &&
        (surfaceIndex == 0 || surfaceIndex == surfaceCount - 1);
    return {
        {"panel_index", panel.index()},
        {"wing_index", wingIndex},
        {"surface_index", surfaceIndex},
        {"surface", surfaceName(panel)},
        {"component", wingletPanel ? "winglet" : "main_wing"},
        {"side", center.y < -1.0e-10 ? "left" : center.y > 1.0e-10 ? "right" : "center"},
        {"x_m", center.x},
        {"y_m", center.y},
        {"z_m", center.z},
        {"nx", normal.x},
        {"ny", normal.y},
        {"nz", normal.z},
        {"area_m2", panel.area()},
        {"cp", cp},
        {"leading_edge_panel", panel.isLeading()},
        {"trailing_edge_panel", panel.isTrailing()},
        {"vertices", vertices},
    };
}

static QJsonObject panelTelemetry(
    const PlaneXfl &plane,
    const PlaneOpp &opp,
    const PlanePolar &polar,
    bool wingletActive,
    double targetCl,
    double sampledCl,
    double sampledAlpha
)
{
    QJsonArray panels;
    double totalArea = 0.0;
    for (int wingIndex = 0; wingIndex < plane.nWings(); ++wingIndex) {
        const WingXfl *wing = plane.wingAt(wingIndex);
        if (!wing) continue;
        const int surfaceCount = wing->nSurfaces();
        if (polar.isTriangleMethod()) {
            const int first = wing->firstPanel3Index();
            const std::vector<double> &cpValues = opp.Cp();
            for (int local = 0; local < wing->nPanel3(); ++local) {
                const Panel3 &panel = plane.panel3At(first + local);
                const int cpIndex = panel.index() * 3;
                if (cpIndex < 0 || cpIndex + 2 >= int(cpValues.size())) continue;
                const double cp = (
                    cpValues[std::size_t(cpIndex)] +
                    cpValues[std::size_t(cpIndex + 1)] +
                    cpValues[std::size_t(cpIndex + 2)]
                ) / 3.0;
                if (!std::isfinite(cp) || !(panel.area() > 0.0)) continue;
                QJsonArray vertices;
                for (int vertex = 0; vertex < 3; ++vertex)
                    vertices.append(vectorArray(panel.vertexAt(vertex)));
                panels.append(panelRecord(
                    panel,
                    wingIndex,
                    surfaceCount,
                    wingletActive,
                    cp,
                    vertices
                ));
                totalArea += panel.area();
            }
        } else if (polar.isQuadMethod() && wingIndex < opp.nWOpps()) {
            const WingOpp &wingOpp = opp.WOpp(wingIndex);
            if (!wingOpp.m_dCp) continue;
            const int first = wing->firstPanel4Index();
            const int count = std::min(wing->nPanel4(), wingOpp.m_nPanel4);
            for (int local = 0; local < count; ++local) {
                const Panel4 &panel = plane.panel4(first + local);
                const double cp = wingOpp.m_dCp[local];
                if (!std::isfinite(cp) || !(panel.area() > 0.0)) continue;
                QJsonArray vertices;
                for (int vertex = 0; vertex < 4; ++vertex)
                    vertices.append(vectorArray(panel.vertex(vertex)));
                panels.append(panelRecord(
                    panel,
                    wingIndex,
                    surfaceCount,
                    wingletActive,
                    cp,
                    vertices
                ));
                totalArea += panel.area();
            }
        }
    }
    return {
        {"target_cl", targetCl},
        {"sampled_cl", sampledCl},
        {"sampled_alpha_deg", sampledAlpha},
        {"panel_count", int(panels.size())},
        {"panel_area_sum_m2", totalArea},
        {"thin_surfaces", polar.bThinSurfaces()},
        {"upper_lower_resolved", !polar.bThinSurfaces()},
        {"cp_definition", polar.bThinSurfaces()
            ? "flow5 thin-surface panel pressure coefficient"
            : "flow5 resolved surface pressure coefficient"},
        {"panels", panels},
    };
}

static QJsonObject runWing(const QJsonObject &request)
{
    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 0);
    gmsh::option::setNumber("Geometry.OCCParallel", 1.0);
    gmsh::option::setNumber("General.NumThreads", integerOr(request, "max_threads", 1));

    const QJsonArray sectionFoils = request.value("section_foils").toArray();
    if (sectionFoils.isEmpty()) {
        loadFoil(request, "foil.dat", stringValue(request, "foil_name"));
    } else {
        for (const QJsonValue value : sectionFoils) {
            const QJsonObject sectionFoil = value.toObject();
            loadFoil(
                request,
                stringValue(sectionFoil, "path_key"),
                stringValue(sectionFoil, "name")
            );
        }
    }
    const QJsonObject paths = request.value("paths").toObject();
    std::string importLog;
    PlaneXfl *plane = io::importPlaneFromXML(stringValue(paths, "plane.xml"), importLog);
    if (!plane) throw std::runtime_error("Could not import flow5 plane XML: " + importLog);
    Objects3d::insertPlane(plane);
    plane->makePlane(false, false, true);

    const std::string methodName = stringValue(request, "method");
    const xfl::enumAnalysisMethod method = analysisMethod(methodName);
    const QJsonObject fluid = request.value("fluid").toObject();
    const QJsonObject alpha = request.value("alpha").toObject();
    const QJsonObject transition = request.value("transition").toObject();
    const double density = number(fluid, "density_kg_m3");
    const double viscosity = number(fluid, "kinematic_viscosity_m2_s");
    const double alphaMin = number(alpha, "min_deg");
    const double alphaMax = number(alpha, "max_deg");
    const double alphaStep = number(alpha, "step_deg");
    const int maxThreads = std::max(1, integerOr(request, "max_threads", 1));
    const bool panelTelemetryRequested = request.value("panel_telemetry").toBool(false);
    const bool wingletActive = request.value("winglet_active").toBool(false);
    const bool thinSurfaces = request.value("thin_surfaces").toBool(true);
    const double panelTelemetryTargetLift = numberOr(
        request,
        "panel_telemetry_target_lift_n",
        std::numeric_limits<double>::quiet_NaN()
    );
    PanelAnalysis::setMaxThreadCount(maxThreads);
    const QJsonArray cases = request.value("cases").toArray();
    if (cases.isEmpty()) throw std::runtime_error("No wing cases supplied");

    QJsonArray outputCases;
    int maximumPanel4Count = 0;
    int maximumPanel3Count = 0;
    for (int caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const double speed = number(cases.at(caseIndex).toObject(), "speed_m_s");
        auto *polar = new PlanePolar;
        polar->setPlaneName(plane->name());
        polar->setType(xfl::T1POLAR);
        polar->setAnalysisMethod(method);
        polar->setReferenceDim(xfl::PROJECTED);
        polar->setReferenceArea(plane->projectedArea());
        polar->setReferenceSpanLength(plane->projectedSpan());
        polar->setReferenceChordLength(plane->mac());
        polar->setThinSurfaces(thinSurfaces);
        polar->setViscous(true);
        polar->setViscOnTheFly(true);
        polar->setViscFromCl(false);
        polar->setDensity(density);
        polar->setViscosity(viscosity);
        polar->setNCrit(numberOr(transition, "ncrit", 9.0));
        polar->setXTrTop(numberOr(transition, "xtr_top", 1.0));
        polar->setXTrBot(numberOr(transition, "xtr_bottom", 1.0));
        polar->setTransAtHinge(true);
        polar->setVelocity(speed);
        polar->setName("AeroOpt-" + methodName + "-" + std::to_string(caseIndex));
        Objects3d::insertPlanePolar(polar);

        auto task = std::make_unique<PlaneTask>();
        task->outputToStdIO(false);
        task->setKeepOpps(true);
        task->setObjects(plane, polar);
        task->setComputeDerivatives(false);
        task->setOppList(alphaList(alphaMin, alphaMax, alphaStep));
        task->run();

        QJsonArray points;
        const std::size_t count = std::min(polar->m_Alpha.size(), polar->m_AF.size());
        const double q = 0.5 * density * speed * speed;
        int telemetryIndex = -1;
        double telemetryTargetCl = std::numeric_limits<double>::quiet_NaN();
        if (panelTelemetryRequested && std::isfinite(panelTelemetryTargetLift)) {
            telemetryTargetCl = panelTelemetryTargetLift /
                std::max(q * polar->referenceArea(), 1.0e-12);
            double bestClDistance = std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < count; ++index) {
                const double cl = polar->m_AF[index].CL();
                if (!std::isfinite(cl)) continue;
                const double distance = std::abs(cl - telemetryTargetCl);
                if (distance < bestClDistance) {
                    telemetryIndex = int(index);
                    bestClDistance = distance;
                }
            }
        }
        for (std::size_t index = 0; index < count; ++index) {
            const AeroForces &forces = polar->m_AF[index];
            const double cl = forces.CL();
            const double cd = forces.CD();
            if (!std::isfinite(cl) || !std::isfinite(cd) || cd <= 0.0) continue;
            const double bending = index < polar->m_MaxBending.size() ? polar->m_MaxBending[index] : 0.0;
            const PlaneOpp *opp = nearestPlaneOpp(*task, polar->m_Alpha[index]);
            bool outOfMesh = false;
            bool viscousConverged = true;
            double convergedFraction = 1.0;
            int stationCount = 0;
            int panel4Count = 0;
            int panel3Count = 0;
            QJsonArray distribution;
            double cpMin = std::numeric_limits<double>::infinity();
            if (opp) {
                outOfMesh = opp->isOut();
                panel4Count = opp->nPanel4();
                panel3Count = opp->nPanel3();
                maximumPanel4Count = std::max(maximumPanel4Count, panel4Count);
                maximumPanel3Count = std::max(maximumPanel3Count, panel3Count);
                for (double cp : opp->Cp()) {
                    if (std::isfinite(cp)) cpMin = std::min(cpMin, cp);
                }
                if (opp->hasWOpp() && opp->nWOpps() > 0) {
                    const WingOpp &wingOpp = opp->WOpp(0);
                    outOfMesh = outOfMesh || wingOpp.m_bOut;
                    const SpanDistribs &span = wingOpp.spanResults();
                    stationCount = span.nStations();
                    if (!span.m_bConverged.empty()) {
                        const int converged = int(std::count(
                            span.m_bConverged.begin(), span.m_bConverged.end(), true
                        ));
                        convergedFraction = double(converged) / double(span.m_bConverged.size());
                        viscousConverged = converged == int(span.m_bConverged.size());
                    }
                    distribution = spanDistribution(span, q);
                }
            }
            QJsonObject point{
                {"alpha_deg", polar->m_Alpha[index]},
                {"cl", cl},
                {"cd", cd},
                {"cdi", forces.CDi()},
                {"cdv", forces.CDv()},
                {"cm", forces.Cm()},
                {"lift_n", q * polar->referenceArea() * cl},
                {"drag_n", q * polar->referenceArea() * cd},
                {"root_bending_moment_nm", bending},
                {"out_of_mesh", outOfMesh},
                {"viscous_converged", viscousConverged},
                {"viscous_converged_fraction", convergedFraction},
                {"station_count", stationCount},
                {"panel4_count", panel4Count},
                {"panel3_count", panel3Count},
                {"distribution", distribution},
            };
            if (std::isfinite(cpMin)) point.insert("cp_min", cpMin);
            points.append(point);
        }
        QJsonObject outputCase{
            {"speed_m_s", speed},
            {"method", QString::fromStdString(methodName)},
            {"points", points},
            {"raw_export", QString::fromStdString(polar->exportToString(","))},
        };
        if (telemetryIndex >= 0 && telemetryIndex < int(count)) {
            const PlaneOpp *telemetryOpp = nearestPlaneOpp(
                *task,
                polar->m_Alpha[std::size_t(telemetryIndex)]
            );
            if (telemetryOpp) {
                outputCase.insert(
                    "panel_telemetry",
                    panelTelemetry(
                        *plane,
                        *telemetryOpp,
                        *polar,
                        wingletActive,
                        telemetryTargetCl,
                        polar->m_AF[std::size_t(telemetryIndex)].CL(),
                        polar->m_Alpha[std::size_t(telemetryIndex)]
                    )
                );
            }
        }
        outputCases.append(outputCase);
    }

    QJsonObject artifacts;
    if (request.value("save_project").toBool(false)) {
        artifacts.insert("project_fl5", QString::fromStdString(saveProject(request, "aeropt-optimized.fl5").string()));
    }
    if (request.value("save_step").toBool(false)) {
        artifacts.insert("wing_step", QString::fromStdString(saveStepWing(request).string()));
    }
    QJsonObject response{
        {"protocol", kProtocol},
        {"ok", true},
        {"mode", "wing"},
        {"solver", solverInfo()},
        {"threading", QJsonObject{
            {"requested_cpu_budget", maxThreads},
            {"panel_thread_limit", maxThreads},
            {"blas_threads_per_process", 1},
            {"nested_parallelism_disabled", true},
        }},
        {"foil_coordinate_points_used", 100},
        {"mesh", QJsonObject{
            {"chordwise_panels", request.value("mesh").toObject().value("chordwise_panels")},
            {"half_span_panels", request.value("mesh").toObject().value("half_span_panels")},
            {"actual_panel4_count", maximumPanel4Count},
            {"actual_panel3_count", maximumPanel3Count},
        }},
        {"cases", outputCases},
        {"artifacts", artifacts},
    };
    globals::deleteObjects();
    gmsh::finalize();
    return response;
}

int main(int argc, char **argv)
{
    QCoreApplication application(argc, argv);
    fs::path requestPath;
    fs::path responsePath;
    for (int index = 1; index + 1 < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--request") requestPath = fs::path(argv[++index]);
        else if (argument == "--response") responsePath = fs::path(argv[++index]);
    }
    if (requestPath.empty() || responsePath.empty()) {
        std::cerr << "Usage: aeropt-flow5-runner --request request.json --response response.json\n";
        return 2;
    }
    try {
        const QJsonObject request = readRequest(requestPath);
        const std::string mode = stringValue(request, "mode");
        QJsonObject response;
        if (mode == "foil") response = runFoil(request);
        else if (mode == "wing") response = runWing(request);
        else throw std::runtime_error("Unknown mode: " + mode);
        if (!writeJson(responsePath, response)) {
            std::cerr << "Could not write response JSON\n";
            return 3;
        }
        return 0;
    } catch (const std::exception &error) {
        QJsonObject response{
            {"protocol", kProtocol},
            {"ok", false},
            {"solver", solverInfo()},
            {"error", QString::fromUtf8(error.what())},
        };
        writeJson(responsePath, response);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
