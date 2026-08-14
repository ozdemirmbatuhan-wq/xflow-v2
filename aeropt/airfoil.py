from __future__ import annotations

from dataclasses import dataclass
from math import atanh, comb, degrees, pi, radians, sqrt
from typing import Iterable

import numpy as np

from .models import AirfoilDesign, AirfoilLike, CSTAirfoilDesign


def naca4_design(code: str) -> AirfoilDesign:
    """Build a conventional four-digit NACA section from a user-facing code."""
    if not isinstance(code, str):
        raise ValueError("NACA profil kodu metin olmalı")
    normalized = "".join(code.upper().split())
    if normalized.startswith("NACA"):
        normalized = normalized[4:]
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError("Winglet profili dört haneli bir NACA kodu olmalı (ör. 0012)")

    camber_digit = int(normalized[0])
    position_digit = int(normalized[1])
    thickness_digits = int(normalized[2:])
    if camber_digit == 0 and position_digit != 0:
        raise ValueError("Simetrik NACA 4 haneli profilde ilk iki hane 00 olmalı")
    if camber_digit > 0 and position_digit == 0:
        raise ValueError("Kamburlu NACA 4 haneli profilde kambur konumu sıfır olamaz")
    if thickness_digits == 0:
        raise ValueError("NACA profil kalınlığı sıfır olamaz")

    return AirfoilDesign(
        max_camber=camber_digit / 100.0,
        camber_position=(position_digit / 10.0 if position_digit else 0.4),
        thickness=thickness_digits / 100.0,
        name=f"NACA{normalized}",
    )


@dataclass(frozen=True)
class PolarPoint:
    alpha_deg: float
    cl: float
    cd: float
    cm_c4: float
    ld: float

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha_deg": float(self.alpha_deg),
            "cl": float(self.cl),
            "cd": float(self.cd),
            "cm_c4": float(self.cm_c4),
            "ld": float(self.ld),
        }


@dataclass(frozen=True)
class ParsedAirfoilCoordinates:
    """Normalized upper/lower coordinate branches loaded from a DAT file."""

    name: str
    upper_x: tuple[float, ...]
    upper_y: tuple[float, ...]
    lower_x: tuple[float, ...]
    lower_y: tuple[float, ...]
    source_point_count: int


def _numeric_pair(line: str) -> tuple[float, float] | None:
    fields = line.replace(",", " ").split()
    if len(fields) < 2:
        return None
    try:
        return float(fields[0]), float(fields[1])
    except ValueError:
        return None


def _clean_surface(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    unique_x, inverse = np.unique(np.round(x, 10), return_inverse=True)
    summed = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=float)
    np.add.at(summed, inverse, y)
    np.add.at(counts, inverse, 1.0)
    y = summed / np.maximum(counts, 1.0)
    x = unique_x.astype(float)
    if x.size < 8:
        raise ValueError("DAT dosyasının her yüzeyinde en az 8 farklı x/c noktası olmalı")
    if x[0] <= 2e-3:
        x[0] = 0.0
    else:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    if x[-1] >= 1.0 - 2e-3:
        x[-1] = 1.0
    else:
        x = np.append(x, 1.0)
        y = np.append(y, y[-1])
    return x, y


def _normalize_coordinate_branches(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_x = np.concatenate((first[:, 0], second[:, 0]))
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    chord = x_max - x_min
    if not np.isfinite(chord) or chord <= 1e-8:
        raise ValueError("DAT profilinin chord uzunluğu belirlenemedi")

    first_x = (first[:, 0] - x_min) / chord
    second_x = (second[:, 0] - x_min) / chord
    first_y = first[:, 1] / chord
    second_y = second[:, 1] / chord
    le_center = 0.5 * (
        first_y[int(np.argmin(first_x))] + second_y[int(np.argmin(second_x))]
    )
    te_center = 0.5 * (
        first_y[int(np.argmax(first_x))] + second_y[int(np.argmax(second_x))]
    )
    first_y -= (1.0 - first_x) * le_center + first_x * te_center
    second_y -= (1.0 - second_x) * le_center + second_x * te_center

    first_x, first_y = _clean_surface(first_x, first_y)
    second_x, second_y = _clean_surface(second_x, second_y)
    probe = np.linspace(0.05, 0.95, 101)
    first_mean = float(np.mean(np.interp(probe, first_x, first_y)))
    second_mean = float(np.mean(np.interp(probe, second_x, second_y)))
    if first_mean >= second_mean:
        return first_x, first_y, second_x, second_y
    return second_x, second_y, first_x, first_y


def parse_airfoil_dat(text: str, *, fallback_name: str = "Imported-DAT") -> ParsedAirfoilCoordinates:
    """Parse common Lednicer or Selig DAT layouts and normalize them to unit chord."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("DAT profil verisi boş")
    lines = [line.strip() for line in text.replace("\ufeff", "").splitlines()]
    nonempty = [line for line in lines if line]
    if not nonempty:
        raise ValueError("DAT profil verisi boş")
    first_pair = _numeric_pair(nonempty[0])
    name = fallback_name if first_pair is not None else nonempty[0][:80]
    data_lines = nonempty if first_pair is not None else nonempty[1:]

    count_index: int | None = None
    counts: tuple[int, int] | None = None
    for index, line in enumerate(data_lines[:4]):
        pair = _numeric_pair(line)
        if pair is None:
            continue
        upper_count, lower_count = int(round(pair[0])), int(round(pair[1]))
        if (
            upper_count >= 8
            and lower_count >= 8
            and abs(pair[0] - upper_count) < 1e-8
            and abs(pair[1] - lower_count) < 1e-8
        ):
            count_index, counts = index, (upper_count, lower_count)
            break

    if counts is not None and count_index is not None:
        rows = [pair for line in data_lines[count_index + 1 :] if (pair := _numeric_pair(line))]
        expected = counts[0] + counts[1]
        if len(rows) < expected:
            raise ValueError(f"DAT başlığı {expected} koordinat bildiriyor; yalnız {len(rows)} bulundu")
        first = np.asarray(rows[: counts[0]], dtype=float)
        second = np.asarray(rows[counts[0] : expected], dtype=float)
        source_count = expected
    else:
        rows = [pair for line in data_lines if (pair := _numeric_pair(line))]
        if len(rows) < 17:
            raise ValueError("DAT dosyasında en az 17 koordinat noktası olmalı")
        points = np.asarray(rows, dtype=float)
        keep = np.ones(points.shape[0], dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
        points = points[keep]
        reset = next(
            (
                index + 1
                for index in range(len(points) - 1)
                if points[index, 0] > 0.80 and points[index + 1, 0] < 0.20
            ),
            None,
        )
        if reset is not None and reset >= 8 and len(points) - reset >= 8:
            first, second = points[:reset], points[reset:]
        else:
            leading_edge = int(np.argmin(points[:, 0]))
            if leading_edge < 7 or len(points) - leading_edge < 8:
                raise ValueError("DAT üst ve alt yüzeyleri güvenilir biçimde ayrılamadı")
            first, second = points[: leading_edge + 1], points[leading_edge:]
        source_count = int(len(points))

    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("DAT koordinatlarında sonlu olmayan değer var")
    upper_x, upper_y, lower_x, lower_y = _normalize_coordinate_branches(first, second)
    thickness = np.interp(np.linspace(0.0, 1.0, 401), upper_x, upper_y) - np.interp(
        np.linspace(0.0, 1.0, 401), lower_x, lower_y
    )
    if float(np.min(thickness[1:-1])) <= -2e-4 or float(np.max(thickness)) <= 0.01:
        raise ValueError("DAT profil yüzeyleri kesişiyor veya geçerli kalınlık oluşturmuyor")
    return ParsedAirfoilCoordinates(
        name=name,
        upper_x=tuple(float(value) for value in upper_x),
        upper_y=tuple(float(value) for value in upper_y),
        lower_x=tuple(float(value) for value in lower_x),
        lower_y=tuple(float(value) for value in lower_y),
        source_point_count=source_count,
    )


def resample_parsed_coordinates(
    coordinates: ParsedAirfoilCoordinates, *, total_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-resample a source DAT contour to an exact number of solver points."""
    if total_points < 40:
        raise ValueError("Profil çevresinde en az 40 koordinat noktası olmalı")
    upper_count = total_points // 2
    lower_count = total_points - upper_count + 1
    upper_x = 0.5 * (1.0 - np.cos(np.linspace(0.0, pi, upper_count)))
    lower_x = 0.5 * (1.0 - np.cos(np.linspace(0.0, pi, lower_count)))
    upper_y = np.interp(
        upper_x, np.asarray(coordinates.upper_x), np.asarray(coordinates.upper_y)
    )
    lower_y = np.interp(
        lower_x, np.asarray(coordinates.lower_x), np.asarray(coordinates.lower_y)
    )
    contour_x = np.concatenate((upper_x[::-1], lower_x[1:]))
    contour_y = np.concatenate((upper_y[::-1], lower_y[1:]))
    if contour_x.size != total_points:
        raise RuntimeError("Kaynak DAT istenen koordinat sayısına örneklenemedi")
    return contour_x, contour_y


def _bernstein_matrix(x: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack(
        [comb(order, index) * x**index * (1.0 - x) ** (order - index) for index in range(order + 1)]
    )


def cst_surfaces(x: np.ndarray, foil: CSTAirfoilDesign) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate independent Kulfan/CST upper and lower ordinates on common x/c values."""
    x = np.asarray(x, dtype=float)
    order = len(foil.upper_weights) - 1
    basis = _bernstein_matrix(x, order)
    class_function = np.sqrt(np.clip(x, 0.0, 1.0)) * (1.0 - x)
    upper = class_function * (basis @ np.asarray(foil.upper_weights))
    lower = class_function * (basis @ np.asarray(foil.lower_weights))
    upper += 0.5 * foil.trailing_edge_gap * x
    lower -= 0.5 * foil.trailing_edge_gap * x
    return upper, lower


def _naca_common_surfaces(x: np.ndarray, foil: AirfoilDesign) -> tuple[np.ndarray, np.ndarray]:
    t = foil.thickness
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    yc, _ = _camber_line_and_slope(x, foil)
    return yc + yt, yc - yt


def make_cst_airfoil(
    upper_weights: Iterable[float],
    lower_weights: Iterable[float],
    *,
    name: str = "AeroOpt-CST",
    trailing_edge_gap: float = 0.0,
) -> CSTAirfoilDesign:
    """Build a CST design and derive its familiar camber/thickness descriptors."""
    upper_tuple = tuple(float(value) for value in upper_weights)
    lower_tuple = tuple(float(value) for value in lower_weights)
    provisional = CSTAirfoilDesign(
        upper_tuple, lower_tuple, 0.0, 0.4, 0.0, name, float(trailing_edge_gap)
    )
    beta = np.linspace(0.0, pi, 801)
    x = 0.5 * (1.0 - np.cos(beta))
    upper, lower = cst_surfaces(x, provisional)
    thickness_curve = upper - lower
    camber_curve = 0.5 * (upper + lower)
    thickness = float(np.max(thickness_curve))
    camber_index = int(np.argmax(camber_curve))
    max_camber = max(float(camber_curve[camber_index]), 0.0)
    camber_position = float(x[camber_index]) if max_camber > 1e-8 else 0.4
    return CSTAirfoilDesign(
        upper_tuple,
        lower_tuple,
        max_camber,
        camber_position,
        thickness,
        name,
        float(trailing_edge_gap),
    )


def fit_coordinates_to_cst(
    coordinates: ParsedAirfoilCoordinates,
    *,
    order: int = 6,
    name: str | None = None,
) -> CSTAirfoilDesign:
    """Fit normalized DAT surfaces to a smooth CST/Kulfan representation."""
    if order < 2 or order > 8:
        raise ValueError("CST derecesi 2 ile 8 arasında olmalı")
    upper_x = np.asarray(coordinates.upper_x, dtype=float)
    upper_y = np.asarray(coordinates.upper_y, dtype=float)
    lower_x = np.asarray(coordinates.lower_x, dtype=float)
    lower_y = np.asarray(coordinates.lower_y, dtype=float)
    trailing_edge_gap = float(upper_y[-1] - lower_y[-1])

    def fit_surface(x: np.ndarray, y: np.ndarray, te_sign: float) -> np.ndarray:
        interior = (x > 1e-6) & (x < 1.0 - 1e-6)
        xi = x[interior]
        target = y[interior] - te_sign * 0.5 * trailing_edge_gap * xi
        matrix = np.sqrt(xi)[:, None] * (1.0 - xi)[:, None] * _bernstein_matrix(xi, order)
        # Keep the leading-edge curvature well represented without overwhelming
        # the rest of the chord, where the laminar bucket is also shape-sensitive.
        weights = 1.0 + 0.65 * np.exp(-xi / 0.035)
        return np.linalg.lstsq(matrix * weights[:, None], target * weights, rcond=None)[0]

    upper_weights = fit_surface(upper_x, upper_y, 1.0)
    lower_weights = fit_surface(lower_x, lower_y, -1.0)
    return make_cst_airfoil(
        upper_weights,
        lower_weights,
        name=name or f"{coordinates.name}-CST{order}",
        trailing_edge_gap=trailing_edge_gap,
    )


def cst_coordinate_fit_error(
    coordinates: ParsedAirfoilCoordinates,
    foil: CSTAirfoilDesign,
) -> tuple[float, float]:
    """Return RMS and maximum ordinate error relative to the source DAT surfaces."""
    beta = np.linspace(0.0, pi, 801)
    x = 0.5 * (1.0 - np.cos(beta))
    upper, lower = cst_surfaces(x, foil)
    source_upper = np.interp(
        x, np.asarray(coordinates.upper_x), np.asarray(coordinates.upper_y)
    )
    source_lower = np.interp(
        x, np.asarray(coordinates.lower_x), np.asarray(coordinates.lower_y)
    )
    error = np.concatenate((upper - source_upper, lower - source_lower))
    return float(np.sqrt(np.mean(error**2))), float(np.max(np.abs(error)))


def fit_naca_to_cst(foil: AirfoilDesign, order: int = 3, name: str = "AeroOpt-CST") -> CSTAirfoilDesign:
    """Least-squares CST fit used as the free-shape optimizer's smooth starting point."""
    if order < 2 or order > 8:
        raise ValueError("CST derecesi 2 ile 8 arasında olmalı")
    beta = np.linspace(0.0, pi, 401)
    x = 0.5 * (1.0 - np.cos(beta))
    upper, lower = _naca_common_surfaces(x, foil)
    interior = (x > 1e-5) & (x < 1.0 - 1e-5)
    xi = x[interior]
    matrix = np.sqrt(xi)[:, None] * (1.0 - xi)[:, None] * _bernstein_matrix(xi, order)
    upper_weights = np.linalg.lstsq(matrix, upper[interior], rcond=None)[0]
    lower_weights = np.linalg.lstsq(matrix, lower[interior], rcond=None)[0]
    fitted = make_cst_airfoil(upper_weights, lower_weights, name=name)
    # Least squares can undershoot t/c by a few 1e-4, which would incorrectly
    # reject a NACA seed sitting exactly on the user's minimum-thickness bound.
    if fitted.thickness > 1e-9:
        center = 0.5 * (upper_weights + lower_weights)
        half_thickness = 0.5 * (upper_weights - lower_weights)
        scale = foil.thickness / fitted.thickness
        upper_weights = center + scale * half_thickness
        lower_weights = center - scale * half_thickness
    return make_cst_airfoil(upper_weights, lower_weights, name=name)


def _naca_surface_coordinates(
    x: np.ndarray, foil: AirfoilDesign
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = foil.thickness
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    yc, dyc = _camber_line_and_slope(x, foil)
    theta = np.arctan(dyc)
    return (
        x - yt * np.sin(theta),
        yc + yt * np.cos(theta),
        x + yt * np.sin(theta),
        yc - yt * np.cos(theta),
    )


def airfoil_coordinates(
    foil: AirfoilLike, *, total_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly ``total_points`` Selig-ordered contour coordinates."""
    if total_points < 40:
        raise ValueError("Profil çevresinde en az 40 koordinat noktası olmalı")
    upper_count = total_points // 2
    lower_count = total_points - upper_count + 1
    upper_beta = np.linspace(0.0, pi, upper_count)
    lower_beta = np.linspace(0.0, pi, lower_count)
    upper_x_base = 0.5 * (1.0 - np.cos(upper_beta))
    lower_x_base = 0.5 * (1.0 - np.cos(lower_beta))
    if isinstance(foil, CSTAirfoilDesign):
        upper_y, _ = cst_surfaces(upper_x_base, foil)
        _, lower_y = cst_surfaces(lower_x_base, foil)
        upper_x, lower_x = upper_x_base, lower_x_base
    else:
        upper_x, upper_y, _, _ = _naca_surface_coordinates(upper_x_base, foil)
        _, _, lower_x, lower_y = _naca_surface_coordinates(lower_x_base, foil)
    contour_x = np.concatenate((upper_x[::-1], lower_x[1:]))
    contour_y = np.concatenate((upper_y[::-1], lower_y[1:]))
    if contour_x.size != total_points:
        raise RuntimeError("İstenen profil koordinat sayısı üretilemedi")
    return contour_x, contour_y


def cst_geometry_is_valid(
    foil: CSTAirfoilDesign,
    *,
    camber_bounds: tuple[float, float],
    camber_position_bounds: tuple[float, float],
    thickness_bounds: tuple[float, float],
) -> bool:
    """Reject crossed, extreme, or out-of-envelope CST candidates before XFOIL."""
    beta = np.linspace(0.0, pi, 241)
    x = 0.5 * (1.0 - np.cos(beta))
    upper, lower = cst_surfaces(x, foil)
    thickness = upper - lower
    if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)):
        return False
    if float(np.max(np.abs(np.concatenate((upper, lower))))) > 0.35:
        return False
    if float(np.min(thickness[1:-1])) <= 1e-6:
        return False
    ordinate_tolerance = 5e-5
    position_tolerance = 0.002
    return bool(
        thickness_bounds[0] - ordinate_tolerance
        <= foil.thickness
        <= thickness_bounds[1] + ordinate_tolerance
        and camber_bounds[0] - ordinate_tolerance
        <= foil.max_camber
        <= camber_bounds[1] + ordinate_tolerance
        and camber_position_bounds[0] - position_tolerance
        <= foil.camber_position
        <= camber_position_bounds[1] + position_tolerance
    )


def _camber_line_and_slope(x: np.ndarray, foil: AirfoilLike) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(foil, CSTAirfoilDesign):
        upper, lower = cst_surfaces(x, foil)
        camber = 0.5 * (upper + lower)
        return camber, np.gradient(camber, x, edge_order=2)
    m = float(foil.max_camber)
    p = float(foil.camber_position)
    if m <= 1e-12:
        return np.zeros_like(x), np.zeros_like(x)

    left = x < p
    yc = np.empty_like(x)
    slope = np.empty_like(x)
    yc[left] = m / p**2 * (2.0 * p * x[left] - x[left] ** 2)
    slope[left] = 2.0 * m / p**2 * (p - x[left])
    yc[~left] = m / (1.0 - p) ** 2 * (
        (1.0 - 2.0 * p) + 2.0 * p * x[~left] - x[~left] ** 2
    )
    slope[~left] = 2.0 * m / (1.0 - p) ** 2 * (p - x[~left])
    return yc, slope


def naca4_coordinates(foil: AirfoilLike, points_per_side: int = 121) -> tuple[np.ndarray, np.ndarray]:
    """Return any supported foil as a closed Selig contour, TE-upper -> LE -> TE-lower."""
    if points_per_side < 20:
        raise ValueError("points_per_side en az 20 olmalı")
    beta = np.linspace(0.0, pi, points_per_side)
    x = 0.5 * (1.0 - np.cos(beta))
    if isinstance(foil, CSTAirfoilDesign):
        upper, lower = cst_surfaces(x, foil)
        return np.concatenate((x[::-1], x[1:])), np.concatenate((upper[::-1], lower[1:]))
    t = foil.thickness
    # -0.1036 gives a closed trailing edge.
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    yc, dyc = _camber_line_and_slope(x, foil)
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return np.concatenate((xu[::-1], xl[1:])), np.concatenate((yu[::-1], yl[1:]))


def thin_airfoil_properties(foil: AirfoilLike, mach: float = 0.0) -> tuple[float, float, float]:
    """Return (lift slope per radian, zero-lift angle rad, Cm at c/4)."""
    theta = np.linspace(1e-6, pi - 1e-6, 1201)
    x = 0.5 * (1.0 - np.cos(theta))
    _, slope = _camber_line_and_slope(x, foil)
    i0 = float(np.trapezoid(slope, theta))
    i1 = float(np.trapezoid(slope * np.cos(theta), theta))
    i2 = float(np.trapezoid(slope * np.cos(2.0 * theta), theta))
    alpha_l0 = (i0 - i1) / pi
    a1 = 2.0 * i1 / pi
    a2 = 2.0 * i2 / pi
    cm_c4 = -pi / 4.0 * (a1 - a2)
    beta = sqrt(max(1.0 - min(abs(mach), 0.72) ** 2, 0.45))
    lift_slope = 2.0 * pi / beta
    return lift_slope, alpha_l0, cm_c4


def cl_max(foil: AirfoilLike, reynolds: float) -> float:
    """Smooth empirical attached-flow CLmax correlation for preliminary design."""
    re = max(float(reynolds), 2.0e4)
    re_gain = 0.55 * (1.0 - np.exp(-re / 180_000.0))
    camber_gain = 4.2 * foil.max_camber
    thickness_gain = 1.6 * max(0.0, min(foil.thickness, 0.18) - 0.08)
    very_low_re_loss = 0.22 * max(0.0, (70_000.0 - re) / 70_000.0)
    return float(np.clip(0.88 + re_gain + camber_gain + thickness_gain - very_low_re_loss, 0.65, 1.9))


def profile_cd(foil: AirfoilLike, cl: float, reynolds: float, mach: float = 0.0) -> float:
    """Viscous profile-drag correlation; not a boundary-layer CFD calculation."""
    re = max(float(reynolds), 2.0e4)
    cf_laminar = 1.328 / sqrt(re)
    cf_turbulent = 0.074 / re**0.2
    # A moderate natural-transition blend is intentionally conservative.
    transition = float(np.clip(0.35 + 0.12 * np.log10(re / 100_000.0), 0.25, 0.68))
    cf = (1.0 - transition) * cf_laminar + transition * cf_turbulent
    form_factor = 1.0 + 2.7 * foil.thickness + 100.0 * foil.thickness**4
    cd_skin = 2.0 * cf * form_factor
    # Cambered four-digit sections tend to place their low-drag range at a
    # positive lift coefficient. The coefficient is a deliberately smooth
    # design correlation, not an attempt to reproduce one specific NACA polar.
    bucket_center = 13.0 * foil.max_camber
    cd_lift = 0.0075 * (cl - bucket_center) ** 2
    cd_camber = 0.00035 * (foil.max_camber / 0.04) ** 2 if foil.max_camber else 0.0
    cd_shape = 0.0012 + 0.0008 * ((foil.camber_position - 0.4) / 0.25) ** 2
    compressibility = 0.0
    if mach > 0.3:
        compressibility = 0.0025 * ((mach - 0.3) / 0.4) ** 2
    stall_ratio = abs(cl) / max(cl_max(foil, re), 1e-6)
    separation = 0.0 if stall_ratio <= 0.82 else 0.045 * (stall_ratio - 0.82) ** 2
    return float(max(cd_skin + cd_lift + cd_camber + cd_shape + compressibility + separation, 1e-5))


def polar_point(foil: AirfoilLike, alpha_deg: float, reynolds: float, mach: float = 0.0) -> PolarPoint:
    a0, alpha_l0, cm_c4 = thin_airfoil_properties(foil, mach)
    cl_linear = a0 * (radians(alpha_deg) - alpha_l0)
    positive_limit = cl_max(foil, reynolds)
    negative_limit = 0.78 * positive_limit
    limit = positive_limit if cl_linear >= 0.0 else negative_limit
    cl = limit * np.tanh(cl_linear / limit)
    cd = profile_cd(foil, float(cl), reynolds, mach)
    return PolarPoint(float(alpha_deg), float(cl), cd, float(cm_c4), float(cl / cd))


def alpha_for_cl(foil: AirfoilLike, target_cl: float, reynolds: float, mach: float = 0.0) -> float:
    a0, alpha_l0, _ = thin_airfoil_properties(foil, mach)
    limit = cl_max(foil, reynolds) if target_cl >= 0.0 else 0.78 * cl_max(foil, reynolds)
    ratio = float(np.clip(target_cl / limit, -0.985, 0.985))
    return degrees(alpha_l0 + limit * atanh(ratio) / a0)


def generate_polar(
    foil: AirfoilLike,
    reynolds: float,
    mach: float,
    alphas_deg: Iterable[float],
) -> list[PolarPoint]:
    return [polar_point(foil, float(alpha), reynolds, mach) for alpha in alphas_deg]


def foil_name(foil: AirfoilLike) -> str:
    family = "CST" if isinstance(foil, CSTAirfoilDesign) else "NACA"
    return f"AeroOpt-{family}-m{100*foil.max_camber:.1f}-p{100*foil.camber_position:.0f}-t{100*foil.thickness:.1f}"
