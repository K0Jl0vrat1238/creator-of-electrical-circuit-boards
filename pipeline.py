from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize


GreenMode = Literal["scanline", "contour-offset"]


class PipelineError(ValueError):
    """Domain error for invalid input or impossible toolpaths."""


@dataclass(frozen=True)
class PipelineConfig:
    image_path: Path
    width_mm: float | None
    height_mm: float | None
    tool_diameter_mm: float
    drill_diameter_mm: float
    stepover_percent: float
    simplify_tolerance_mm: float
    green_mode: GreenMode
    stock_thickness_mm: float
    cut_speed_mm_s: float
    max_accel_mm_s2: float
    trace_depth_mm: float
    green_depth_mm: float


@dataclass(frozen=True)
class HoleCircle:
    x_px: float
    y_px: float
    radius_px: float


@dataclass(frozen=True)
class PipelineResult:
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    holes: list[HoleCircle]
    green_paths: list[list[tuple[float, float]]]
    paths_paths: list[list[tuple[float, float]]]
    cut_paths: list[list[tuple[float, float]]]


_ANCHOR_RGB = np.array(
    [
        [0, 0, 255],  # blue
        [0, 255, 0],  # green
        [255, 0, 0],  # red
        [0, 0, 0],  # black
    ],
    dtype=np.uint8,
)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    _validate_config(config)
    rgba = _load_rgba_image(config.image_path)

    height_px, width_px = rgba.shape[:2]
    width_mm, height_mm = _resolve_mm_size(width_px, height_px, config.width_mm, config.height_mm)
    px_per_mm = width_px / width_mm

    raw_masks = _classify_color_masks(rgba)
    masks = _segment_color_masks(rgba, raw_masks=raw_masks)
    _validate_color_overlaps(masks)

    tool_radius_px = (config.tool_diameter_mm * px_per_mm) / 2.0
    drill_radius_px = (config.drill_diameter_mm * px_per_mm) / 2.0
    stepover_px = max(1.0, (config.stepover_percent / 100.0) * config.tool_diameter_mm * px_per_mm)
    simplify_tolerance_px = max(0.0, config.simplify_tolerance_mm * px_per_mm)

    hole_centers = _detect_hole_centers(masks["blue"], drill_radius_px)
    holes = [HoleCircle(x, y, drill_radius_px) for x, y in hole_centers]
    
    # Очищаем дубликаты: сливаем касающиеся отверстия в одно по центру
    holes = _merge_hole_clusters(holes)
    
    hole_exclusion = _hole_exclusion_mask(masks["green"].shape, holes)
    black_pad_exclusion = _hole_pad_exclusion_mask(raw_masks["black"], holes)
    black_collision_mask = _clean_mask(
        raw_masks["black"] & (~black_pad_exclusion),
        min_area=max(2, (rgba.shape[0] * rgba.shape[1]) // 80000),
        do_close=False,
        do_open=False,
    )

    black_geometry = _mask_to_geometry(masks["black"])
    _validate_drill_collision_mask(holes, black_collision_mask)

    green_paths = _green_paths(
        masks["green"],
        hole_exclusion=hole_exclusion,
        tool_radius_px=tool_radius_px,
        stepover_px=stepover_px,
        mode=config.green_mode,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    if False and (not green_paths):
        raise PipelineError("Не удалось построить траектории для green.svg: зеленая область пуста или слишком тонкая.")

    black_paths = _black_paths(
        black_geometry,
        holes=holes,
        tool_radius_px=tool_radius_px,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    if not black_paths:
        raise PipelineError("Не удалось построить траектории для paths.svg: черная геометрия пуста.")

    cut_paths = _cut_paths(
        masks["red"],
        simplify_tolerance_px=simplify_tolerance_px,
    )
    if not cut_paths:
        raise PipelineError("Не удалось построить траектории для cut.svg: красная геометрия пуста.")

    return PipelineResult(
        width_px=width_px,
        height_px=height_px,
        width_mm=width_mm,
        height_mm=height_mm,
        holes=holes,
        green_paths=green_paths,
        paths_paths=black_paths,
        cut_paths=cut_paths,
    )


def _validate_config(config: PipelineConfig) -> None:
    if config.green_mode not in {"scanline", "contour-offset"}:
        raise PipelineError("Режим green должен быть scanline или contour-offset.")

    numeric_checks = {
        "Диаметр фрезы": config.tool_diameter_mm,
        "Диаметр сверла": config.drill_diameter_mm,
        "Stepover %": config.stepover_percent,
        "Толщина заготовки": config.stock_thickness_mm,
        "Скорость реза": config.cut_speed_mm_s,
        "Макс. ускорение": config.max_accel_mm_s2,
        "Глубина дорожек": config.trace_depth_mm,
        "Глубина green": config.green_depth_mm,
    }
    for name, value in numeric_checks.items():
        if value <= 0:
            raise PipelineError(f"{name} должно быть > 0.")

    if config.simplify_tolerance_mm < 0:
        raise PipelineError("Допуск упрощения не может быть отрицательным.")

    if not (1 <= config.stepover_percent <= 100):
        raise PipelineError("Stepover должен быть в диапазоне 1..100%.")

    if (config.width_mm is None) == (config.height_mm is None):
        raise PipelineError("Нужно задать только одну сторону в мм: ширину ИЛИ высоту.")


def _load_rgba_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise PipelineError(f"Файл не найден: {path}")

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PipelineError("Не удалось прочитать изображение. Поддерживаются jpg/png/bmp.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise PipelineError("Неподдерживаемый формат изображения.")

    return image


def _resolve_mm_size(
    width_px: int,
    height_px: int,
    width_mm: float | None,
    height_mm: float | None,
) -> tuple[float, float]:
    if width_mm is not None:
        if width_mm <= 0:
            raise PipelineError("Ширина в мм должна быть > 0.")
        ratio = height_px / width_px
        return float(width_mm), float(width_mm * ratio)

    if height_mm is not None:
        if height_mm <= 0:
            raise PipelineError("Высота в мм должна быть > 0.")
        ratio = width_px / height_px
        return float(height_mm * ratio), float(height_mm)

    raise PipelineError("Нужно задать ширину или высоту в мм.")


def _classify_color_masks(rgba: np.ndarray) -> dict[str, np.ndarray]:
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3].astype(np.uint8)

    # 1. Двусторонняя фильтрация: убирает шум сжатия и бледные ареолы, 
    # при этом сохраняя края дорожек резкими и четкими
    rgb = cv2.bilateralFilter(rgb, d=7, sigmaColor=50, sigmaSpace=50)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 2. Усиливаем порог прозрачности (отсекаем полупрозрачный мусор на краях)
    transparent = alpha < 128 
    
    # 3. Расширяем понятие "фона" (отсекаем бледные артефакты, близкие к белому)
    white_like = (rgb[:, :, 0] > 190) & (rgb[:, :, 1] > 190) & (rgb[:, :, 2] > 190)
    neutral_light_bg = (s < 45) & (v > 130)
    
    ignored = transparent | white_like | neutral_light_bg

    red = ((((h <= 12) | (h >= 170)) & (s >= 60) & (v >= 40)) & (~ignored))
    green = (((h >= 35) & (h <= 95) & (s >= 60) & (v >= 40)) & (~ignored))
    blue = (((h >= 90) & (h <= 145) & (s >= 60) & (v >= 40)) & (~ignored))
    black = ((v <= 105) & (~ignored))

    assigned = red | green | blue | black
    fallback = (~ignored) & (~assigned)
    black = black | (fallback & (s < 55))

    return {
        "blue": blue,
        "green": green,
        "red": red,
        "black": black,
    }


def _segment_color_masks(
    rgba: np.ndarray,
    raw_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    masks = raw_masks if raw_masks is not None else _classify_color_masks(rgba)
    cleaned: dict[str, np.ndarray] = {}
    area = rgba.shape[0] * rgba.shape[1]
    for name, mask in masks.items():
        if name == "black":
            min_area = max(4, area // 35000)
            do_close = True
            do_open = False
        elif name == "blue":
            min_area = max(4, area // 45000)
            do_close = False
            do_open = True
        else:
            min_area = max(8, area // 22000)
            do_close = False
            do_open = True
        cleaned[name] = _clean_mask(mask, min_area=min_area, do_close=do_close, do_open=do_open)
    return cleaned


def _clean_mask(mask: np.ndarray, min_area: int, do_close: bool, do_open: bool = True) -> np.ndarray:
    src = (mask.astype(np.uint8) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if do_open:
        src = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    if do_close:
        src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    keep = np.zeros_like(src)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == idx] = 255
    return keep > 0


def _validate_color_overlaps(masks: dict[str, np.ndarray]) -> None:
    red = masks["red"]
    green = masks["green"]
    black = masks["black"]

    overlap_rg = red & green
    overlap_rb = red & black
    overlap_gb = green & black

    if np.any(overlap_rg) or np.any(overlap_rb) or np.any(overlap_gb):
        raise PipelineError("Обнаружено пересечение red/green/black. Экспорт заблокирован.")


def _detect_hole_centers(blue_mask: np.ndarray, drill_radius_px: float) -> list[tuple[float, float]]:
    src = (blue_mask.astype(np.uint8) * 255).astype(np.uint8)
    if not np.any(src):
        return []

    blurred = cv2.medianBlur(src, 5)
    min_dist = max(4.0, drill_radius_px * 1.1)
    min_radius = max(1, int(round(drill_radius_px * 0.2)))
    max_radius = max(min_radius + 1, int(round(drill_radius_px * 3.2)))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=60,
        param2=6,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    hough_centers = [(float(c[0]), float(c[1])) for c in circles[0]] if circles is not None and circles.size > 0 else []
    hough_centers = _dedupe_points(hough_centers, max(2.0, min_dist * 0.65))

    peak_centers = _distance_peak_centers(src, drill_radius_px)
    candidates = _dedupe_points(hough_centers + peak_centers, max(2.0, drill_radius_px * 0.55))

    labels_count, labels, _, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    final_centers: list[tuple[float, float]] = []
    for idx in range(1, labels_count):
        in_component: list[tuple[float, float]] = []
        for x, y in candidates:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < labels.shape[0] and 0 <= xi < labels.shape[1] and labels[yi, xi] == idx:
                in_component.append((x, y))
        if in_component:
            final_centers.extend(_dedupe_points(in_component, max(2.0, drill_radius_px * 0.6)))
        else:
            cx, cy = centroids[idx]
            final_centers.append((float(cx), float(cy)))

    return _dedupe_points(final_centers, max(2.0, drill_radius_px * 0.6))


def _distance_peak_centers(mask_u8: np.ndarray, drill_radius_px: float) -> list[tuple[float, float]]:
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.size == 0:
        return []
    threshold = max(1.0, drill_radius_px * 0.24)
    dilated = cv2.dilate(dist, np.ones((3, 3), dtype=np.uint8))
    peaks = (dist >= threshold) & (dist >= dilated - 1e-4)
    ys, xs = np.where(peaks)
    points = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
    return _dedupe_points(points, max(2.0, drill_radius_px * 0.55))


def _component_centers(mask_u8: np.ndarray) -> list[tuple[float, float]]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    points: list[tuple[float, float]] = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        cx, cy = centroids[idx]
        points.append((float(cx), float(cy)))
    return points


def _dedupe_points(points: list[tuple[float, float]], min_dist: float) -> list[tuple[float, float]]:
    if not points:
        return []
    accepted: list[tuple[float, float]] = []
    min_dist_sq = min_dist * min_dist
    for x, y in points:
        if all((x - ax) ** 2 + (y - ay) ** 2 >= min_dist_sq for ax, ay in accepted):
            accepted.append((x, y))
    return accepted


def _validate_drill_collision_mask(holes: list[HoleCircle], black_mask: np.ndarray) -> None:
    if not holes or not np.any(black_mask):
        return
    black_u8 = (black_mask.astype(np.uint8) * 255).astype(np.uint8)
    for idx, hole in enumerate(holes, start=1):
        disk = np.zeros_like(black_u8)
        center = (int(round(hole.x_px)), int(round(hole.y_px)))
        radius = max(1, int(round(hole.radius_px)))
        cv2.circle(disk, center, radius, 255, thickness=-1)
        overlap = cv2.bitwise_and(black_u8, disk)
        if np.any(overlap):
            raise PipelineError(f"Drill touches black track at hole #{idx}.")


def _validate_drill_collision(holes: list[HoleCircle], black_geometry: Polygon | MultiPolygon | GeometryCollection) -> None:
    if black_geometry.is_empty or not holes:
        return
    for idx, hole in enumerate(holes, start=1):
        area = Point(hole.x_px, hole.y_px).buffer(hole.radius_px, quad_segs=16)
        if area.intersects(black_geometry):
            raise PipelineError(f"Сверло задевает черную дорожку у отверстия #{idx}.")


def _green_paths(
    green_mask: np.ndarray,
    hole_exclusion: np.ndarray,
    tool_radius_px: float,
    stepover_px: float,
    mode: GreenMode,
    simplify_tolerance_px: float,
) -> list[list[tuple[float, float]]]:
    work_mask = green_mask & (~hole_exclusion)
    center_mask = _erode_for_tool_center(work_mask, tool_radius_px)
    if not np.any(center_mask):
        return []

    if mode == "scanline":
        paths = _scanline_fill(center_mask, stepover_px)
    else:
        geom = _mask_to_geometry(center_mask)
        paths = _contour_offset_fill(geom, stepover_px)

    return _simplify_paths(paths, simplify_tolerance_px)


def _black_paths(
    black_geometry: Polygon | MultiPolygon | GeometryCollection,
    holes: list[HoleCircle],
    tool_radius_px: float,
    simplify_tolerance_px: float,
) -> list[list[tuple[float, float]]]:
    if black_geometry.is_empty:
        return []
    offset = black_geometry.buffer(tool_radius_px, quad_segs=16)
    hole_points = [Point(h.x_px, h.y_px) for h in holes]
    paths = _polygon_boundary_paths_without_hole_interiors(offset, hole_points)
    return _simplify_paths(paths, simplify_tolerance_px)


def _cut_paths(red_mask: np.ndarray, simplify_tolerance_px: float) -> list[list[tuple[float, float]]]:
    if not np.any(red_mask):
        return []
    skel = skeletonize(red_mask)
    polylines = _skeleton_to_polylines(skel)
    return _simplify_paths(polylines, simplify_tolerance_px)


def _erode_for_tool_center(mask: np.ndarray, radius_px: float) -> np.ndarray:
    r = max(1, int(np.ceil(radius_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    src = (mask.astype(np.uint8) * 255).astype(np.uint8)
    eroded = cv2.erode(src, kernel)
    return eroded > 0


def _scanline_fill(mask: np.ndarray, stepover_px: float) -> list[list[tuple[float, float]]]:
    paths: list[list[tuple[float, float]]] = []
    for component in _connected_component_masks(mask):
        paths.extend(_scanline_fill_component(component, stepover_px))
    return paths


def _scanline_fill_component(mask: np.ndarray, stepover_px: float) -> list[list[tuple[float, float]]]:
    step = max(1, int(round(stepover_px)))
    h, w = mask.shape
    paths: list[list[tuple[float, float]]] = []
    current_path: list[tuple[float, float]] = []
    reverse = False
    for y in range(0, h, step):
        row = mask[y]
        x = 0
        segments: list[tuple[float, float]] = []
        while x < w:
            while x < w and not row[x]:
                x += 1
            start = x
            while x < w and row[x]:
                x += 1
            end = x - 1
            if end > start:
                segments.append((float(start), float(end)))
        if reverse:
            segments = list(reversed(segments))
        for idx, (start, end) in enumerate(segments):
            if reverse:
                seg_start = (end + 0.5, y + 0.5)
                seg_end = (start + 0.5, y + 0.5)
            else:
                seg_start = (start + 0.5, y + 0.5)
                seg_end = (end + 0.5, y + 0.5)

            if not current_path:
                current_path.extend([seg_start, seg_end])
                continue

            last = current_path[-1]
            if idx == 0:
                # Prefer vertical connector to keep one continuous snake per component.
                if abs(last[1] - seg_start[1]) <= step + 0.51:
                    current_path.append((last[0], seg_start[1]))
                    current_path.append(seg_start)
                else:
                    paths.append(current_path)
                    current_path = [seg_start]
            else:
                current_path.append(seg_start)
            current_path.append(seg_end)
        reverse = not reverse

    if current_path:
        paths.append(current_path)
    return paths


def _connected_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    src = (mask.astype(np.uint8) * 255).astype(np.uint8)
    labels_count, labels = cv2.connectedComponents(src, connectivity=8)
    parts: list[np.ndarray] = []
    for idx in range(1, labels_count):
        component = labels == idx
        if np.any(component):
            parts.append(component)
    return parts


def _contour_offset_fill(
    geometry: Polygon | MultiPolygon | GeometryCollection,
    stepover_px: float,
) -> list[list[tuple[float, float]]]:
    if geometry.is_empty:
        return []

    step = max(0.75, stepover_px)
    paths: list[list[tuple[float, float]]] = []
    current = geometry
    max_loops = 2000
    loops = 0

    while not current.is_empty and loops < max_loops:
        loops += 1
        boundary_paths = _polygon_boundary_paths(current)
        if not boundary_paths:
            break
        paths.extend(boundary_paths)
        current = current.buffer(-step, quad_segs=8)

    return paths


def _mask_to_geometry(mask: np.ndarray) -> Polygon | MultiPolygon | GeometryCollection:
    src = (mask.astype(np.uint8) * 255).astype(np.uint8)
    found = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(found) == 3:
        _, contours, hierarchy = found
    else:
        contours, hierarchy = found

    if hierarchy is None or not contours:
        return GeometryCollection()

    hierarchy = hierarchy[0]
    polygons: list[Polygon] = []
    for idx, h in enumerate(hierarchy):
        parent = int(h[3])
        if parent != -1:
            continue

        shell = _contour_to_ring(contours[idx])
        if len(shell) < 3:
            continue
        holes: list[list[tuple[float, float]]] = []

        child = int(h[2])
        while child != -1:
            ring = _contour_to_ring(contours[child])
            if len(ring) >= 3:
                holes.append(ring)
            child = int(hierarchy[child][0])

        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if isinstance(poly, Polygon):
            polygons.append(poly)
        elif isinstance(poly, MultiPolygon):
            polygons.extend(list(poly.geoms))

    if not polygons:
        return GeometryCollection()

    merged = unary_union(polygons)
    if not merged.is_valid:
        merged = merged.buffer(0)
    return merged


def _contour_to_ring(contour: np.ndarray) -> list[tuple[float, float]]:
    pts = [(float(p[0][0]), float(p[0][1])) for p in contour]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _polygon_boundary_paths(geometry: Polygon | MultiPolygon | GeometryCollection) -> list[list[tuple[float, float]]]:
    paths: list[list[tuple[float, float]]] = []
    for poly in _iter_polygons(geometry):
        ext = list(poly.exterior.coords)
        if len(ext) >= 2:
            paths.append([(float(x), float(y)) for x, y in ext])
        for interior in poly.interiors:
            ring = list(interior.coords)
            if len(ring) >= 2:
                paths.append([(float(x), float(y)) for x, y in ring])
    return paths


def _polygon_boundary_paths_without_hole_interiors(
    geometry: Polygon | MultiPolygon | GeometryCollection,
    hole_points: list[Point],
) -> list[list[tuple[float, float]]]:
    if not hole_points:
        return _polygon_boundary_paths(geometry)

    paths: list[list[tuple[float, float]]] = []
    for poly in _iter_polygons(geometry):
        ext = list(poly.exterior.coords)
        if len(ext) >= 2:
            paths.append([(float(x), float(y)) for x, y in ext])

        for interior in poly.interiors:
            ring = list(interior.coords)
            if len(ring) < 2:
                continue
            interior_poly = Polygon(ring)
            if interior_poly.is_valid and any(interior_poly.buffer(1.0).contains(pt) for pt in hole_points):
                continue
            paths.append([(float(x), float(y)) for x, y in ring])
    return paths


def _iter_polygons(geometry: Polygon | MultiPolygon | GeometryCollection):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            yield poly
    elif isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, Polygon):
                yield geom
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    yield poly


def _skeleton_to_polylines(skeleton: np.ndarray) -> list[list[tuple[float, float]]]:
    ys, xs = np.where(skeleton)
    points = {(int(x), int(y)) for x, y in zip(xs, ys, strict=True)}
    if not points:
        return []

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    for x, y in points:
        nbs = []
        for dx, dy in offsets:
            cand = (x + dx, y + dy)
            if cand in points:
                nbs.append(cand)
        neighbors[(x, y)] = nbs

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    polylines: list[list[tuple[int, int]]] = []

    junctions = [p for p, nbs in neighbors.items() if len(nbs) != 2]
    for start in junctions:
        for nxt in neighbors[start]:
            ek = edge_key(start, nxt)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)
            line = [start]
            prev = start
            curr = nxt
            while True:
                line.append(curr)
                curr_nbs = neighbors[curr]
                if len(curr_nbs) != 2:
                    break
                candidates = [p for p in curr_nbs if p != prev]
                if not candidates:
                    break
                nxt2 = candidates[0]
                ek2 = edge_key(curr, nxt2)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                prev, curr = curr, nxt2
            polylines.append(line)

    for start in points:
        for nxt in neighbors[start]:
            ek = edge_key(start, nxt)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)
            line = [start]
            prev = start
            curr = nxt
            guard = 0
            while guard < len(points) * 3:
                guard += 1
                line.append(curr)
                candidates = [p for p in neighbors[curr] if p != prev]
                if not candidates:
                    break
                nxt2 = candidates[0]
                ek2 = edge_key(curr, nxt2)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                prev, curr = curr, nxt2
            polylines.append(line)

    result: list[list[tuple[float, float]]] = []
    for line in polylines:
        if len(line) < 2:
            continue
        result.append([(x + 0.5, y + 0.5) for x, y in line])
    return result


def _simplify_paths(
    paths: list[list[tuple[float, float]]],
    tolerance_px: float,
) -> list[list[tuple[float, float]]]:
    cleaned: list[list[tuple[float, float]]] = []
    for pts in paths:
        if len(pts) < 2:
            continue
        line = LineString(pts)
        if tolerance_px > 0:
            line = line.simplify(tolerance_px, preserve_topology=False)
        if line.is_empty:
            continue

        if line.geom_type == "LineString":
            coords = [(float(x), float(y)) for x, y in line.coords]
            if len(coords) >= 2:
                cleaned.append(coords)
        elif line.geom_type == "MultiLineString":
            for part in line.geoms:
                coords = [(float(x), float(y)) for x, y in part.coords]
                if len(coords) >= 2:
                    cleaned.append(coords)
    return cleaned


def _hole_exclusion_mask(shape: tuple[int, int], holes: list[HoleCircle]) -> np.ndarray:
    if not holes:
        return np.zeros(shape, dtype=bool)
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for hole in holes:
        center = (int(round(hole.x_px)), int(round(hole.y_px)))
        radius = max(1, int(round(hole.radius_px)))
        cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask > 0


def _hole_pad_exclusion_mask(black_raw_mask: np.ndarray, holes: list[HoleCircle]) -> np.ndarray:
    if not holes:
        return np.zeros_like(black_raw_mask, dtype=bool)

    h, w = black_raw_mask.shape
    mask_u8 = (black_raw_mask.astype(np.uint8) * 255).astype(np.uint8)
    exclusion = np.zeros((h, w), dtype=np.uint8)

    for hole in holes:
        cx = float(hole.x_px)
        cy = float(hole.y_px)
        default_radius = 2.5
        search_radius = max(hole.radius_px * 4.5, 12.0)

        x0 = max(0, int(np.floor(cx - search_radius)))
        x1 = min(w - 1, int(np.ceil(cx + search_radius)))
        y0 = max(0, int(np.floor(cy - search_radius)))
        y1 = min(h - 1, int(np.ceil(cy + search_radius)))

        roi = mask_u8[y0 : y1 + 1, x0 : x1 + 1]
        ys, xs = np.where(roi > 0)

        exclusion_radius = default_radius
        if xs.size > 10:
            gx = xs.astype(np.float32) + float(x0)
            gy = ys.astype(np.float32) + float(y0)
            d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
            nearby = d[d <= search_radius]
            if nearby.size > 24:
                dx = gx[d <= search_radius] - cx
                dy = gy[d <= search_radius] - cy
                angles = np.arctan2(dy, dx)
                bins = 36
                hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
                coverage = float(np.count_nonzero(hist) / bins)

                r10 = float(np.percentile(nearby, 10))
                r90 = float(np.percentile(nearby, 90))
                ring_like = coverage >= 0.55 and (r90 - r10) <= (search_radius * 0.65)
                if ring_like:
                    exclusion_radius = max(default_radius, min(r90 + 1.5, search_radius))

        cv2.circle(
            exclusion,
            (int(round(cx)), int(round(cy))),
            max(1, int(round(exclusion_radius))),
            255,
            thickness=-1,
        )

    return exclusion > 0

def _merge_hole_clusters(holes: list[HoleCircle]) -> list[HoleCircle]:
    if not holes:
        return []

    n = len(holes)
    adj = {i: [] for i in range(n)}
    
    # Строим граф: соединяем отверстия, если они касаются друг друга
    for i in range(n):
        for j in range(i + 1, n):
            dx = holes[i].x_px - holes[j].x_px
            dy = holes[i].y_px - holes[j].y_px
            dist_sq = dx * dx + dy * dy
            
            # Порог касания: сумма радиусов + 15% запаса на погрешности пикселей
            threshold = (holes[i].radius_px + holes[j].radius_px) 
            
            if dist_sq <= threshold * threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    merged_holes = []

    # Обходим граф и собираем кластеры
    for i in range(n):
        if i not in visited:
            cluster = []
            q = [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)

            # Вычисляем центр масс кластера
            avg_x = sum(holes[idx].x_px for idx in cluster) / len(cluster)
            avg_y = sum(holes[idx].y_px for idx in cluster) / len(cluster)
            radius = holes[cluster[0]].radius_px  # Радиус у всех одинаковый

            merged_holes.append(HoleCircle(avg_x, avg_y, radius))

    return merged_holes