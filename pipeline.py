# pipeline.py
from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal
import math
import cv2
import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize

logger = logging.getLogger("pipeline")

GreenMode = Literal["scanline", "contour-offset", "centerline"]

class PipelineError(ValueError):
    """Domain error for invalid input or impossible toolpaths."""

@dataclass(frozen=True)
class PipelineConfig:
    image_path: Path
    width_mm: float | None
    height_mm: float | None
    tool_angle_deg: float          
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
    black_paths: list[list[tuple[float, float]]]
    cut_paths: list[list[tuple[float, float]]]
    green_tool_radius_px: float
    black_tool_radius_px: float
    cut_tool_radius_px: float
    warnings: list[str]

def _calc_effective_radius_mm(depth_mm: float, angle_deg: float) -> float:
    """Эффективный радиус V-фрезы: r = depth * tan(angle/2)"""
    if depth_mm <= 0 or angle_deg <= 0 or angle_deg >= 180:
        return 0.0
    return depth_mm * math.tan(math.radians(angle_deg / 2.0))


# --- Вспомогательные функции топологического анализа ---

def _rasterize_vector_paths(paths: list[list[tuple[float, float]]], width: int, height: int, thickness_px: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for path in paths:
        if len(path) < 2:
            continue
        pts = np.array([[int(round(x)), int(round(y))] for x, y in path], np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness_px, lineType=cv2.LINE_AA)
    return mask > 0

def _get_components(mask: np.ndarray) -> dict[int, np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask.astype(np.uint8) * 255).astype(np.uint8), connectivity=8
    )
    comps = {}
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] > 5:
            comps[idx] = labels == idx
    return comps

def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
    h, w = padded.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    inv_img = cv2.bitwise_not((padded.astype(np.uint8) * 255))
    cv2.floodFill(inv_img, flood_mask, (0, 0), 255)
    holes = flood_mask[1:-1, 1:-1] == 0
    return np.logical_or(padded, holes)[1:-1, 1:-1]

def _analyze_topology_internal(gt_comps: dict[int, np.ndarray], vec_comps: dict[int, np.ndarray], min_overlap_ratio: float = 0.05):
    breaks = []
    shorts = []
    vec_to_gt = {v_id: [] for v_id in vec_comps}
    gt_to_vec = {g_id: [] for g_id in gt_comps}
    for v_id, v_mask in vec_comps.items():
        for g_id, g_mask in gt_comps.items():
            intersection = np.logical_and(v_mask, g_mask).sum()
            gt_area = g_mask.sum()
            if gt_area > 0 and intersection > min_overlap_ratio * gt_area:
                vec_to_gt[v_id].append(g_id)
                gt_to_vec[g_id].append(v_id)
    for g_id, v_ids in gt_to_vec.items():
        if len(v_ids) == 0:
            breaks.append(f"Дорожка #{g_id} полностью потеряна.")
        elif len(v_ids) > 1:
            breaks.append(f"Дорожка #{g_id} разорвана на {len(v_ids)} частей.")
    for v_id, g_ids in vec_to_gt.items():
        if len(g_ids) > 1:
            shorts.append(f"Замыкание: непреднамеренное объединение дорожек {g_ids}.")
    return breaks, shorts

def _validate_topology_and_collisions(
    gt_masks: dict[str, np.ndarray],
    black_paths: list[list[tuple[float, float]]],
    cut_paths: list[list[tuple[float, float]]],
    width_px: int,
    height_px: int,
    black_radius_px: float,
    cut_radius_px: float
) -> None:
    logger.info("Executing topological consistency checks...")
    errors = []

    # 1. Валидация дорожек (BLACK)
    gt_black = gt_masks.get("black")
    if gt_black is not None and gt_black.sum() > 0:
        thickness_px = max(1, int(round(black_radius_px * 2)))
        vec_black_mask = _rasterize_vector_paths(black_paths, width_px, height_px, thickness_px)
        gt_comps = _get_components(gt_black)
        vec_comps = _get_components(vec_black_mask)
        breaks, shorts = _analyze_topology_internal(gt_comps, vec_comps)
        errors.extend(breaks)
        errors.extend(shorts)

    # 2. Валидация контура реза (RED)
    gt_red = gt_masks.get("red")
    if gt_red is not None and gt_red.sum() > 0:
        thickness_px = max(1, int(round(cut_radius_px * 2)))
        vec_red_mask = _rasterize_vector_paths(cut_paths, width_px, height_px, thickness_px)
        gt_comps = _get_components(gt_red)
        vec_red_filled = _fill_internal_holes(vec_red_mask)
        vec_comps = _get_components(vec_red_filled)
        breaks, _ = _analyze_topology_internal(gt_comps, vec_comps)
        if breaks:
            errors.extend([f"Ошибка контура платы: {b}" for b in breaks])

    # 3. Валидация пересечений контура (RED) и дорожек (BLACK)
    if gt_black is not None and gt_black.sum() > 0:
        thickness_px = max(1, int(round(cut_radius_px * 2)))
        vec_red_mask = _rasterize_vector_paths(cut_paths, width_px, height_px, thickness_px)
        overlap = np.logical_and(gt_black, vec_red_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (overlap.astype(np.uint8) * 255), connectivity=8
        )
        collisions = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 3:
                collisions += 1
        if collisions > 0:
            errors.append(f"Коллизия: обнаружено {collisions} пересечений контурной фрезы с дорожками.")

    if errors:
        logger.error(f"Topological consistency checks failed with {len(errors)} issues.")
        raise PipelineError("Обнаружены критические дефекты трассировки:\n\n" + "\n".join(errors))
    
    logger.info("Topological verification passed successfully.")


# --- Основной конвейер ---

def run_pipeline(config: PipelineConfig) -> PipelineResult:
    logger.info("Pipeline thread started.")
    _validate_config(config)
    rgba = _load_rgba_image(config.image_path)
    height_px, width_px = rgba.shape[:2]
    width_mm, height_mm = _resolve_mm_size(width_px, height_px, config.width_mm, config.height_mm)
    px_per_mm = width_px / width_mm

    logger.info(f"Loaded image size: {width_px}x{height_px}px ({width_mm:.2f}x{height_mm:.2f}mm) - {px_per_mm:.2f} px/mm")

    raw_masks = _classify_color_masks(rgba)
    masks = _segment_color_masks(rgba, raw_masks=raw_masks)
    _validate_color_overlaps(masks)

    drill_radius_px = (config.drill_diameter_mm * px_per_mm) / 2.0
    green_r_mm = _calc_effective_radius_mm(config.green_depth_mm, config.tool_angle_deg)
    black_r_mm = _calc_effective_radius_mm(config.trace_depth_mm, config.tool_angle_deg)
    cut_r_mm = _calc_effective_radius_mm(config.stock_thickness_mm, config.tool_angle_deg)

    green_tool_radius_px = green_r_mm * px_per_mm
    black_tool_radius_px = black_r_mm * px_per_mm
    cut_tool_radius_px = cut_r_mm * px_per_mm

    logger.debug(
        f"Calculated px radii: drill={drill_radius_px:.2f}, "
        f"green_tool={green_tool_radius_px:.2f}, "
        f"black_tool={black_tool_radius_px:.2f}, "
        f"cut_tool={cut_tool_radius_px:.2f}"
    )

    green_stepover_px = max(1.0, (config.stepover_percent / 100.0) * 2.0 * green_tool_radius_px)
    simplify_tolerance_px = max(0.0, config.simplify_tolerance_mm * px_per_mm)

    logger.info("Detecting drill holes centers...")
    hole_centers = _detect_hole_centers(masks["blue"], drill_radius_px)
    holes = [HoleCircle(x, y, drill_radius_px) for x, y in hole_centers]
    holes = _merge_hole_clusters(holes)
    logger.info(f"Deduplicated and merged drill holes count: {len(holes)}")

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

    logger.info(f"Generating green layer (engraving) with mode: '{config.green_mode}'...")
    green_paths = _green_paths(
        masks["green"],
        hole_exclusion=hole_exclusion,
        tool_radius_px=green_tool_radius_px,
        stepover_px=green_stepover_px,
        mode=config.green_mode,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    logger.info(f"Green layer: {len(green_paths)} vector path segments generated.")

    logger.info("Generating copper trace isolation paths (black)...")
    black_paths = _black_paths(
        black_geometry,
        holes=holes,
        tool_radius_px=black_tool_radius_px,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    logger.info(f"Black layer: {len(black_paths)} vector path segments generated.")

    logger.info("Generating board outline paths (red)...")
    cut_paths = _cut_paths(
        masks["red"],
        simplify_tolerance_px=simplify_tolerance_px,
    )
    logger.info(f"Red layer: {len(cut_paths)} vector path segments generated.")

    warnings_list = []
    if not holes:
        warnings_list.append("Слой отверстий (Blue) не обнаружен.")
    if not green_paths:
        warnings_list.append("Слой текста/шелкографии (Green) не обнаружен.")
    if not black_paths:
        warnings_list.append("Слой проводящих дорожек (Black) не обнаружен.")
    if not cut_paths:
        warnings_list.append("Слой контура обрезки (Red) не обнаружен.")
    
    if not (holes or green_paths or black_paths or cut_paths):
        raise PipelineError("Не найдено ни одного рабочего слоя (отверстия, текст, дорожки или контур).")

    # Топологическая валидация
    _validate_topology_and_collisions(
        gt_masks=masks,
        black_paths=black_paths,
        cut_paths=cut_paths,
        width_px=width_px,
        height_px=height_px,
        black_radius_px=black_tool_radius_px,
        cut_radius_px=cut_tool_radius_px
    )
    
    logger.info("Pipeline processing completed successfully.")
    return PipelineResult(
        width_px=width_px,
        height_px=height_px,
        width_mm=width_mm,
        height_mm=height_mm,
        holes=holes,
        green_paths=green_paths,
        black_paths=black_paths,
        cut_paths=cut_paths,
        green_tool_radius_px=green_tool_radius_px,
        black_tool_radius_px=black_tool_radius_px,
        cut_tool_radius_px=cut_tool_radius_px,
        warnings=warnings_list,
    )

def _validate_config(config: PipelineConfig) -> None:
    if config.green_mode not in {"scanline", "contour-offset", "centerline"}:
        raise PipelineError("Режим green должен быть scanline, contour-offset или centerline.")
        
    numeric_checks = {
        "Угол фрезы": config.tool_angle_deg,
        "Диаметр сверла": config.drill_diameter_mm,
        "Stepover %": config.stepover_percent,
        "Толщина заготовки": config.stock_thickness_mm,
        "Скорость реза": config.cut_speed_mm_s,
        "Макс. ускорение": config.max_accel_mm_s2,
        "Глубина дорожек": config.trace_depth_mm,
        "Глубина green": config.green_depth_mm,
    }
    for name, value in numeric_checks.items():
        if value <= 0: raise PipelineError(f"{name} должно быть > 0.")
    if not (0 < config.tool_angle_deg < 180): raise PipelineError("Угол фрезы должен быть в диапазоне (0, 180) градусов.")
    if config.simplify_tolerance_mm < 0: raise PipelineError("Допуск упрощения не может быть отрицательным.")
    if not (1 <= config.stepover_percent <= 100): raise PipelineError("Stepover должен быть в диапазоне 1..100%.")
    if (config.width_mm is None) == (config.height_mm is None): raise PipelineError("Нужно задать только одну сторону в мм: ширину ИЛИ высоту.")

def _load_rgba_image(path: Path) -> np.ndarray:
    if not path.exists(): raise PipelineError(f"Файл не найден: {path}")
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None: raise PipelineError("Не удалось прочитать изображение. Поддерживаются jpg/png/bmp.")
    if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] == 3: return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.ndim == 3 and image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else: raise PipelineError("Неподдерживаемый формат изображения.")

def _resolve_mm_size(width_px: int, height_px: int, width_mm: float | None, height_mm: float | None) -> tuple[float, float]:
    if width_mm is not None:
        if width_mm <= 0: raise PipelineError("Ширина в мм должна быть > 0.")
        return float(width_mm), float(width_mm * (height_px / width_px))
    if height_mm is not None:
        if height_mm <= 0: raise PipelineError("Высота в мм должна быть > 0.")
        return float(height_mm * (width_px / height_px)), float(height_mm)
    raise PipelineError("Нужно задать ширину или высоту в мм.")

def _classify_color_masks(rgba: np.ndarray) -> dict[str, np.ndarray]:
    if rgba.ndim == 3 and rgba.shape[2] == 3: rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)])
    rgb, alpha = rgba[:, :, :3].astype(np.uint8), rgba[:, :, 3].astype(np.uint8)
    rgb = cv2.bilateralFilter(rgb, d=7, sigmaColor=50, sigmaSpace=50)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    transparent, white_like = alpha < 128, (rgb[:, :, 0] > 190) & (rgb[:, :, 1] > 190) & (rgb[:, :, 2] > 190)
    neutral_light_bg = (s < 45) & (v > 130)
    ignored = transparent | white_like | neutral_light_bg
    red = ((((h <= 12) | (h >= 170)) & (s >= 60) & (v >= 40)) & (~ignored))
    green = (((h >= 35) & (h <= 95) & (s >= 60) & (v >= 40)) & (~ignored))
    blue = (((h >= 90) & (h <= 145) & (s >= 60) & (v >= 40)) & (~ignored))
    black = ((v <= 105) & (~ignored))
    assigned = red | green | blue | black
    black = black | ((~ignored) & (~assigned) & (s < 55))
    return {"blue": blue, "green": green, "red": red, "black": black}

def _segment_color_masks(rgba: np.ndarray, raw_masks: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    masks, cleaned, area = raw_masks if raw_masks is not None else _classify_color_masks(rgba), {}, rgba.shape[0] * rgba.shape[1]
    for name, mask in masks.items():
        if name == "black": min_a, do_c, do_o = max(4, area // 35000), True, False
        elif name == "blue": min_a, do_c, do_o = max(4, area // 45000), False, True
        elif name == "green": min_a, do_c, do_o = max(4, area // 45000), True, False
        else: min_a, do_c, do_o = max(8, area // 22000), False, True
        cleaned[name] = _clean_mask(mask, min_a, do_c, do_o)
    return cleaned

def _clean_mask(mask: np.ndarray, min_area: int, do_close: bool, do_open: bool = True) -> np.ndarray:
    src, kernel = (mask.astype(np.uint8) * 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if do_open: src = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    if do_close: src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    keep = np.zeros_like(src)
    for idx in range(1, num_labels):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area: keep[labels == idx] = 255
    return keep > 0

def _validate_color_overlaps(masks: dict[str, np.ndarray]) -> None:
    if np.any(masks["red"] & masks["green"]) or np.any(masks["red"] & masks["black"]) or np.any(masks["green"] & masks["black"]):
        raise PipelineError("Обнаружено пересечение red/green/black. Экспорт заблокирован.")

def _detect_hole_centers(blue_mask: np.ndarray, drill_radius_px: float) -> list[tuple[float, float]]:
    src = (blue_mask.astype(np.uint8) * 255).astype(np.uint8)
    if not np.any(src): return []
    blurred, min_dist = cv2.medianBlur(src, 5), max(4.0, drill_radius_px * 1.1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist, param1=60, param2=6, minRadius=max(1, int(round(drill_radius_px * 0.2))), maxRadius=max(2, int(round(drill_radius_px * 3.2))))
    candidates = _dedupe_points(([(float(c[0]), float(c[1])) for c in circles[0]] if circles is not None else []) + _distance_peak_centers(src, drill_radius_px), max(2.0, drill_radius_px * 0.55))
    labels_count, labels, _, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    final_centers = []
    for idx in range(1, labels_count):
        in_comp = [(x, y) for x, y in candidates if 0 <= int(round(y)) < labels.shape[0] and 0 <= int(round(x)) < labels.shape[1] and labels[int(round(y)), int(round(x))] == idx]
        if in_comp: final_centers.extend(_dedupe_points(in_comp, max(2.0, drill_radius_px * 0.6)))
        else: final_centers.append((float(centroids[idx][0]), float(centroids[idx][1])))
    return _dedupe_points(final_centers, max(2.0, drill_radius_px * 0.6))

def _distance_peak_centers(mask_u8: np.ndarray, drill_radius_px: float) -> list[tuple[float, float]]:
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.size == 0: return []
    peaks = (dist >= max(1.0, drill_radius_px * 0.24)) & (dist >= cv2.dilate(dist, np.ones((3, 3), dtype=np.uint8)) - 1e-4)
    ys, xs = np.where(peaks)
    return _dedupe_points([(float(x), float(y)) for x, y in zip(xs, ys, strict=True)], max(2.0, drill_radius_px * 0.55))

def _dedupe_points(points: list[tuple[float, float]], min_dist: float) -> list[tuple[float, float]]:
    if not points: return []
    accepted, min_dist_sq = [], min_dist * min_dist
    for x, y in points:
        if all((x - ax) ** 2 + (y - ay) ** 2 >= min_dist_sq for ax, ay in accepted): accepted.append((x, y))
    return accepted

def _validate_drill_collision_mask(holes: list[HoleCircle], black_mask: np.ndarray) -> None:
    if not holes or not np.any(black_mask): return
    black_u8 = (black_mask.astype(np.uint8) * 255).astype(np.uint8)
    for idx, hole in enumerate(holes, start=1):
        disk = np.zeros_like(black_u8)
        cv2.circle(disk, (int(round(hole.x_px)), int(round(hole.y_px))), max(1, int(round(hole.radius_px))), 255, thickness=-1)
        if np.any(cv2.bitwise_and(black_u8, disk)): raise PipelineError(f"Drill touches black track at hole #{idx}.")

def _green_paths(
    green_mask: np.ndarray, hole_exclusion: np.ndarray, tool_radius_px: float, stepover_px: float, mode: GreenMode, simplify_tolerance_px: float
) -> list[list[tuple[float, float]]]:
    work_mask = (cv2.morphologyEx(((green_mask & (~hole_exclusion)).astype(np.uint8) * 255), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))) > 0
    if not np.any(work_mask): return []
    if mode == "centerline": return _simplify_paths(_skeleton_to_polylines(skeletonize(work_mask)), simplify_tolerance_px)
    center_mask = _erode_for_tool_center(work_mask, tool_radius_px)
    if not np.any(center_mask): center_mask = work_mask
    if mode == "scanline": paths = _scanline_fill(center_mask, stepover_px)
    else:
        geom = _mask_to_geometry(center_mask)
        paths = _contour_offset_fill(geom, stepover_px) if not geom.is_empty else []
        if not paths and not geom.is_empty: paths = _polygon_boundary_paths(geom)
    return _simplify_paths(paths, simplify_tolerance_px)

def _black_paths(
    black_geometry: Polygon | MultiPolygon | GeometryCollection, holes: list[HoleCircle], tool_radius_px: float, simplify_tolerance_px: float
) -> list[list[tuple[float, float]]]:
    if black_geometry.is_empty: return []
    offset = black_geometry.buffer(tool_radius_px, quad_segs=16)
    return _simplify_paths(_polygon_boundary_paths_without_hole_interiors(offset, [Point(h.x_px, h.y_px) for h in holes]), simplify_tolerance_px)

def _cut_paths(
    red_mask: np.ndarray, simplify_tolerance_px: float
) -> list[list[tuple[float, float]]]:
    if not np.any(red_mask):
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx((red_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel)

    h, w = closed.shape
    padded = np.pad(closed, pad_width=2, mode='constant', constant_values=0)
    flood_mask = np.zeros((h + 6, w + 6), dtype=np.uint8)
    
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    solid_board = cv2.bitwise_not(padded)[2:-2, 2:-2]

    geom = _mask_to_geometry(solid_board > 0)
    if geom.is_empty:
        return []

    paths: list[list[tuple[float, float]]] = []
    for poly in _iter_polygons(geom):
        ext = list(poly.exterior.coords)
        if len(ext) >= 2:
            paths.append([(float(x), float(y)) for x, y in ext])
            
    return _simplify_paths(paths, simplify_tolerance_px)

def _erode_for_tool_center(mask: np.ndarray, radius_px: float) -> np.ndarray:
    r = max(1, int(np.ceil(radius_px)))
    return cv2.erode((mask.astype(np.uint8) * 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0

def _scanline_fill(mask: np.ndarray, stepover_px: float) -> list[list[tuple[float, float]]]:
    paths: list[list[tuple[float, float]]] = []
    for component in _connected_component_masks(mask): paths.extend(_scanline_fill_component(component, stepover_px))
    return paths

def _scanline_fill_component(mask: np.ndarray, stepover_px: float) -> list[list[tuple[float, float]]]:
    step, h, w = max(1, int(round(stepover_px))), mask.shape[0], mask.shape[1]
    paths, current_path, reverse = [], [], False
    for y in range(0, h, step):
        row, x, segments = mask[y], 0, []
        while x < w:
            while x < w and not row[x]: x += 1
            start = x
            while x < w and row[x]: x += 1
            if (x - 1) > start: segments.append((float(start), float(x - 1)))
        if reverse: segments = list(reversed(segments))
        for idx, (start, end) in enumerate(segments):
            seg_start, seg_end = ((end + 0.5, y + 0.5), (start + 0.5, y + 0.5)) if reverse else ((start + 0.5, y + 0.5), (end + 0.5, y + 0.5))
            if not current_path: current_path.extend([seg_start, seg_end]); continue
            last = current_path[-1]
            if idx == 0:
                if abs(last[1] - seg_start[1]) <= step + 0.51: current_path.extend([(last[0], seg_start[1]), seg_start])
                else: paths.append(current_path); current_path = [seg_start]
            else: current_path.append(seg_start)
            current_path.append(seg_end)
        reverse = not reverse
    if current_path: paths.append(current_path)
    return paths

def _connected_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    labels_count, labels = cv2.connectedComponents((mask.astype(np.uint8) * 255).astype(np.uint8), connectivity=8)
    return [labels == idx for idx in range(1, labels_count) if np.any(labels == idx)]

def _contour_offset_fill(geometry: Polygon | MultiPolygon | GeometryCollection, stepover_px: float) -> list[list[tuple[float, float]]]:
    if geometry.is_empty: return []
    step, paths, current, loops = max(0.75, stepover_px), [], geometry, 0
    while not current.is_empty and loops < 2000:
        loops += 1
        boundary_paths = _polygon_boundary_paths(current)
        if not boundary_paths: break
        paths.extend(boundary_paths)
        current = current.buffer(-step, quad_segs=8)
    return paths

def _mask_to_geometry(mask: np.ndarray) -> Polygon | MultiPolygon | GeometryCollection:
    found = cv2.findContours((mask.astype(np.uint8) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = (found[1], found[2]) if len(found) == 3 else found
    if hierarchy is None or not contours: return GeometryCollection()
    polygons: list[Polygon] = []
    for idx, h in enumerate(hierarchy[0]):
        if int(h[3]) != -1: continue
        shell = _contour_to_ring(contours[idx])
        if len(shell) < 3: continue
        holes, child = [], int(h[2])
        while child != -1:
            ring = _contour_to_ring(contours[child])
            if len(ring) >= 3: holes.append(ring)
            child = int(hierarchy[0][child][0])
        poly = Polygon(shell, holes)
        if not poly.is_valid: poly = poly.buffer(0)
        if poly.is_empty: continue
        if isinstance(poly, Polygon): polygons.append(poly)
        elif isinstance(poly, MultiPolygon): polygons.extend(list(poly.geoms))
    if not polygons: return GeometryCollection()
    merged = unary_union(polygons)
    return merged.buffer(0) if not merged.is_valid else merged

def _contour_to_ring(contour: np.ndarray) -> list[tuple[float, float]]:
    pts = [(float(p[0][0]), float(p[0][1])) for p in contour]
    return pts[:-1] if len(pts) > 1 and pts[0] == pts[-1] else pts

def _polygon_boundary_paths(geometry: Polygon | MultiPolygon | GeometryCollection) -> list[list[tuple[float, float]]]:
    paths: list[list[tuple[float, float]]] = []
    for poly in _iter_polygons(geometry):
        ext = list(poly.exterior.coords)
        if len(ext) >= 2: paths.append([(float(x), float(y)) for x, y in ext])
        for interior in poly.interiors:
            ring = list(interior.coords)
            if len(ring) >= 2: paths.append([(float(x), float(y)) for x, y in ring])
    return paths

def _polygon_boundary_paths_without_hole_interiors(geometry: Polygon | MultiPolygon | GeometryCollection, hole_points: list[Point]) -> list[list[tuple[float, float]]]:
    if not hole_points: return _polygon_boundary_paths(geometry)
    paths: list[list[tuple[float, float]]] = []
    for poly in _iter_polygons(geometry):
        ext = list(poly.exterior.coords)
        if len(ext) >= 2: paths.append([(float(x), float(y)) for x, y in ext])
        for interior in poly.interiors:
            ring = list(interior.coords)
            if len(ring) < 2: continue
            interior_poly = Polygon(ring)
            if interior_poly.is_valid and any(interior_poly.buffer(1.0).contains(pt) for pt in hole_points): continue
            paths.append([(float(x), float(y)) for x, y in ring])
    return paths

def _iter_polygons(geometry: Polygon | MultiPolygon | GeometryCollection):
    if geometry.is_empty: return
    if isinstance(geometry, Polygon): yield geometry
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms: yield poly
    elif isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, Polygon): yield geom
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms: yield poly

def _skeleton_to_polylines(skeleton: np.ndarray) -> list[list[tuple[float, float]]]:
    ys, xs = np.where(skeleton)
    points = {(int(x), int(y)) for x, y in zip(xs, ys, strict=True)}
    if not points: return []
    neighbors = {p: [c for c in [(p[0]+dx, p[1]+dy) for dx, dy in ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))] if c in points] for p in points}
    visited_edges, polylines = set(), []
    for start in [p for p, nbs in neighbors.items() if len(nbs) != 2] + list(points):
        for nxt in neighbors[start]:
            ek = (start, nxt) if start <= nxt else (nxt, start)
            if ek in visited_edges: continue
            visited_edges.add(ek)
            line, prev, curr, guard = [start], start, nxt, 0
            while guard < len(points) * 3:
                guard += 1
                line.append(curr)
                candidates = [p for p in neighbors[curr] if p != prev]
                if not candidates or len(neighbors[curr]) != 2: break
                nxt2 = candidates[0]
                ek2 = (curr, nxt2) if curr <= nxt2 else (nxt2, curr)
                if ek2 in visited_edges: break
                visited_edges.add(ek2)
                prev, curr = curr, nxt2
            polylines.append(line)
    return [[(x + 0.5, y + 0.5) for x, y in line] for line in polylines if len(line) >= 2]

def _simplify_paths(paths: list[list[tuple[float, float]]], tolerance_px: float) -> list[list[tuple[float, float]]]:
    cleaned: list[list[tuple[float, float]]] = []
    for pts in paths:
        if len(pts) < 2: continue
        line = LineString(pts).simplify(tolerance_px, preserve_topology=False) if tolerance_px > 0 else LineString(pts)
        if line.is_empty: continue
        if line.geom_type == "LineString":
            if len(line.coords) >= 2: cleaned.append([(float(x), float(y)) for x, y in line.coords])
        elif line.geom_type == "MultiLineString":
            for part in line.geoms:
                if len(part.coords) >= 2: cleaned.append([(float(x), float(y)) for x, y in part.coords])
    return cleaned

def _hole_exclusion_mask(shape: tuple[int, int], holes: list[HoleCircle]) -> np.ndarray:
    if not holes: return np.zeros(shape, dtype=bool)
    mask = np.zeros(shape, dtype=np.uint8)
    for hole in holes: cv2.circle(mask, (int(round(hole.x_px)), int(round(hole.y_px))), max(1, int(round(hole.radius_px))), 255, thickness=-1)
    return mask > 0

def _hole_pad_exclusion_mask(black_raw_mask: np.ndarray, holes: list[HoleCircle]) -> np.ndarray:
    if not holes: return np.zeros_like(black_raw_mask, dtype=bool)
    h, w = black_raw_mask.shape
    mask_u8 = (black_raw_mask.astype(np.uint8) * 255).astype(np.uint8)
    exclusion = np.zeros((h, w), dtype=np.uint8)
    for hole in holes:
        cx, cy, search_radius = float(hole.x_px), float(hole.y_px), max(hole.radius_px * 4.5, 12.0)
        x0, x1 = max(0, int(np.floor(cx - search_radius))), min(w - 1, int(np.ceil(cx + search_radius)))
        y0, y1 = max(0, int(np.floor(cy - search_radius))), min(h - 1, int(np.ceil(cy + search_radius)))
        ys, xs = np.where(mask_u8[y0 : y1 + 1, x0 : x1 + 1] > 0)
        exclusion_radius = 2.5
        if xs.size > 10:
            gx, gy = xs.astype(np.float32) + float(x0), ys.astype(np.float32) + float(y0)
            d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
            nearby = d[d <= search_radius]
            if nearby.size > 24:
                hist, _ = np.histogram(np.arctan2(gy[d <= search_radius] - cy, gx[d <= search_radius] - cx), bins=36, range=(-np.pi, np.pi))
                if float(np.count_nonzero(hist) / 36) >= 0.55 and (float(np.percentile(nearby, 90)) - float(np.percentile(nearby, 10))) <= (search_radius * 0.65):
                    exclusion_radius = max(2.5, min(float(np.percentile(nearby, 90)) + 1.5, search_radius))
        cv2.circle(exclusion, (int(round(cx)), int(round(cy))), max(1, int(round(exclusion_radius))), 255, thickness=-1)
    return exclusion > 0

def _merge_hole_clusters(holes: list[HoleCircle]) -> list[HoleCircle]:
    if not holes: return []
    n, adj, visited, merged = len(holes), {i: [] for i in range(len(holes))}, set(), []
    for i in range(n):
        for j in range(i + 1, n):
            if (holes[i].x_px - holes[j].x_px)**2 + (holes[i].y_px - holes[j].y_px)**2 <= (holes[i].radius_px + holes[j].radius_px)**2:
                adj[i].append(j); adj[j].append(i)
    for i in range(n):
        if i not in visited:
            cluster, q = [], [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
            merged.append(HoleCircle(sum(holes[idx].x_px for idx in cluster) / len(cluster), sum(holes[idx].y_px for idx in cluster) / len(cluster), holes[cluster[0]].radius_px))
    return merged