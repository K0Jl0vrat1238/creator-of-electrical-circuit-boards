#!/usr/bin/env python3
"""
Топологическая проверка связности векторизации.
Проверяет сохранение связности (breaks) и отсутствие коротких замыканий (shorts)
для слоёв BLACK (дорожки) и RED (контур) с учётом геометрии V-фрезы.
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
import math
from pathlib import Path
import sys
from pipeline import run_pipeline, PipelineConfig, _classify_color_masks, _segment_color_masks


def rasterize_vector_paths(paths: list[list[tuple[float, float]]], width: int, height: int, thickness_px: int) -> np.ndarray:
    """Растеризует векторные пути линией заданной толщины (физический диаметр фрезы)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for path in paths:
        if len(path) < 2:
            continue
        pts = np.array([[int(round(x)), int(round(y))] for x, y in path], np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness_px, lineType=cv2.LINE_AA)
    return mask > 0


def get_components(mask: np.ndarray) -> dict[int, np.ndarray]:
    """Возвращает словарь {component_id: binary_mask} для всех связных областей."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask.astype(np.uint8) * 255).astype(np.uint8), connectivity=8
    )
    comps = {}
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] > 5:  # Игнорируем микро-шум
            comps[idx] = labels == idx
    return comps


def analyze_topology(gt_comps: dict[int, np.ndarray], vec_comps: dict[int, np.ndarray], min_overlap_ratio: float = 0.05):
    """
    Сравнивает топологию Ground Truth и Вектора.
    Возвращает списки проблем: breaks, shorts, и статус ok_comps.
    """
    breaks = []   # (gt_id, vec_ids) -> одна дорожка разорвалась на несколько
    shorts = []   # (vec_id, gt_ids) -> несколько изолированных дорожек слиплись
    ok_comps = [] # (gt_id, vec_id) -> топология сохранена

    vec_to_gt = {v_id: [] for v_id in vec_comps}
    gt_to_vec = {g_id: [] for g_id in gt_comps}

    for v_id, v_mask in vec_comps.items():
        for g_id, g_mask in gt_comps.items():
            intersection = np.logical_and(v_mask, g_mask).sum()
            gt_area = g_mask.sum()
            if gt_area > 0 and intersection > min_overlap_ratio * gt_area:
                vec_to_gt[v_id].append(g_id)
                gt_to_vec[g_id].append(v_id)

    # Анализ разрывов (BREAKS)
    for g_id, v_ids in gt_to_vec.items():
        if len(v_ids) == 0:
            breaks.append({"type": "MISSING", "gt_id": g_id, "detail": "Дорожка полностью потеряна"})
        elif len(v_ids) > 1:
            breaks.append({"type": "SPLIT", "gt_id": g_id, "vec_ids": v_ids, "detail": f"Разорвана на {len(v_ids)} частей"})
        else:
            ok_comps.append((g_id, v_ids[0]))

    # Анализ коротких замыканий (SHORTS)
    for v_id, g_ids in vec_to_gt.items():
        if len(g_ids) > 1:
            shorts.append({"vec_id": v_id, "gt_ids": g_ids, "detail": f"Слили {len(g_ids)} изолированных дорожки"})

    return breaks, shorts, ok_comps


def save_topology_report(gt_mask: np.ndarray, vec_mask: np.ndarray, breaks: list, shorts: list, save_dir: Path, layer_name: str):
    """Сохраняет визуальный отчёт с подсветкой ошибок."""
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[gt_mask, 1] = 150  # GT полупрозрачный зелёный
    overlay[vec_mask, 2] = 150 # Vector полупрозрачный красный

    cv2.putText(overlay, f"BREAKS: {len(breaks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"SHORTS: {len(shorts)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imencode(".png", overlay)[1].tofile(str(save_dir / f"{layer_name}_topology.png"))


def _calc_vbit_width_mm(depth_mm: float, angle_deg: float) -> float:
    """Эффективная ширина реза V-фрезы на заданной глубине."""
    if depth_mm <= 0 or angle_deg <= 0 or angle_deg >= 180:
        return 0.0
    return 2.0 * depth_mm * math.tan(math.radians(angle_deg / 2.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Топологическая проверка связности (breaks/shorts)")
    parser.add_argument("image", type=Path, help="Путь к исходному изображению")
    parser.add_argument("--width-mm", type=float, default=100.0, help="Физическая ширина в мм")
    parser.add_argument("--tool-angle", type=float, default=30.0, help="Угол V-фрезы (градусы)")
    parser.add_argument("--min-gap-px", type=float, default=0.0, help="Допустимый зазор (0.0 = касание 1px = коротыш)")
    parser.add_argument("--output-dir", type=Path, default=Path("connectivity_test_results"), help="Папка для отчётов")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f" Загрузка: {args.image}")
    raw = np.fromfile(str(args.image), dtype=np.uint8)
    rgba = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        print("❌ Ошибка чтения изображения")
        return 1

    if rgba.ndim == 3 and rgba.shape[2] == 3:
        rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)])

    h, w = rgba.shape[:2]
    px_per_mm = w / args.width_mm

    print(f"⚙️ Запуск пайплайна...")
    config = PipelineConfig(
        image_path=args.image, width_mm=args.width_mm, height_mm=None,
        tool_angle_deg=args.tool_angle, drill_diameter_mm=0.8,
        stepover_percent=50.0, simplify_tolerance_mm=0.0, green_mode="centerline",
        stock_thickness_mm=1.5, cut_speed_mm_s=20.0, max_accel_mm_s2=500.0,
        trace_depth_mm=0.2, green_depth_mm=0.1
    )
    result = run_pipeline(config)

    print("🎨 Извлечение эталонных масок...")
    raw_masks = _classify_color_masks(rgba)
    gt_masks = _segment_color_masks(rgba, raw_masks=raw_masks)

    layers = {
        "black": ("black_paths", result.black_paths, "Дорожки (BLACK)", config.trace_depth_mm),  # ⬅️ Обновлено
        "red":   ("cut_paths",   result.cut_paths,   "Контур (RED)",   config.stock_thickness_mm)
    }

    print("\n" + "=" * 60)
    print("🔍 ТОПОЛОГИЧЕСКАЯ ПРОВЕРКА СВЯЗНОСТИ")
    print("=" * 60)
    print(f"📏 Масштаб: {px_per_mm:.2f} px/mm | 🛠 Угол фрезы: {args.tool_angle}°")

    total_breaks = 0
    total_shorts = 0

    for color, (res_attr, vec_data, label, depth_mm) in layers.items():
        print(f"\n📦 Анализ слоя: {label}")
        gt_mask = gt_masks[color]
        if gt_mask.sum() == 0:
            print("  ⏭️ [SKIP] Слой пуст на оригинале.")
            continue

        # Расчет физической толщины линии для растеризации под V-фрезу
        eff_width_mm = _calc_vbit_width_mm(depth_mm, args.tool_angle)
        thickness_px = max(1, int(round(eff_width_mm * px_per_mm)))
        print(f"  📐 Глубина: {depth_mm}мм → Эффективная толщина: {eff_width_mm:.4f} мм ({thickness_px} px)")

        kernel_size = max(1, int(round(args.min_gap_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        gt_clean = gt_mask.copy()
        if args.min_gap_px > 0:
            gt_u8 = (gt_clean.astype(np.uint8) * 255)
            gt_u8 = cv2.morphologyEx(gt_u8, cv2.MORPH_CLOSE, kernel)
            gt_clean = gt_u8 > 0

        gt_comps = get_components(gt_clean)
        vec_mask = rasterize_vector_paths(vec_data, w, h, thickness_px)
        vec_comps = get_components(vec_mask)

        breaks, shorts, ok = analyze_topology(gt_comps, vec_comps)
        total_breaks += len(breaks)
        total_shorts += len(shorts)

        print(f"  ✅ Сохранено связных областей: {len(ok)}")
        if breaks:
            print(f"  ⚠️  РАЗРЫВЫ (BREAKS): {len(breaks)}")
            for b in breaks:
                print(f"     -> {b['detail']}")
        if shorts:
            print(f"  🔴 КОРОТКИЕ ЗАМЫКАНИЯ (SHORTS): {len(shorts)}")
            for s in shorts:
                print(f"     -> {s['detail']}")
        save_topology_report(gt_mask, vec_mask, breaks, shorts, args.output_dir, color)

    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"Всего разрывов:  {total_breaks}")
    print(f"Всего коротышей: {total_shorts}")
    if total_breaks == 0 and total_shorts == 0:
        print("✅ ТОПОЛОГИЯ ИДЕАЛЬНА. Плата готова к фрезеровке.")
    else:
        print("⚠️ Обнаружены топологические ошибки. Проверьте overlay-файлы в отчёте.")
    print(f"Визуальные отчёты: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())