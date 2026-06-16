
#!/usr/bin/env python3
"""
Топологическая проверка связности векторизации.
Проверяет сохранение связности (breaks), отсутствие коротких замыканий (shorts)
и пересечений между контуром платы (RED) и медными дорожками (BLACK).
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from client_files.pipeline import run_pipeline, PipelineConfig, _classify_color_masks, _segment_color_masks


def rasterize_vector_paths(paths: list[list[tuple[float, float]]], width: int, height: int, thickness_px: int) -> np.ndarray:
    """Растеризует векторные пути линией заданной толщины (симметрично в обе стороны от центра)."""
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


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """
    Заполняет внутренние отверстия в бинарной маске.
    Превращает контур (линию) в сплошную область (блин).
    """
    if mask.sum() == 0:
        return mask
    
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
    h, w = padded.shape
    
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    inv_img = cv2.bitwise_not((padded.astype(np.uint8) * 255))
    cv2.floodFill(inv_img, flood_mask, (0, 0), 255)
    
    holes = flood_mask[1:-1, 1:-1] == 0
    filled = np.logical_or(padded, holes)[1:-1, 1:-1]
    return filled


def analyze_topology(gt_comps: dict[int, np.ndarray], vec_comps: dict[int, np.ndarray], min_overlap_ratio: float = 0.05):
    """
    Сравнивает топологию Ground Truth и Вектора.
    Возвращает списки проблем: breaks, shorts, и статус ok_comps.
    """
    breaks = []   
    shorts = []   
    ok_comps = [] 
    
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


def save_topology_report(gt_mask: np.ndarray, vec_mask: np.ndarray, breaks: list, shorts: list, save_dir: Path, layer_name: str, original_image: np.ndarray = None):
    """Сохраняет визуальный отчёт."""
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    if layer_name == "red":
        gt_filled = fill_internal_holes(gt_mask)
        overlay[gt_filled, 1] = 150  
        
        red_mask = np.logical_or(gt_mask, vec_mask)
        overlay[red_mask, 1] = 0     
        overlay[red_mask, 2] = 200   
    else:
        overlay[gt_mask, 1] = 150  
        overlay[vec_mask, 2] = 150 
    
    cv2.putText(overlay, f"BREAKS: {len(breaks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"SHORTS: {len(shorts)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    if original_image is not None:
        bg = original_image[..., :3]
        blended = cv2.addWeighted(overlay, 0.5, bg, 0.5, 0)
        cv2.imencode(".png", blended)[1].tofile(str(save_dir / f"{layer_name}_topology.png"))
    else:
        cv2.imencode(".png", overlay)[1].tofile(str(save_dir / f"{layer_name}_topology.png"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Топологическая проверка связности (breaks/shorts/collisions)")
    parser.add_argument("--image", type=Path, default=project_root / "test_files/test_imgs/test_png.png", help="Путь к исходному изображению")
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
        "black": ("black_paths", result.black_paths, "Дорожки (BLACK)", result.black_tool_radius_px),
        "red":   ("cut_paths",   result.cut_paths,   "Контур (RED)",    result.cut_tool_radius_px)
    }

    print("\n" + "=" * 60)
    print(" ТОПОЛОГИЧЕСКАЯ ПРОВЕРКА СВЯЗНОСТИ")
    print("=" * 60)
    print(f"📏 Масштаб: {px_per_mm:.2f} px/mm | 🛠 Угол фрезы: {args.tool_angle}°")

    total_breaks = 0
    total_shorts = 0
    rasterized_masks = {}  

    for color, (res_attr, vec_data, label, radius_px) in layers.items():
        print(f"\n📦 Анализ слоя: {label}")
        gt_mask = gt_masks[color]
        if gt_mask.sum() == 0:
            print("  ️ [SKIP] Слой пуст на оригинале.")
            continue

        # Умножаем радиус на 2 для получения полной ширины реза
        thickness_px = max(1, int(round(radius_px * 2)))
        thickness_mm = (radius_px * 2) / px_per_mm
        print(f"  📐 Расчетная толщина реза (диаметр фрезы): {thickness_mm:.3f} мм ({thickness_px} px)")

        kernel_size = max(1, int(round(args.min_gap_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        gt_clean = gt_mask.copy()
        if args.min_gap_px > 0:
            gt_u8 = (gt_clean.astype(np.uint8) * 255)
            gt_u8 = cv2.morphologyEx(gt_u8, cv2.MORPH_CLOSE, kernel)
            gt_clean = gt_u8 > 0

        gt_comps = get_components(gt_clean)
        
        # Растеризуем с точной физической толщиной фрезы
        vec_mask = rasterize_vector_paths(vec_data, w, h, thickness_px)
        
        vec_mask_vis = vec_mask.copy()
        rasterized_masks[color] = vec_mask_vis
        
        if color == "red":
            vec_mask = fill_internal_holes(vec_mask)
            
        vec_comps = get_components(vec_mask)

        breaks, shorts, ok = analyze_topology(gt_comps, vec_comps)
        
        if color == "red":
            shorts = []

        total_breaks += len(breaks)
        total_shorts += len(shorts)

        print(f"  ✅ Сохранено связных областей: {len(ok)}")
        if breaks:
            print(f"  ⚠️  РАЗРЫВЫ (BREAKS): {len(breaks)}")
        if shorts:
            print(f"  🔴 КОРОТКИЕ ЗАМЫКАНИЯ (SHORTS): {len(shorts)}")
                
        save_topology_report(gt_mask, vec_mask_vis, breaks, shorts, args.output_dir, color, original_image=rgba)

    # ==========================================================
    # ПРОВЕРКА ПЕРЕСЕЧЕНИЙ КОНТУРА (RED) И ДОРОЖЕК (BLACK)
    # ==========================================================
    total_collisions = 0
    gt_black = gt_masks.get("black")
    vec_red = rasterized_masks.get("red")

    if gt_black is not None and vec_red is not None and gt_black.sum() > 0 and vec_red.sum() > 0:
        print("\n" + "=" * 60)
        print(" ПРОВЕРКА ПЕРЕСЕЧЕНИЙ (ДОРОЖКИ vs КОНТУР)")
        print("=" * 60)

        overlap = np.logical_and(gt_black, vec_red)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (overlap.astype(np.uint8) * 255), connectivity=8
        )
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 3:  
                total_collisions += 1

        if total_collisions > 0:
            print(f"  ❌ ВНИМАНИЕ: Найдено {total_collisions} мест(а) где контурная фреза задевает медные дорожки!")
        else:
            print("  ✅ Пересечений контура и медных дорожек не найдено. Плата не будет повреждена.")

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[gt_black] = [150, 150, 150] 
        overlay[vec_red] = [0, 0, 255]      
        overlay[overlap] = [0, 255, 255]    
        
        bg = rgba[..., :3].copy()
        blended = cv2.addWeighted(overlay, 0.9, bg, 0.4, 0)
        
        status_color = (0, 255, 255) if total_collisions > 0 else (0, 255, 0)
        cv2.putText(blended, f"COLLISIONS: {total_collisions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        out_file = args.output_dir / "collision_black_red.png"
        cv2.imencode(".png", blended)[1].tofile(str(out_file))
        print(f"  📸 Карта пересечений сохранена: {out_file}")

    print("\n" + "=" * 60)
    print(" ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"Всего разрывов:   {total_breaks}")
    print(f"Всего коротышей:  {total_shorts}")
    print(f"Всего коллизий:   {total_collisions}")
    
    if total_breaks == 0 and total_shorts == 0 and total_collisions == 0:
        print("✅ ТОПОЛОГИЯ ИДЕАЛЬНА. Плата готова к фрезеровке.")
    else:
        print("⚠️ Обнаружены топологические ошибки. Проверьте overlay-файлы в отчёте.")
    print(f"Визуальные отчёты: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
