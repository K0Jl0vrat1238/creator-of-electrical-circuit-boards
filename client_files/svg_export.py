# svg_export.py
from __future__ import annotations
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml  # Используем библиотеку PyYAML
from pipeline import PipelineResult, PipelineConfig

logger = logging.getLogger("svg_export")

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def export_yaml_config(config: PipelineConfig, result: PipelineResult, path: Path) -> Path:
    """Сохраняет конфигурацию запуска в формате YAML с использованием PyYAML."""
    logger.info(f"Saving YAML configuration using PyYAML to: {path}")
    
    data = {
        "source_image": config.image_path.resolve().as_posix(),
        "physical_dimensions": {
            "width_mm": round(result.width_mm, 4),
            "height_mm": round(result.height_mm, 4),
            "resolved_from": "width" if config.width_mm is not None else "height"
        },
        "tool_parameters": {
            "angle_deg": config.tool_angle_deg,
            "drill_diameters_mm": config.drill_diameters_mm,
            "stepover_percent": config.stepover_percent,
            "simplify_tolerance_mm": config.simplify_tolerance_mm,
            "green_mode": config.green_mode
        },
        "gcode_generation": {
            "stock_thickness_mm": config.stock_thickness_mm,
            "cut_speed_mm_s": config.cut_speed_mm_s,
            "plunge_speed_steps_s": config.plunge_speed_steps_s,
            "max_accel_mm_s2": config.max_accel_mm_s2,
            "trace_depth_mm": config.trace_depth_mm,
            "green_depth_mm": config.green_depth_mm
        }
    }
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data, 
                f, 
                default_flow_style=False, 
                sort_keys=False, 
                allow_unicode=True,
                indent=2
            )
    except Exception as e:
        logger.error(f"Failed to write YAML config: {e}", exc_info=True)
        raise e

    return path


def export_svg_bundle(result: PipelineResult, config: PipelineConfig, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    
    logger.info(f"Exporting SVG layers and configuration to target directory: {output_dir}")
    
    # 1. Экспорт конфигурационного файла YAML
    files["config.yaml"] = export_yaml_config(config, result, output_dir / "config.yaml")
    
    # 2. Экспортируем файлы только для тех слоев, которые содержат данные
    if result.holes:
        files["holes.svg"] = _write_holes_svg(result, output_dir / "holes.svg")
        logger.debug(f"Holes layer saved: {files['holes.svg'].name}")
        
    if result.green_paths:
        files["green.svg"] = _write_paths_svg(
            result,
            result.green_paths,
            output_dir / "green.svg",
            stroke="#00a000",
        )
        logger.debug(f"Green layer saved: {files['green.svg'].name}")
        
    if result.black_paths:
        files["paths.svg"] = _write_paths_svg(
            result,
            result.black_paths,
            output_dir / "paths.svg",
            stroke="#222222",
        )
        logger.debug(f"Black layer saved: {files['paths.svg'].name}")
        
    if result.cut_paths:
        files["cut.svg"] = _write_paths_svg(
            result,
            result.cut_paths,
            output_dir / "cut.svg",
            stroke="#cc0000",
        )
        logger.debug(f"Cut layer saved: {files['cut.svg'].name}")
        
    return files


def _svg_root(result: PipelineResult) -> ET.Element:
    return ET.Element(
        f"{{{SVG_NS}}}svg",
        attrib={
            "version": "1.1",
            "width": f"{result.width_mm:.6f}mm",
            "height": f"{result.height_mm:.6f}mm",
            "viewBox": f"0 0 {result.width_px} {result.height_px}",
        },
    )


def _write_holes_svg(result: PipelineResult, path: Path) -> Path:
    root = _svg_root(result)
    for hole in result.holes:
        color_hex = f"#{hole.color[0]:02x}{hole.color[1]:02x}{hole.color[2]:02x}"
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}circle",
            attrib={
                "cx": f"{hole.x_px:.4f}",
                "cy": f"{hole.y_px:.4f}",
                "r": f"{hole.radius_px:.4f}",
                "fill": color_hex,
                "stroke": "none",
            },
        )
    _save_xml(root, path)
    return path


def _write_paths_svg(
    result: PipelineResult,
    paths: list[list[tuple[float, float]]],
    path: Path,
    stroke: str,
) -> Path:
    root = _svg_root(result)
    for polyline in paths:
        d = _polyline_to_path_d(polyline)
        if not d:
            continue
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}path",
            attrib={
                "d": d,
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "1",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )
    _save_xml(root, path)
    return path


def _polyline_to_path_d(polyline: list[tuple[float, float]]) -> str:
    if len(polyline) < 2:
        return ""
    start = polyline[0]
    parts = [f"M {start[0]:.4f} {start[1]:.4f}"]
    for x, y in polyline[1:]:
        parts.append(f"L {x:.4f} {y:.4f}")
    return " ".join(parts)


def _save_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)