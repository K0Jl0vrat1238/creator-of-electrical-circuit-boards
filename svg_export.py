# svg_export.py
from __future__ import annotations
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from pipeline import PipelineResult

logger = logging.getLogger("svg_export")

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def export_svg_bundle(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    
    logger.info(f"Exporting SVG layers to target directory: {output_dir}")

    # Экспортируем файлы только для тех слоев, которые содержат данные
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
        
    logger.info(f"SVG Export finalized successfully. Total written layers: {len(files)}")
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
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}circle",
            attrib={
                "cx": f"{hole.x_px:.4f}",
                "cy": f"{hole.y_px:.4f}",
                "r": f"{hole.radius_px:.4f}",
                "fill": "#0050ff",
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