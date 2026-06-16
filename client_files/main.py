from __future__ import annotations
import shutil
import sys
import argparse
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QStyleFactory
)
from pipeline import PipelineConfig, run_pipeline
from svg_export import export_svg_bundle

from common import setup_logging
from main_window import MainWindow
import logging
logger = logging.getLogger("gui")

def build_stylesheet() -> str:
    return """
QWidget { background-color: #c0c0c0; color: #000000; font-family: "MS Sans Serif", sans-serif; font-size: 11px; }
QTabWidget::pane { border: 1px solid #7f7f7f; top: -1px; }
QTabBar::tab { background: #c0c0c0; border: 1px solid #7f7f7f; padding: 4px 10px; }
QTabBar::tab:selected { background: #d6d6d6; }
QGroupBox { border: 1px solid #7f7f7f; margin-top: 10px; padding-top: 10px; font-weight: bold; }
QGroupBox::title { left: 8px; top: -2px; padding: 0 3px; }
QLineEdit, QDoubleSpinBox, QComboBox, QListWidget, QCheckBox { background-color: #ffffff; border: 1px solid #808080; padding: 2px 4px; }
QPushButton { background-color: #c0c0c0; border-top: 2px solid #ffffff; border-left: 2px solid #ffffff; border-right: 2px solid #404040; border-bottom: 2px solid #404040; min-height: 20px; padding: 2px 10px; }
QPushButton:pressed { border-top: 2px solid #404040; border-left: 2px solid #404040; border-right: 2px solid #ffffff; border-bottom: 2px solid #ffffff; }
"""

def run_headless_workflow(config_path: Path, output_dir: Path | None, skip_validation: bool) -> int:
    try:
        logger.info("Initializing headless workflow execution...")
        config = PipelineConfig.from_yaml(config_path, skip_validation=skip_validation)
        
        if output_dir is None:
            resolved_output_dir = config_path.parent / "outputs"
        else:
            resolved_output_dir = Path(output_dir)
            
        logger.info(f"Preparing target output directory: {resolved_output_dir}")
        if resolved_output_dir.exists():
            shutil.rmtree(resolved_output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Executing processing pipeline...")
        result = run_pipeline(config)
        
        logger.info("Exporting processed vector tracks to SVG...")
        files = export_svg_bundle(result, config, resolved_output_dir)
        
        logger.info(f"Processing successfully finalized. Created files ({len(files)}):")
        for key, path in files.items():
            logger.info(f"  [{key}] -> {path.resolve()}")
            
        return 0
    except Exception as e:
        logger.error(f"Execution failed with critical error: {e}", exc_info=True)
        return 1

def main() -> int:
    parser = argparse.ArgumentParser(description="PCB Toolpath Generator")
    parser.add_argument("-s", "--skip-validation", action="store_true", help="Игнорировать ошибки трассировки")
    parser.add_argument("-c", "--config", type=str, default=None, help="Путь к YAML-файлу")
    parser.add_argument("--headless", action="store_true", help="Запуск в консольном режиме")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Путь сохранения для headless")
    
    parsed_args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    
    base_dir = Path(__file__).resolve().parent
    setup_logging(base_dir / "app.log")
    
    logger.info(f"Command line parsed. skip_validation={parsed_args.skip_validation}, config={parsed_args.config}")
    
    if parsed_args.headless:
        if not parsed_args.config:
            logger.error("Error: --headless requires a config file. Please specify --config <path_to_yaml>")
            return 1
        return run_headless_workflow(
            config_path=Path(parsed_args.config),
            output_dir=Path(parsed_args.output_dir) if parsed_args.output_dir else None,
            skip_validation=parsed_args.skip_validation
        )

    if "windows" in {name.lower() for name in QStyleFactory.keys()}: 
        app.setStyle("windows")
        
    app.setStyleSheet(build_stylesheet())
    
    window = MainWindow(skip_validation=parsed_args.skip_validation)
    
    if parsed_args.config:
        config_path = Path(parsed_args.config)
        if config_path.exists(): window.load_config_from_yaml(config_path)
        else: logger.error(f"Config file not found: {parsed_args.config}")
            
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
