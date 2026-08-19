from __future__ import annotations
import sys
import argparse
import time
import json
import threading
import urllib.request
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QStyleFactory
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
QPushButton:checked { background-color: #a0a0a0; border-top: 2px solid #404040; border-left: 2px solid #404040; border-right: 2px solid #ffffff; border-bottom: 2px solid #ffffff; }
"""

def sync_req(ip: str, endpoint: str, method: str = "POST", data: dict | None = None) -> dict:
    url = f"http://{ip}:8000/api/v0/{endpoint}"
    
    req_json = json.dumps(data, ensure_ascii=False) if data else ""
    if len(req_json) > 250:
        req_json = req_json[:250] + "..."
    log_req = f"📡 [SYNC Запрос] {method} /{endpoint} | {req_json}" if req_json else f"📡 [SYNC Запрос] {method} /{endpoint}"
    logger.info(log_req)
    
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.data = json.dumps(data).encode('utf-8')
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode('utf-8')
            res_data = json.loads(raw) if raw else {}
            res_json = json.dumps(res_data, ensure_ascii=False) if res_data else "{}"
            if len(res_json) > 250:
                res_json = res_json[:250] + "..."
            logger.info(f"📥 [SYNC Ответ] /{endpoint} | {res_json}")
            return res_data
    except Exception as e:
        logger.error(f"❌ Ошибка соединения ({endpoint}): {e}")
        raise

# ---- Управление консольным вводом для Headless-режима ----
headless_enter_event = threading.Event()

def headless_key_listener(ip: str):
    logger.info("⌨️  Горячие клавиши (Консоль): [Esc] - Стоп | [Пробел] - Пауза | [R] - Возобновить")
    try:
        import msvcrt
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Игнорируем префиксы спец-клавиш (стрелочек и т.д.), из-за которых ломалась пауза
                if key in (b'\x00', b'\xe0'):
                    msvcrt.getch()
                    continue

                if key == b'\x1b':
                    logger.warning("🛑 Экстренная остановка (Esc)!")
                    sync_req(ip, "stop")
                elif key == b' ':  # Пробел работает надежнее, чем Ctrl+Space
                    logger.info("⏸️ Пауза (Пробел)")
                    sync_req(ip, "pause")
                elif key in (b'r', b'R', b'\x10'): # Поддержка 'r', 'R' и старого 'Ctrl+P'
                    logger.info("▶️ Возобновление (R)")
                    sync_req(ip, "resume")
                elif key in (b'\r', b'\n'):
                    headless_enter_event.set()
                time.sleep(0.05)
    except ImportError:
        import sys, select, tty, termios, atexit
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        def restore_tty():
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        atexit.register(restore_tty)
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)
                if key == '\x1b':
                    logger.warning("🛑 Экстренная остановка (Esc)!")
                    sync_req(ip, "stop")
                elif key == ' ':
                    logger.info("⏸️ Пауза (Пробел)")
                    sync_req(ip, "pause")
                elif key in ('r', 'R', '\x10'):
                    logger.info("▶️ Возобновление (R)")
                    sync_req(ip, "resume")
                elif key in ('\n', '\r'):
                    headless_enter_event.set()

def start_headless_key_listener(ip: str):
    t = threading.Thread(target=headless_key_listener, args=(ip,), daemon=True)
    t.start()

def wait_for_enter(prompt: str):
    print(f"\n{prompt}")
    headless_enter_event.clear()
    headless_enter_event.wait()
# --------------------------------------------------------

def check_and_park(ip: str):
    try:
        coords = sync_req(ip, "get_coords", method="GET").get("coordinates")
        if not coords:
            logger.info("⏳ Станок не отпаркован. Выполняю парковку...")
            sync_req(ip, "parking")
            logger.info("✅ Успешно отпарковано!")
        else:
            logger.info("✅ Станок уже отпаркован.")
    except Exception as e:
        logger.error("❌ Не удалось выполнить парковку.")
        sys.exit(1)

def run_headless_full(args: argparse.Namespace, target_ip: str) -> int:
    try:
        if not args.config:
            logger.error("❌ Укажите путь к конфигу через --config")
            return 1
            
        start_headless_key_listener(target_ip)
            
        app = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow(skip_validation=args.skip_validation)
        window.load_config_from_yaml(Path(args.config))
        
        logger.info("🛠️ Генерация векторных путей...")
        window._generate()
        if not window._last_result:
            logger.error("❌ Ошибка генерации SVG. Прерывание.")
            return 1
            
        check_and_park(target_ip)

        logger.info("📐 Получаю лимиты рабочей зоны...")
        limits_resp = sync_req(target_ip, "get_limits", method="GET")
        limits = limits_resp.get("limits")
        if not limits:
            logger.error("❌ Не удалось получить лимиты рабочей зоны с сервера!")
            return 1
        window.machine_limits = limits
        
        config = window._build_config() # Получаем конфиг, чтобы узнать состояние чекбоксов
        
        logger.info("📸 Запрашиваю фото и разметку (fast)...")
        if config.auto_light_on:
            try: sync_req(target_ip, "set_light_state", data={"state": 1})
            except: pass
            
        try: sync_req(target_ip, "get_photo?camera=up&cut=true", method="GET")
        except: pass
        time.sleep(2)
        
        photo_data = sync_req(target_ip, "get_yoled_photo?camera=up&confidence_threshold=0.3&model_type=fast&cut=true", method="GET")
        
        if config.auto_light_off:
            try: sync_req(target_ip, "set_light_state", data={"state": 0})
            except: pass
            
        conv = sync_req(target_ip, "pixels_to_mm_bulk", data={"segments": photo_data.get("segments", [])})
        
        window.placement_detections = photo_data.get("detections", [])
        window.placement_segments_mm = conv.get("segments_mm", [])
        
        logger.info("🤖 Вычисляю авторазмещение...")
        res = window.find_auto_placement_sync()
        
        if not res:
            logger.error("❌ Не удалось найти место для авторазмещения (мало места в blank или пересечение с danger).")
            return 1
            
        pos, angle = res
        window.placed_pcbs = [{'pos': pos, 'angle': angle}]
        logger.info(f"✅ Плата успешно размещена: X={pos.x():.2f}, Y={pos.y():.2f}, Угол={angle}°")
        
        wait_for_enter("⚠️ Проверьте подключение крокодилов на шпинделе и заготовке!\nНажмите Enter, чтобы продолжить пробинг...")
        
        points = window._generate_probe_points()
        if not points:
            logger.error("❌ Не найдено безопасных точек для пробинга.")
            return 1
            
        logger.info(f"📏 Запуск Z-пробинга ({len(points)} точек)...")
        payload = {
            "points": points,
            "V_MAX": float(window.rapid_speed_steps_s.value()),
            "A_MAX": float(window.rapid_accel_steps_s2.value())
        }
        probe_res = sync_req(target_ip, "z_probe", data=payload)
        res_points = probe_res.get("points", [])
        
        if res_points:
            logger.info("🗺️ Сохраняю 3D сплайн...")
            out_dir = Path(args.output_dir) if args.output_dir else Path(args.config).parent / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            window.spline_widget.plot_points(res_points)
            window.spline_widget.fig.savefig(out_dir / "spline.png", dpi=150)
            logger.info(f"✅ Сплайн сохранен в {out_dir / 'spline.png'}")
            
        logger.info("🏠 Возврат в (0, 0, 0)...")
        sync_req(target_ip, "move_absolute", data={"x": 0, "y": 0, "z": 0, "diagonal": True})
        logger.info("🎉 Работа успешно завершена!")
        return 0
    except Exception as e:
        logger.error(f"❌ Критическая ошибка Headless: {e}", exc_info=True)
        return 1

def main() -> int:
    parser = argparse.ArgumentParser(description="PCB Toolpath Generator")
    parser.add_argument("-s", "--skip-validation", action="store_true", help="Игнорировать ошибки трассировки")
    parser.add_argument("-c", "--config", type=str, default=None, help="Путь к YAML-файлу")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Путь сохранения для headless")
    
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP адрес станка")
    parser.add_argument("--mac", type=str, default=None, help="MAC адрес станка (перезапишет --ip)")
    
    parser.add_argument("--coords", action="store_true", help="Вывести текущие координаты станка и выйти")
    parser.add_argument("--headless", action="store_true", help="Только генерация SVG в фоне")
    parser.add_argument("--move", nargs=3, type=float, metavar=('X','Y','Z'), help="Переместиться в координаты (X Y Z)")
    parser.add_argument("--move-pixel", nargs=2, type=float, metavar=('X','Y'), help="Переместиться в пиксель камеры (X Y)")
    parser.add_argument("--full-auto", action="store_true", help="Полный автоматический цикл (SVG -> Размещение -> Пробинг)")
    
    # Новые и старые аргументы управления
    parser.add_argument("--park", action="store_true", help="Сделать паркинг станка")
    parser.add_argument("--stop", action="store_true", help="Остановить станок")
    parser.add_argument("--pause", action="store_true", help="Поставить станок на паузу")
    parser.add_argument("--resume", action="store_true", help="Возобновить работу станка")

    # Новые аргументы:
    parser.add_argument("--sleep", action="store_true", help="Перевести станок в режим сна")
    parser.add_argument("--wake", action="store_true", help="Разбудить станок")
    parser.add_argument("--light-on", action="store_true", help="Включить свет")
    parser.add_argument("--light-off", action="store_true", help="Выключить свет")
    
    parsed_args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    
    base_dir = Path(__file__).resolve().parent
    setup_logging(base_dir / "app.log")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    target_ip = parsed_args.ip
    if parsed_args.mac:
        from ip_by_mac import get_ip_by_mac
        resolved = get_ip_by_mac(parsed_args.mac)
        if resolved:
            target_ip = resolved
            logger.info(f"🌐 MAC {parsed_args.mac} успешно разрешен в IP: {target_ip}")
        else:
            logger.error(f"❌ Не удалось разрешить MAC {parsed_args.mac} в IP адрес.")
            return 1
    
    # --- Блок быстрого управления станком через CLI ---
    if parsed_args.stop:
        logger.warning("🛑 Отправка команды остановки (--stop)")
        sync_req(target_ip, "stop")
        return 0

    if parsed_args.pause:
        logger.info("⏸️ Отправка команды паузы (--pause)")
        sync_req(target_ip, "pause")
        return 0

    if parsed_args.resume:
        logger.info("▶️ Отправка команды возобновления (--resume)")
        sync_req(target_ip, "resume")
        return 0
        
    if parsed_args.park:
        logger.info("🏠 Отправка команды парковки (--park)")
        start_headless_key_listener(target_ip)
        check_and_park(target_ip)
        return 0

    if parsed_args.sleep:
        logger.info("💤 Отправка команды режима сна (--sleep)")
        sync_req(target_ip, "sleep_mode", data={"state": 1})
        return 0

    if parsed_args.wake:
        logger.info("☀️ Отправка команды пробуждения (--wake)")
        sync_req(target_ip, "sleep_mode", data={"state": 0})
        return 0
        
    if parsed_args.light_on:
        logger.info("💡 Отправка команды включения света (--light-on)")
        sync_req(target_ip, "set_light_state", data={"state": 1})
        return 0
        
    if parsed_args.light_off:
        logger.info("💡 Отправка команды выключения света (--light-off)")
        sync_req(target_ip, "set_light_state", data={"state": 0})
        return 0
    # --------------------------------------------------
    
    if parsed_args.coords:
        try:
            res = sync_req(target_ip, "get_coords", method="GET")
            c = res.get("coordinates")
            if c: logger.info(f"📍 Текущие координаты: X:{c['x']:.2f} Y:{c['y']:.2f} Z:{c['z']:.2f}")
            else: logger.info("📍 Станок не отпаркован (координаты неизвестны).")
        except Exception:
            pass
        return 0

    if parsed_args.full_auto:
        return run_headless_full(parsed_args, target_ip)
        
    if parsed_args.move:
        start_headless_key_listener(target_ip)
        check_and_park(target_ip)
        logger.info(f"🚀 Перемещение в X:{parsed_args.move[0]} Y:{parsed_args.move[1]} Z:{parsed_args.move[2]}")
        sync_req(target_ip, "move_absolute", data={"x": parsed_args.move[0], "y": parsed_args.move[1], "z": parsed_args.move[2], "diagonal": True})
        logger.info("✅ Перемещение выполнено.")
        return 0
        
    if parsed_args.move_pixel:
        start_headless_key_listener(target_ip)
        check_and_park(target_ip)
        logger.info(f"🎯 Перемещение в пиксель X:{parsed_args.move_pixel[0]} Y:{parsed_args.move_pixel[1]}")
        sync_req(target_ip, "move_pixel", data={"px_x": parsed_args.move_pixel[0], "px_y": parsed_args.move_pixel[1], "z": 0, "diagonal": True})
        logger.info("✅ Перемещение выполнено.")
        return 0
        
    if parsed_args.headless:
        if not parsed_args.config:
            logger.error("❌ Ошибка: --headless требует --config <path_to_yaml>")
            return 1
        config_path = Path(parsed_args.config)
        out_dir = Path(parsed_args.output_dir) if parsed_args.output_dir else config_path.parent / "outputs"
        config = PipelineConfig.from_yaml(config_path, skip_validation=parsed_args.skip_validation)
        result = run_pipeline(config)
        export_svg_bundle(result, config, out_dir)
        logger.info("✅ SVG успешно сгенерирован.")
        return 0

    if "windows" in {name.lower() for name in QStyleFactory.keys()}: 
        app.setStyle("windows")
        
    app.setStyleSheet(build_stylesheet())
    window = MainWindow(skip_validation=parsed_args.skip_validation)
    
    if parsed_args.config:
        config_path = Path(parsed_args.config)
        if config_path.exists(): window.load_config_from_yaml(config_path)
            
    window.show()
    return app.exec_()

if __name__ == "__main__":
    raise SystemExit(main())