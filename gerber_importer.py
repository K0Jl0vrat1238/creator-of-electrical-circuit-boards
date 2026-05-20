# gerber_importer.py
from __future__ import annotations
import re
import math
from pathlib import Path
import cv2
import numpy as np

_last_avg_drill_dia = 0.8

def get_last_avg_drill_diameter() -> float:
    return _last_avg_drill_dia


class GerberParser:
    def __init__(self):
        self.units = "mm" 
        self.num_int = 2
        self.num_dec = 4
        self.leading_zeros = True
        self.apertures = {}
        self.current_aperture = None
        self.polygon_mode = False
        self.polygon_points = []
        self.paths = []  
        self.cx, self.cy = 0.0, 0.0

    def _determine_global_settings(self, content: str):
        if '%MOMM' in content or 'G71' in content:
            self.units = "mm"
        elif '%MOIN' in content or 'G70' in content:
            self.units = "inch"

        fs_match = re.search(r'%FSLAX(\d)(\d)Y\1\2', content)
        if fs_match:
            self.num_int = int(fs_match.group(1))
            self.num_dec = int(fs_match.group(2))
            self.leading_zeros = True
        elif '%FST' in content:
            self.leading_zeros = False

    def parse_coordinate(self, val_str: str) -> float:
        if not val_str:
            return 0.0
        sign = 1.0
        if val_str[0] == '-':
            sign = -1.0
            val_str = val_str[1:]
        elif val_str[0] == '+':
            val_str = val_str[1:]

        if '.' in val_str:
            val = float(val_str)
        else:
            total_digits = self.num_int + self.num_dec
            val_str = val_str.zfill(total_digits) if self.leading_zeros else val_str.ljust(total_digits, '0')
            if len(val_str) > self.num_dec:
                int_part = val_str[:-self.num_dec]
                dec_part = val_str[-self.num_dec:]
            else:
                int_part = "0"
                dec_part = val_str.zfill(self.num_dec)
            val = float(f"{int_part}.{dec_part}")

        if self.units == "inch":
            val *= 25.4
        return sign * val

    def parse(self, filepath: Path) -> list:
        if not filepath.exists():
            return []
        
        content = filepath.read_text(encoding="utf-8", errors="ignore").replace("\r", "").replace("\n", "")
        self._determine_global_settings(content)
        
        # 1. Извлечение и парсинг параметров %...% до разбиения по звездочкам
        params = re.findall(r'%([^%]+)%', content)
        for param in params:
            param = param.strip()
            if param.startswith('MO'):
                if 'MM' in param:
                    self.units = "mm"
                else:
                    self.units = "inch"
            elif param.startswith('FS'):
                match = re.search(r'X(\d)(\d)Y\1\2', param)
                if match:
                    self.num_int = int(match.group(1))
                    self.num_dec = int(match.group(2))
                    self.leading_zeros = True
                if 'T' in param:
                    self.leading_zeros = False
            elif param.startswith('ADD'):
                match = re.match(r'ADD(\d+)([a-zA-Z]+),?([0-9.X]+)?', param)
                if match:
                    ap_id = int(match.group(1))
                    ap_type = match.group(2)
                    ap_dims = match.group(3)
                    dims = []
                    if ap_dims:
                        scale_factor = 25.4 if self.units == "inch" else 1.0
                        dims = [float(x) * scale_factor for x in re.split(r'[xX]', ap_dims)]
                    self.apertures[ap_id] = {"type": ap_type, "dims": dims}

        # 2. Очистка от блоков параметров и парсинг геометрии
        clean_content = re.sub(r'%[^%]+%', '', content)
        blocks = clean_content.split('*')
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if re.match(r'^D\d+$', block):
                ap_id = int(block[1:])
                if ap_id in self.apertures:
                    self.current_aperture = self.apertures[ap_id]
                continue

            if 'G36' in block:
                self.polygon_mode = True
                self.polygon_points = []
                continue
            if 'G37' in block:
                self.polygon_mode = False
                if len(self.polygon_points) >= 3:
                    self.paths.append(('polygon', list(self.polygon_points), None))
                self.polygon_points = []
                continue

            match_coords = re.search(r'(G\d+)?(?:X(-?\d+))?(?:Y(-?\d+))?(?:D(01|02|03))?$', block)
            if match_coords:
                x_str = match_coords.group(2)
                y_str = match_coords.group(3)
                d_code = match_coords.group(4)

                nx = self.parse_coordinate(x_str) if x_str else self.cx
                ny = self.parse_coordinate(y_str) if y_str else self.cy

                if d_code == '01':  
                    if self.polygon_mode:
                        self.polygon_points.append((nx, ny))
                    else:
                        width = self.current_aperture["dims"][0] if (self.current_aperture and self.current_aperture["dims"]) else 0.4
                        self.paths.append(('line', (self.cx, self.cy, nx, ny), width))
                elif d_code == '03':  
                    ap_type = self.current_aperture["type"] if self.current_aperture else 'C'
                    ap_dims = self.current_aperture["dims"] if (self.current_aperture and self.current_aperture["dims"]) else [1.2]
                    self.paths.append(('flash', (nx, ny), ap_type, ap_dims))

                self.cx, self.cy = nx, ny

        return self.paths


class ExcellonParser:
    def __init__(self):
        self.scale = 25.4
        self.tools = {}
        self.current_tool = None
        self.drills = []
        self.cx, self.cy = 0.0, 0.0

    def parse(self, filepath: Path) -> list[tuple[float, float, float]]:
        if not filepath.exists():
            return []

        content = filepath.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        header_mode = True

        for line in lines:
            line = line.strip().upper()
            if not line:
                continue

            if 'METRIC' in line:
                self.scale = 1.0
            elif 'INCH' in line:
                self.scale = 25.4

            if header_mode:
                if line.startswith('T') and 'C' in line:
                    match = re.match(r'T(\d+)C([0-9.]+)', line)
                    if match:
                        t_id = int(match.group(1))
                        diameter = float(match.group(2)) * self.scale
                        self.tools[t_id] = diameter
                elif line in ('M30', '%') or 'DETECT' in line:
                    header_mode = False
                continue

            if line.startswith('T') and 'C' not in line:
                match = re.match(r'T(\d+)', line)
                if match:
                    t_id = int(match.group(1))
                    self.current_tool = self.tools.get(t_id, 0.8)
                continue

            match_drill = re.match(r'(?:X(-?[0-9.]+))?(?:Y(-?[0-9.]+))?', line)
            if match_drill and (match_drill.group(1) or match_drill.group(2)):
                raw_x = match_drill.group(1)
                raw_y = match_drill.group(2)
                
                def parse_val(val_str: str) -> float:
                    if not val_str:
                        return 0.0
                    if '.' in val_str:
                        return float(val_str) * self.scale
                    divisor = 100000.0 if self.scale == 25.4 else 1000.0
                    return (float(val_str) / divisor) * self.scale

                if raw_x:
                    self.cx = parse_val(raw_x)
                if raw_y:
                    self.cy = parse_val(raw_y)
                
                diameter = self.current_tool if self.current_tool else 0.8
                self.drills.append((self.cx, self.cy, diameter / 2.0))

        return self.drills


def scale_drills_to_match_copper(
    drills: list[tuple[float, float, float]], 
    c_min_x: float, c_max_x: float, c_min_y: float, c_max_y: float
) -> list[tuple[float, float, float]]:
    if not drills:
        return []
    
    test_factors = [1.0, 0.1, 0.01, 10.0, 25.4, 2.54, 0.254, 0.0254]
    best_scale = 1.0
    best_score = float('inf')
    
    for scale in test_factors:
        xs = [d[0] * scale for d in drills]
        ys = [d[1] * scale for d in drills]
        d_min_x, d_max_x = min(xs), max(xs)
        d_min_y, d_max_y = min(ys), max(ys)
        
        score = (abs(d_min_x - c_min_x) + abs(d_max_x - c_max_x) + 
                 abs(d_min_y - c_min_y) + abs(d_max_y - c_max_y))
        
        if score < best_score:
            best_score = score
            best_scale = scale
            
    return [(x * best_scale, y * best_scale, r * best_scale) for x, y, r in drills]


def distance_to_copper(x: float, y: float, copper_points: list[tuple[float, float]]) -> float:
    if not copper_points:
        return 0.0
    pts = np.array(copper_points)
    dists = np.sum((pts - np.array([x, y]))**2, axis=1)
    return math.sqrt(np.min(dists))


def render_gerber_and_excellon(
    layers_paths: dict[str, Path], dpi: float = 600.0
) -> tuple[np.ndarray, float, float]:
    global _last_avg_drill_dia
    parsed_data = {'paths': [], 'green': [], 'cut': [], 'holes': []}
    raw_copper_points = []

    # 1. Чтение медного слоя
    if 'paths' in layers_paths:
        p = GerberParser()
        parsed_data['paths'] = p.parse(layers_paths['paths'])
        for item in parsed_data['paths']:
            if item[0] == 'line':
                raw_copper_points.extend([(item[1][0], item[1][1]), (item[1][2], item[1][3])])
            elif item[0] in ('flash', 'polygon'):
                if item[0] == 'polygon':
                    raw_copper_points.extend(item[1])
                else:
                    raw_copper_points.append(item[1])

    if not raw_copper_points:
        raw_copper_points = [(0.0, 0.0), (30.0, 25.0)]

    c_xs = [pt[0] for pt in raw_copper_points]
    c_ys = [pt[1] for pt in raw_copper_points]
    
    med_x = np.median(c_xs)
    med_y = np.median(c_ys)
    
    # Зона платы лежит строго в пределах 18.0 мм от медианы плотности
    limit_min_x, limit_max_x = med_x - 18.0, med_x + 18.0
    limit_min_y, limit_max_y = med_y - 18.0, med_y + 18.0

    copper_points = [
        pt for pt in raw_copper_points 
        if limit_min_x <= pt[0] <= limit_max_x and limit_min_y <= pt[1] <= limit_max_y
    ]
    if not copper_points:
        copper_points = raw_copper_points

    c_xs_clean = [pt[0] for pt in copper_points]
    c_ys_clean = [pt[1] for pt in copper_points]
    c_min_x, c_max_x = min(c_xs_clean), max(c_xs_clean)
    c_min_y, c_max_y = min(c_ys_clean), max(c_ys_clean)

    # 2. Чтение и выравнивание Excellon
    if 'holes' in layers_paths:
        p = ExcellonParser()
        raw_drills = p.parse(layers_paths['holes'])
        parsed_data['holes'] = scale_drills_to_match_copper(raw_drills, c_min_x, c_max_x, c_min_y, c_max_y)
        if parsed_data['holes']:
            _last_avg_drill_dia = sum(r * 2.0 for _, _, r in parsed_data['holes']) / len(parsed_data['holes'])

    # 3. Чтение остальных слоев с жесткой фильтрацией угловых маркеров (лимит < 1.5 мм до меди)
    dist_threshold = 1.5

    if 'green' in layers_paths:
        raw_green = GerberParser().parse(layers_paths['green'])
        for item in raw_green:
            if item[0] == 'line':
                x1, y1, x2, y2 = item[1]
                if distance_to_copper(x1, y1, copper_points) < dist_threshold or distance_to_copper(x2, y2, copper_points) < dist_threshold:
                    parsed_data['green'].append(item)
            elif item[0] == 'polygon':
                if any(distance_to_copper(pt[0], pt[1], copper_points) < dist_threshold for pt in item[1]):
                    parsed_data['green'].append(item)

    if 'cut' in layers_paths:
        raw_cut = GerberParser().parse(layers_paths['cut'])
        for item in raw_cut:
            if item[0] == 'line':
                x1, y1, x2, y2 = item[1]
                if distance_to_copper(x1, y1, copper_points) < dist_threshold or distance_to_copper(x2, y2, copper_points) < dist_threshold:
                    parsed_data['cut'].append(item)

    # Настройка холста
    margin = 1.0
    min_x = c_min_x - margin
    max_x = c_max_x + margin
    min_y = c_min_y - margin
    max_y = c_max_y + margin

    width_mm = max_x - min_x
    height_mm = max_y - min_y

    scale_px_per_mm = dpi / 25.4
    w_px = int(math.ceil(width_mm * scale_px_per_mm))
    h_px = int(math.ceil(height_mm * scale_px_per_mm))

    canvas = np.full((h_px, w_px, 4), 255, dtype=np.uint8)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = int(round((x - min_x) * scale_px_per_mm))
        py = int(round((max_y - y) * scale_px_per_mm))
        return px, py

    # --- Отрисовка ---

    # 1. Текст (Green)
    for item in parsed_data['green']:
        if item[0] == 'line':
            p1, p2 = to_px(item[1][0], item[1][1]), to_px(item[1][2], item[1][3])
            w = max(1, int(round(item[2] * scale_px_per_mm)))
            cv2.line(canvas, p1, p2, (0, 200, 0, 255), w)
        elif item[0] == 'flash':
            p = to_px(item[1][0], item[1][1])
            r = max(1, int(round((item[2] / 2.0) * scale_px_per_mm)))
            cv2.circle(canvas, p, r, (0, 200, 0, 255), -1)
        elif item[0] == 'polygon':
            pts = np.array([to_px(pt[0], pt[1]) for pt in item[1]], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], (0, 200, 0, 255))

    # 2. Дорожки и контактные площадки (Black)
    for item in parsed_data['paths']:
        if item[0] == 'line':
            x1, y1, x2, y2 = item[1]
            if distance_to_copper(x1, y1, copper_points) > dist_threshold:
                continue
            p1, p2 = to_px(x1, y1), to_px(x2, y2)
            w = max(1, int(round(item[2] * scale_px_per_mm)))
            cv2.line(canvas, p1, p2, (20, 20, 20, 255), w)
        elif item[0] == 'flash':
            # item: ('flash', (x, y), ap_type, ap_dims)
            (fx, fy), ap_type, ap_dims = item[1], item[2], item[3]
            if distance_to_copper(fx, fy, copper_points) > dist_threshold:
                continue
            p = to_px(fx, fy)
            if ap_type == 'C':  
                r = max(1, int(round((ap_dims[0] / 2.0) * scale_px_per_mm)))
                cv2.circle(canvas, p, r, (20, 20, 20, 255), -1)
            elif ap_type in ('R', 'O'):  
                w_mm = ap_dims[0]
                h_mm = ap_dims[1] if len(ap_dims) > 1 else ap_dims[0]
                w_px = max(1, int(round(w_mm * scale_px_per_mm)))
                h_px = max(1, int(round(h_mm * scale_px_per_mm)))
                p1 = (p[0] - w_px // 2, p[1] - h_px // 2)
                p2 = (p[0] + w_px // 2, p[1] + h_px // 2)
                cv2.rectangle(canvas, p1, p2, (20, 20, 20, 255), -1)
        elif item[0] == 'polygon':
            if not all(distance_to_copper(pt[0], pt[1], copper_points) < dist_threshold for pt in item[1]):
                continue
            pts = np.array([to_px(pt[0], pt[1]) for pt in item[1]], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], (20, 20, 20, 255))

    # 3. Контур (Red)
    for item in parsed_data['cut']:
        if item[0] == 'line':
            p1, p2 = to_px(item[1][0], item[1][1]), to_px(item[1][2], item[1][3])
            w = max(1, int(round(item[2] * scale_px_per_mm)))
            cv2.line(canvas, p1, p2, (255, 0, 0, 255), w)

    # 4. Отверстия (Blue)
    for x, y, r in parsed_data['holes']:
        p = to_px(x, y)
        r_px = max(1, int(round(r * scale_px_per_mm)))
        cv2.circle(canvas, p, r_px, (0, 80, 255, 255), -1)

    return canvas, width_mm, height_mm