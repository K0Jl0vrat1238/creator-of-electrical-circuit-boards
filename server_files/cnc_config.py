import logging

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- НАСТРОЙКИ ПИНОВ И ПАРАМЕТРОВ (BOARD режим) ---
X_STEPS_MM = 800.0
Y_STEPS_MM = 800.0
Z_STEPS_MM = 800.0

# Добавили параметр 'final_offset_mm' индивидуально для каждой оси в конфигурацию
AXES_CONFIG = {
    'Z': {
        'step': 8, 'dir': 10, 'en': 12, 'endstop': 18, 
        'steps_per_mm': Z_STEPS_MM, 'final_offset_mm': 2.0  # Отъезд 2 мм
    },
    'X': {
        'step': 3, 'dir': 5, 'en': 7, 'endstop': 22, 
        'steps_per_mm': X_STEPS_MM, 'final_offset_mm': 2.0  # Отъезд 2 мм
    },
    'Y': {
        'step': 11, 'dir': 13, 'en': 15, 'endstop': 24, 
        'steps_per_mm': Y_STEPS_MM, 'final_offset_mm': 162.0 # Отъезд 92 мм
    }
}

# Запретные зоны (софтверные лимиты в мм)
# PCB touch probe pin (BOARD mode). Change this to match the RED_CRAB wiring.
RED_CRAB = 16

MIN_X, MAX_X = 0.0, 260.0
MIN_Y, MAX_Y = -160.0, 20.0
MIN_Z, MAX_Z = 0.0, 30.0

# Количество подходов для усреднения точности
NUM_APPROACHES = 8

# Скорости (задержки) паркинга
FAST_SEEK_DELAY = 0.000001  # Ваша сверхбыстрая скорость (50 микросекунд)
SLOW_SEEK_DELAY = 0.0001   # Медленная скорость для точного замера (200 микросекунд)

# Параметры перемещения по апи запросу
V_MIN = 800.0   # Стартовая скорость (шагов/сек)
V_MAX = 30000.0  # Максимальная скорость (шагов/сек)
A_MAX = 15000.0  # Ускорение (шагов/сек²)

# Параметры для паркинга - поиск концевика и и точное позиционирование, в шагах в секунду
V_FAST_HOME = 90000.0  # Максимальная скорость для быстрого набега на концевик
V_SLOW_HOME = 10000.0  # Максимальная скорость для точных подходов, откатов и позиционирования
