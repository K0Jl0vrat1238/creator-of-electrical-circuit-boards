import time
import logging
import RPi.GPIO as GPIO
from statistics import mean
from cnc_config import *

logger = logging.getLogger(__name__)

class CNCMachine:
    def __init__(self):
        self.pos = {'X': 0, 'Y': 0, 'Z': 0}
        self.spm = {'X': X_STEPS_MM, 'Y': Y_STEPS_MM, 'Z': Z_STEPS_MM}
        self.is_homed = False
        self.is_busy = False
        self.endstop_triggered = {'X': False, 'Y': False, 'Z': False}
        self.pause = False  # Флаг остановки
        self.resume_target = None    # Сюда пишем конечную точку пути
        
    def _endstop_callback(self, pin):
        """Фоновый обработчик прерывания от процессора"""
        if GPIO.getmode() is None:
            return 
            
        for axis, cfg in AXES_CONFIG.items():
            if cfg['endstop'] == pin:
                time.sleep(0.002) 
                try:
                    if GPIO.input(pin) == GPIO.LOW:
                        self.endstop_triggered[axis] = True
                except RuntimeError:
                    pass 

    def setup(self):
        """Инициализация портов перед работой или парковкой"""
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        for axis in ['X', 'Y', 'Z']:
            cfg = AXES_CONFIG[axis]
            GPIO.setup(cfg['endstop'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            try:
                GPIO.remove_event_detect(cfg['endstop'])
            except EnvironmentError:
                pass
            
            self.endstop_triggered[axis] = False
            GPIO.add_event_detect(cfg['endstop'], GPIO.FALLING, callback=self._endstop_callback, bouncetime=50)
            
            GPIO.setup([cfg['step'], cfg['dir'], cfg['en']], GPIO.OUT)
            GPIO.output(cfg['en'], GPIO.LOW) 
        time.sleep(0.1)
    
    def shutdown(self):
        """Обесточивание обмоток моторов"""
        # Если режим пинов не задан, значит мы даже не стартовали моторы — просто уходим
        if GPIO.getmode() is None:
            logger.info("Cервер закрылся сразу после старта.")
            # return
         
        self.setup()
            
        for axis in ['X', 'Y', 'Z']:
            if axis in AXES_CONFIG:
                try:
                    GPIO.output(AXES_CONFIG[axis]['en'], GPIO.HIGH)
                except RuntimeError:
                    pass
        GPIO.cleanup()

    def step(self, axis, direction, delay):
        if self.pause:
            return False
            
        cfg = AXES_CONFIG[axis]
        
        if direction == GPIO.LOW and self.endstop_triggered[axis]:
            return False 
        
        if direction == GPIO.HIGH:
            self.endstop_triggered[axis] = False
        
        GPIO.output(cfg['dir'], direction)
        
        # Наша спасительная пауза для DIR-тайминга драйвера
        time.sleep(0.000005) 
        
        GPIO.output(cfg['step'], GPIO.HIGH)
        time.sleep(delay / 2.0)
        GPIO.output(cfg['step'], GPIO.LOW)
        time.sleep(delay / 2.0)
        
        return True

    def move_axis_trapezoid(self, axis, steps, direction, v_max=None):
        if steps <= 0: 
            return 0
            
        steps = int(steps)
        max_speed = v_max if v_max is not None else V_MAX
        
        s_accel = int((max_speed**2 - V_MIN**2) / (2 * A_MAX))
        if s_accel > steps / 2:
            s_accel = steps // 2
            
        s_decel = steps - s_accel
        v_current = V_MIN
        
        actual_steps = 0
        
        try:
            for i in range(steps):
                if i < s_accel:
                    v_current += A_MAX * (1.0 / v_current)
                    if v_current > max_speed: v_current = max_speed
                elif i >= s_decel:
                    v_current -= A_MAX * (1.0 / v_current)
                    if v_current < V_MIN: v_current = V_MIN
                    
                if not self.step(axis, direction, 1.0 / v_current):
                    break
                    
                actual_steps += 1
        finally:
            pass
        
        return actual_steps

    def home_axis(self, axis):
        isStopped = False
        if self.pause:
            isStopped = True
            return False
        
        """Твоя оригинальная умная калибровка оси с многократным подходом"""
        cfg = AXES_CONFIG[axis]
        logger.info(f"========== ПАРКОВКА ОСИ [{axis}] ==========")
        
        initial_state = GPIO.input(cfg['endstop']) == GPIO.LOW
        logger.info(f"Статус концевика перед стартом: {initial_state} (False - свободен, True - зажат)")
        
        # 1. ПРЕДВАРИТЕЛЬНЫЙ ОТСКОК
        if GPIO.input(cfg['endstop']) == GPIO.LOW:
            logger.info(f"[{axis}] Концевик нажат! Освобождаем датчик с разгоном...")
            self.move_axis_trapezoid(axis, 15 * self.spm[axis], GPIO.HIGH, v_max=V_SLOW_HOME)
            time.sleep(0.3)
            if GPIO.input(cfg['endstop']) == GPIO.LOW:
                raise RuntimeError(f"Авария: Концевик [{axis}] не разблокировался!")

        # 2. БЫСТРЫЙ ПОИСК
        logger.info(f"[{axis}] Быстрый поиск концевика с разгоном до {V_FAST_HOME} шагов/сек...")
        self.endstop_triggered[axis] = False
        max_homing_steps = int(1000 * self.spm[axis])  # Запас хода 1000 мм
        self.move_axis_trapezoid(axis, max_homing_steps, GPIO.LOW, v_max=V_FAST_HOME)
        time.sleep(0.3)
        
        # 3. ЦИКЛ ТОЧНЫХ ПОДХОДОВ (Твоя легендарная математика!)
        logger.info(f"[{axis}] Запуск калибровки. Требуется успешных подходов: {NUM_APPROACHES}...")
        measured_steps = []
        bounce_distance_mm = 2.0
        bounce_steps = int(bounce_distance_mm * self.spm[axis])
        attempt_counter = 0 
        
        while len(measured_steps) < NUM_APPROACHES:
            if self.pause:
                isStopped = True
                return False
            
            attempt_counter += 1
            
            # Плавный откат назад по трапеции
            self.endstop_triggered[axis] = False
            self.move_axis_trapezoid(axis, bounce_steps, GPIO.HIGH, v_max=V_SLOW_HOME)
            time.sleep(0.2)
            
            # Аккуратное скольжение вперёд до клика концевика
            self.endstop_triggered[axis] = False
            steps_count = self.move_axis_trapezoid(axis, max_homing_steps, GPIO.LOW, v_max=V_SLOW_HOME)
                
            if abs(steps_count - bounce_steps) >= 100:
                logger.warning(f"  [!] Попытка {attempt_counter}: Пропуск! Шагов: {steps_count}. Отклонение слишком велико.")
                time.sleep(0.3)
                continue
                
            measured_steps.append(steps_count)
            logger.info(f"  -> Подход {len(measured_steps)}/{NUM_APPROACHES} (Попытка {attempt_counter}): концевик сработал через {steps_count} шагов.")
            time.sleep(0.2)
        
        if not isStopped:
            avg_steps = round(mean(measured_steps))
            logger.info(f"[{axis}] Результаты замеров: {measured_steps}. Среднее значение: {avg_steps} шагов.")
            
            # 4. ВЫСТАВЛЕНИЕ В СРЕДНИЙ НОЛЬ
            logger.info(f"[{axis}] Финальное позиционирование по среднему значению...")
            self.endstop_triggered[axis] = False
            self.move_axis_trapezoid(axis, bounce_steps, GPIO.HIGH, v_max=V_SLOW_HOME)
            time.sleep(0.2)
            
            self.endstop_triggered[axis] = False
            self.move_axis_trapezoid(axis, avg_steps, GPIO.LOW, v_max=V_SLOW_HOME)
            time.sleep(0.2)
            
            # 5. РАЗГРУЗОЧНЫЙ ОТХОД
            logger.info(f"[{axis}] Снятие нагрузки с датчика (отход на {cfg['final_offset_mm']} мм)...")
            self.endstop_triggered[axis] = False
            self.move_axis_trapezoid(axis, cfg['final_offset_mm'] * self.spm[axis], GPIO.HIGH, v_max=V_SLOW_HOME)
            
            self.pos[axis] = 0
            logger.info(f"Ось {axis} успешно встала в домашнее положение.")
        
        return not isStopped

    def home_all(self):
        """Полный цикл парковки всех осей в безопасном порядке"""
        self.pause = False
        self.resume_target = None
        self.setup()
        
        good = True
    
        good = self.home_axis('Z')   # Сначала поднимаем фрезу вверх!
        if good is not None and good:
            good = self.home_axis('X')
        if good is not None and good:
            good = self.home_axis('Y')
        if good is not None and good:
            self.is_homed = True
        else:
            self.is_homed = False
            self.pos['X'] = None
            self.pos['Y'] = None
            self.pos['Z'] = None
            
            raise RuntimeError("Парковка прервана пользователем!")
            
        logger.info("Станок полностью готов к работе, все координаты идеально обнулены!")
        
    def _step_multi(self, axes_to_step, dirs, delay):
            """Делает синхронный шаг сразу по нескольким осям."""
            if self.pause:
                return False

            # 1. Выставляем направления для всех двигающихся осей
            for ax in axes_to_step:
                GPIO.output(AXES_CONFIG[ax]['dir'], dirs[ax])
                
                # Если едем от концевика, сбрасываем флаг
                if dirs[ax] == GPIO.HIGH:
                    self.endstop_triggered[ax] = False
                    
            # Пауза для драйвера (DIR setup time)
            time.sleep(0.000005)

            # 2. Поднимаем STEP синхронно
            for ax in axes_to_step:
                GPIO.output(AXES_CONFIG[ax]['step'], GPIO.HIGH)
                
            time.sleep(delay / 2.0)
            
            # 3. Опускаем STEP синхронно
            for ax in axes_to_step:
                GPIO.output(AXES_CONFIG[ax]['step'], GPIO.LOW)
                
            time.sleep(delay / 2.0)
            
            return True
        
    def move_diagonal_trapezoid(self, target_steps_dict, v_max=None):
        """
        Многоосевое движение по алгоритму Брезенхэма с трапециевидным профилем скорости.
        target_steps_dict: словарь вида {'X': 1000, 'Y': 500, 'Z': 0} в абсолютных шагах
        """
        # Считаем разницу (дельты) и направления
        deltas = {}
        dirs = {}
        for ax in ['X', 'Y', 'Z']:
            diff = target_steps_dict[ax] - self.pos[ax]
            deltas[ax] = abs(diff)
            dirs[ax] = GPIO.HIGH if diff > 0 else GPIO.LOW
        max_ax = max(deltas, key=deltas.get)
        max_delta = deltas[max_ax]

        if max_delta == 0:
            return  # Ехать никуда не надо

        # Остальные две оси, которые будут подстраиваться под ведущую
        other_axes = [ax for ax in ['X', 'Y', 'Z'] if ax != max_ax]
        
        # Переменные для накопления ошибки (Алгоритм Брезенхэма)
        errors = {ax: max_delta / 2.0 for ax in other_axes}

        # Настройка профиля скорости по ведущей оси
        max_speed = v_max if v_max is not None else V_MAX
        s_accel = int((max_speed**2 - V_MIN**2) / (2 * A_MAX))
        if s_accel > max_delta / 2:
            s_accel = max_delta // 2
            
        s_decel = max_delta - s_accel
        v_current = V_MIN

        for i in range(max_delta):
            if self.pause:
                raise RuntimeError("Движение прервано пользователем через Pause!")

            # Разгон и торможение ведущей оси
            if i < s_accel:
                v_current += A_MAX * (1.0 / v_current)
                if v_current > max_speed: v_current = max_speed
            elif i >= s_decel:
                v_current -= A_MAX * (1.0 / v_current)
                if v_current < V_MIN: v_current = V_MIN

            # Определяем, какие оси должны сделать шаг в этой итерации
            axes_to_step = [max_ax]
            for ax in other_axes:
                errors[ax] -= deltas[ax]
                if errors[ax] < 0:
                    axes_to_step.append(ax)
                    errors[ax] += max_delta

            # Проверка концевиков ПЕРЕД шагом (защита от аварии)
            for ax in axes_to_step:
                if dirs[ax] == GPIO.LOW and self.endstop_triggered[ax]:
                    raise RuntimeError(f"Аварийный останов! Ось {ax} досрочно врезалась в концевик во время диагонального движения!")

            # Делаем синхронный шаг только для тех осей, кому пора
            if not self._step_multi(axes_to_step, dirs, 1.0 / v_current):
                break

            # Обновляем фактические координаты в self.pos
            for ax in axes_to_step:
                if dirs[ax] == GPIO.HIGH:
                    self.pos[ax] += 1
                else:
                    self.pos[ax] -= 1
        
    def move_absolute(self, x=None, y=None, z=None, diagonal=False):
        if self.pause:
            raise RuntimeError("Выполните Resume")
        
        if not self.is_homed:
            raise RuntimeError("Сперва запустите паркинг")
            
        # Вычисляем финальные абсолютные координаты (если ось не передана, берем текущую из self.pos)
        t_x = x if x is not None else (self.pos['X'] / self.spm['X'])
        t_y = y if y is not None else (self.pos['Y'] / self.spm['Y'])
        t_z = z if z is not None else (self.pos['Z'] / self.spm['Z'])
        
        # Запоминаем абсолютную цель, куда станок должен по итогу приехать
        self.resume_target = {'x': t_x, 'y': t_y, 'z': t_z, 'diagonal': diagonal}
        
        if not (MIN_X <= t_x <= MAX_X) or not (MIN_Y <= t_y <= MAX_Y) or not (MIN_Z <= t_z <= MAX_Z):
            raise RuntimeError(f"Попытка выезда в запретную зону! Target: X{t_x} Y{t_y} Z{t_z}")
            
        targets = {'X': t_x, 'Y': t_y, 'Z': t_z}
        
        if diagonal:
            # Переводим мм в шаги для диагонального алгоритма
            target_steps = {ax: int(targets[ax] * self.spm[ax]) for ax in ['X', 'Y', 'Z']}
            self.move_diagonal_trapezoid(target_steps)
            
        else:
            for axis in ['Z', 'X', 'Y']:
                target_steps = int(targets[axis] * self.spm[axis])
                diff = target_steps - self.pos[axis]
                
                if diff != 0:
                    direction = GPIO.HIGH if diff > 0 else GPIO.LOW
                    made_steps = self.move_axis_trapezoid(axis, abs(diff), direction)
                    
                    # Запоминаем текущие координаты строго в self.pos по факту сделанных шагов
                    if direction == GPIO.HIGH:
                        self.pos[axis] += made_steps
                    else:
                        self.pos[axis] -= made_steps
                        
                    # Если не доехали до конца (пауза или концевик)
                    if made_steps < abs(diff):
                        if self.pause:
                            raise RuntimeError("Движение прервано пользователем через Pause!")
                        if self.endstop_triggered[axis]:
                            raise RuntimeError(f"Аварийный останов! Ось {axis} досрочно врезалась в концевик!")

    def move_relative(self, x=None, y=None, z=None, diagonal=False):
        if self.pause:
            raise RuntimeError("Выполните Resume")
        
        if not self.is_homed:
            raise RuntimeError("Сперва запустите паркинг")
            
        # Пересчитываем относительный сдвиг в абсолютные целевые координаты
        t_x = (self.pos['X'] / self.spm['X']) + (x if x is not None else 0.0)
        t_y = (self.pos['Y'] / self.spm['Y']) + (y if y is not None else 0.0)
        t_z = (self.pos['Z'] / self.spm['Z']) + (z if z is not None else 0.0)
        
        # Просто вызываем абсолютное смещение — оно само всё запишет и отработает!
        self.move_absolute(x=t_x, y=t_y, z=t_z, diagonal=diagonal)

    def get_coordinates_mm(self):
        return {ax: self.pos[ax] / self.spm[ax] for ax in ['X', 'Y', 'Z']}