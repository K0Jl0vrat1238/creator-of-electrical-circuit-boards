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
        self.red_crab_triggered = False
        self.z_probe_active = False
        self.active_operation = None
        self.interrupted_operation = None
        self.pause = False  # Флаг остановки
        self.resume_target = None    # Сюда пишем конечную точку пути
        
        # Новые поля для идеального пауза/резьюма пробинга
        self.remaining_z_probe_points = []
        self.z_probe_results = []
        self.first_touch_z = None
        self.z_probe_v_max = None
        self.z_probe_a_max = None
        
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

    def _red_crab_callback(self, pin):
        """Interrupt callback for the PCB touch probe."""
        if GPIO.getmode() is None or not self.z_probe_active:
            return

        # Сильно уменьшаем первичную паузу до 0.5 миллисекунды (просто чтобы прошел первый пик искры)
        time.sleep(0.002) 
        
        try:
            high_count = 0
            num_reads = 5000
            
            # Делаем 100 быстрых замеров состояния пина подряд
            for _ in range(num_reads):
                if GPIO.input(pin) == GPIO.HIGH:
                    high_count += 1
            
            # Голосование: если состояние HIGH было зафиксировано больше чем в половине случаев
            if high_count > (num_reads / 2):
                self.red_crab_triggered = True
                
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

        # ИЗМЕНЕНИЕ: Меняем PUD_UP на PUD_DOWN, так как в воздухе у нас LOW
        GPIO.setup(RED_CRAB, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Снимаем прерывание (если оно висело), но НЕ вешаем его заново!
        try:
            GPIO.remove_event_detect(RED_CRAB)
        except EnvironmentError:
            pass
        self.red_crab_triggered = False
        time.sleep(0.1)
        
    def shutdown(self):
        """Обесточивание обмоток моторов без риска Segmentation fault"""
        # Если режим еще не задан (сервер только включился) - задаем его
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
        # Настраиваем ТОЛЬКО пины EN на выход и сразу рубим питание моторов
        for axis in ['X', 'Y', 'Z']:
            if axis in AXES_CONFIG:
                en_pin = AXES_CONFIG[axis]['en']
                try:
                    GPIO.setup(en_pin, GPIO.OUT)
                    GPIO.output(en_pin, GPIO.HIGH)
                except Exception:
                    pass

        # Безопасно отписываемся от прерываний (если они были запущены)
        for axis in ['X', 'Y', 'Z']:
            if axis in AXES_CONFIG:
                try:
                    GPIO.remove_event_detect(AXES_CONFIG[axis]['endstop'])
                except Exception:
                    pass
        try:
            GPIO.remove_event_detect(RED_CRAB)
        except Exception:
            pass

        # Очищаем память
        GPIO.cleanup()

    def step(self, axis, direction, delay):
        if self.pause:
            return False
            
        cfg = AXES_CONFIG[axis]
        
        if direction == GPIO.LOW and self.endstop_triggered[axis]:
            return False 

        if self.z_probe_active and axis == 'Z' and direction == GPIO.HIGH and self.red_crab_triggered:
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

    def move_axis_trapezoid(self, axis, steps, direction, v_max=None, a_max=None):
        if steps <= 0: 
            return 0
            
        steps = int(steps)
        max_speed = v_max if v_max is not None else V_MAX
        accel = a_max if a_max is not None else A_MAX
        
        s_accel = int((max_speed**2 - V_MIN**2) / (2 * accel))
        if s_accel > steps / 2:
            s_accel = steps // 2
            
        s_decel = steps - s_accel
        v_current = V_MIN
        
        actual_steps = 0
        
        try:
            for i in range(steps):
                if i < s_accel:
                    v_current += accel * (1.0 / v_current)
                    if v_current > max_speed: v_current = max_speed
                elif i >= s_decel:
                    v_current -= accel * (1.0 / v_current)
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
            self.move_axis_trapezoid(axis, 5 * self.spm[axis], GPIO.HIGH, v_max=V_SLOW_HOME)
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
        self.z_probe_active = False
        self.remaining_z_probe_points = []
        self.z_probe_results = []
        self.first_touch_z = None
        self.z_probe_v_max = None
        self.z_probe_a_max = None
        self.interrupted_operation = None
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
        
    def move_diagonal_trapezoid(self, target_steps_dict, v_max=None, a_max=None):
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
        accel = a_max if a_max is not None else A_MAX
        s_accel = int((max_speed**2 - V_MIN**2) / (2 * accel))
        if s_accel > max_delta / 2:
            s_accel = max_delta // 2
            
        s_decel = max_delta - s_accel
        v_current = V_MIN

        for i in range(max_delta):
            if self.pause:
                raise RuntimeError("Движение прервано пользователем через Pause!")

            # Разгон и торможение ведущей оси
            if i < s_accel:
                v_current += accel * (1.0 / v_current)
                if v_current > max_speed: v_current = max_speed
            elif i >= s_decel:
                v_current -= accel * (1.0 / v_current)
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
        
    def move_absolute(self, x=None, y=None, z=None, diagonal=False, v_max=None, a_max=None):
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
            self.move_diagonal_trapezoid(target_steps, v_max=v_max, a_max=a_max)
            
        else:
            for axis in ['Z', 'X', 'Y']:
                target_steps = int(targets[axis] * self.spm[axis])
                diff = target_steps - self.pos[axis]
                
                if diff != 0:
                    direction = GPIO.HIGH if diff > 0 else GPIO.LOW
                    made_steps = self.move_axis_trapezoid(axis, abs(diff), direction, v_max=v_max, a_max=a_max)
                    
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

                        if self.z_probe_active and axis == 'Z' and direction == GPIO.HIGH and self.red_crab_triggered:
                            raise RuntimeError("Emergency stop: RED_CRAB triggered during fast Z move")

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

    def _ensure_red_crab_released(self):
        try:
            is_triggered = GPIO.input(RED_CRAB) == GPIO.HIGH
        except RuntimeError as err:
            raise RuntimeError(f"Cannot read RED_CRAB: {err}")

        if is_triggered:
            self.red_crab_triggered = True
            raise RuntimeError("RED_CRAB is still triggered")
        self.red_crab_triggered = False

    def _probe_z_slow(self):
        self._ensure_red_crab_released()

        current_z = self.pos['Z'] / self.spm['Z']
        max_steps = int((MAX_Z - current_z) * self.spm['Z'])
        if max_steps <= 0:
            raise RuntimeError("No Z travel left for probing")

        was_probe_active = self.z_probe_active
        self.z_probe_active = True
        try:
            for _ in range(max_steps):
                if self.pause:
                    raise RuntimeError("Z probing interrupted by Pause")
                if self.red_crab_triggered:
                    return self.pos['Z'] / self.spm['Z']

                if not self.step('Z', GPIO.HIGH, 1.0 / V_MIN):
                    raise RuntimeError("Z probing step failed")
                self.pos['Z'] += 1

                if self.red_crab_triggered:
                    return self.pos['Z'] / self.spm['Z']
        finally:
            self.z_probe_active = was_probe_active

        raise RuntimeError("RED_CRAB was not triggered before MAX_Z")

    def z_probe_points(self, points, v_max=None, a_max=None, is_resume=False):
        if self.pause:
            raise RuntimeError("Сперва выполните resume")
        if not self.is_homed:
            raise RuntimeError("Сперва выполните паркинг")

        if not is_resume and not points:
            return []

        max_speed = v_max if v_max is not None else V_MAX
        accel = a_max if a_max is not None else A_MAX
        if max_speed < V_MIN:
            raise RuntimeError("V_MAX must be greater than or equal to V_MIN")
        if accel <= 0:
            raise RuntimeError("A_MAX must be greater than zero")

        self.active_operation = 'z_probe'
        
        # --- НАЧАЛО Z-ПРОБИНГА: Включаем прерывание RED_CRAB ---
        try:
            GPIO.remove_event_detect(RED_CRAB)
        except EnvironmentError:
            pass
        self.red_crab_triggered = False
        GPIO.add_event_detect(RED_CRAB, GPIO.RISING, callback=self._red_crab_callback, bouncetime=10)

        # Если это новый запуск, а не продолжение, то инициализируем состояние
        if not is_resume:
            raw_points = [dict(p) if hasattr(p, 'dict') else p for p in points]
            
            # --- ОПТИМИЗАЦИЯ МАРШРУТА ПРОБИНГА (ЗМЕЙКА) ---
            if raw_points:
                raw_points.sort(key=lambda p: p['y'], reverse=True)
                
                rows = []
                current_row = []
                y_tolerance = 2.0 
                
                for p in raw_points:
                    if not current_row:
                        current_row.append(p)
                    else:
                        if abs(p['y'] - current_row[0]['y']) <= y_tolerance:
                            current_row.append(p)
                        else:
                            rows.append(current_row)
                            current_row = [p]
                if current_row:
                    rows.append(current_row)
                    
                optimized_points = []
                for i, row in enumerate(rows):
                    if i % 2 == 0:
                        row.sort(key=lambda p: p['x'])
                    else:
                        row.sort(key=lambda p: p['x'], reverse=True)
                    optimized_points.extend(row)
                    
                self.remaining_z_probe_points = optimized_points
            else:
                self.remaining_z_probe_points = []
            # ----------------------------------------------

            self.z_probe_results = []
            self.first_touch_z = None
            self.z_probe_v_max = max_speed
            self.z_probe_a_max = accel
        
        try:
            # Крутимся, пока есть незавершенные точки
            while self.remaining_z_probe_points:
                if self.pause:
                    raise RuntimeError("Z probing interrupted by Pause")

                point = self.remaining_z_probe_points[0]
                x = float(point['x'])
                y = float(point['y'])

                if not (MIN_X <= x <= MAX_X) or not (MIN_Y <= y <= MAX_Y):
                    raise RuntimeError(f"Point is outside XY limits: X{x} Y{y}")

                # Поднимаем Z (z_probe_active = False, игнорируем датчик)
                self.move_absolute(z=0.0, v_max=max_speed, a_max=accel)
                self._ensure_red_crab_released()
                
                # Переезд в XY (z_probe_active = False)
                self.move_absolute(x=x, y=y, z=0.0, v_max=max_speed, a_max=accel)

                # Быстрый спуск вниз (если уже был первый замер)
                if self.first_touch_z is not None:
                    self._ensure_red_crab_released()
                    fast_z = max(MIN_Z, self.first_touch_z - 2.0)
                    if fast_z > 0.0:
                        # Включаем флаг активности только для езды ВНИЗ
                        self.z_probe_active = True
                        try:
                            self.move_absolute(z=fast_z, v_max=max_speed, a_max=accel)
                        finally:
                            # Обязательно выключаем даже при краше/паузе
                            self.z_probe_active = False

                # Медленный замер (функция _probe_z_slow сама включает/выключает флаг z_probe_active внутри себя)
                touched_z = self._probe_z_slow()
                
                if self.first_touch_z is None:
                    self.first_touch_z = touched_z

                # Записываем результат и удаляем точку из списка только ПОСЛЕ успешного замера
                self.z_probe_results.append({'x': x, 'y': y, 'z': touched_z})
                self.remaining_z_probe_points.pop(0)

            # Если всё прошли успешно — безопасно поднимаем фрезу (z_probe_active = False)
            self.move_absolute(z=0.0, v_max=max_speed, a_max=accel)
            final_res = list(self.z_probe_results)
            
            self.interrupted_operation = None
            self.remaining_z_probe_points = []
            self.z_probe_results = []
            self.first_touch_z = None
            return final_res

        except Exception:
            if self.pause:
                self.interrupted_operation = 'z_probe'
            raise
        finally:
            # --- КОНЕЦ Z-ПРОБИНГА: Зачищаем флаги и прерывания при любом исходе ---
            self.z_probe_active = False
            self.active_operation = None
            
            try:
                GPIO.remove_event_detect(RED_CRAB)
            except EnvironmentError:
                pass
            self.red_crab_triggered = False
