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
        
        self.motor_is_on = False      
        self.saved_motor_state = False

        # Добавляем состояние света
        self.light_is_on = False

        self.sleep_mode = True # По умолчанию при запуске станок в спячке (на тормозах)

        # Для G-кода
        self.gcode_commands = []
        self.gcode_index = 0
        self.current_feedrate = 600.0 # Скорость по умолчанию (мм/мин)
        
    def _endstop_callback(self, pin):
        """Фоновый обработчик прерывания от процессора с защитой от наводок"""
        if GPIO.getmode() is None:
            return 
            
        for axis, cfg in AXES_CONFIG.items():
            if cfg['endstop'] == pin:
                time.sleep(0.002) 
                
                try:
                    low_count = 0
                    high_count = 0
                    num_reads = 2000
                    threshold = num_reads // 2
                    
                    for _ in range(num_reads):
                        if GPIO.input(pin) == GPIO.LOW:
                            low_count += 1
                            # Если набрали половину + 1 — это точно нажатие, выходим
                            if low_count > threshold:
                                self.endstop_triggered[axis] = True
                                break
                        else:
                            high_count += 1
                            # Если набрали половину + 1 — это точно помеха, выходим
                            if high_count > threshold:
                                break
                                
                except RuntimeError:
                    pass 
                
                break

    def _red_crab_callback(self, pin):
        """Interrupt callback for the PCB touch probe."""
        if GPIO.getmode() is None or not self.z_probe_active:
            return

        time.sleep(0.002) 
        
        try:
            high_count = 0
            low_count = 0
            num_reads = 5000
            threshold = num_reads // 2
            
            for _ in range(num_reads):
                if GPIO.input(pin) == GPIO.HIGH:
                    high_count += 1
                    if high_count > threshold:
                        self.red_crab_triggered = True
                        break
                else:
                    low_count += 1
                    if low_count > threshold:
                        break
                        
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

            GPIO.output(cfg['en'], GPIO.HIGH) 

        # ИЗМЕНЕНИЕ: Меняем PUD_UP на PUD_DOWN, так как в воздухе у нас LOW
        GPIO.setup(RED_CRAB, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Снимаем прерывание (если оно висело), но НЕ вешаем его заново!
        try:
            GPIO.remove_event_detect(RED_CRAB)
        except EnvironmentError:
            pass
        self.red_crab_triggered = False
        time.sleep(0.1)
        
        # Выключаем пин аппаратно
        try:
            GPIO.setup(M_MOTOR, GPIO.OUT, initial=GPIO.HIGH)
        except Exception:
            pass
        
    def shutdown(self):
        """Мгновенное аппаратное обесточивание и торможение (Аварийный стоп)"""
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)

        # 1. МГНОВЕННО ГАСИМ ШПИНДЕЛЬ
        try:
            GPIO.setup(M_MOTOR, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.output(M_MOTOR, GPIO.HIGH)
        except Exception:
            pass

        # 2. МГНОВЕННО БЬЕМ ПО ТОРМОЗАМ (без всяких time.sleep)
        for ax in ['X', 'Y', 'Z']:
            pin = BRAKE_PINS.get(ax)
            if pin:
                try:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.HIGH)
                except Exception:
                    pass

        # 3. МГНОВЕННО ОБЕСТОЧИВАЕМ ДРАЙВЕРА
        for axis in ['X', 'Y', 'Z']:
            if axis in AXES_CONFIG:
                en_pin = AXES_CONFIG[axis]['en']
                try:
                    GPIO.setup(en_pin, GPIO.OUT)
                    GPIO.output(en_pin, GPIO.HIGH)
                except Exception:
                    pass

        # 4. Безопасно отписываемся от прерываний концевиков
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

        # 5. ОЧИЩАЕМ ПАМЯТЬ ВЫБОРОЧНО (ЗАЩИТА ОТ ПАДЕНИЯ ОСИ Z)
        # Мы НЕ чистим пины тормозов, света и E-STOP, чтобы они сохранили свое состояние!
        pins_to_clean = [RED_CRAB, M_MOTOR]
        for ax in ['X', 'Y', 'Z']:
            if ax in AXES_CONFIG:
                cfg = AXES_CONFIG[ax]
                pins_to_clean.extend([cfg['step'], cfg['dir'], cfg['en'], cfg['endstop']])
                
        try:
            GPIO.cleanup(pins_to_clean)
        except Exception:
            pass

    def emergency_stop(self):
        """Полная экстренная остановка всего станка. Вызывается по кнопке или API."""
        logger.warning("АВАРИЙНЫЙ СТОП! Остановка всех систем!")
        self.pause = True
        self.is_homed = False
        self.z_probe_active = False
        self.active_operation = None
        self.interrupted_operation = None
        self.resume_target = None
        
        # Сбрасываем мотор логически
        self.motor_is_on = False
        self.saved_motor_state = False
        
        # Очищаем кэш пробинга
        self.remaining_z_probe_points = []
        self.z_probe_results = []
        self.first_touch_z = None
        
        # Очищаем G-код
        self.gcode_commands = []
        self.gcode_index = 0
        
        # Вызываем аппаратное обесточивание
        self.shutdown()

    def _estop_callback(self, pin):
        """Аппаратное прерывание от физической кнопки E-Stop"""
        if GPIO.getmode() is None:
            return
            
        time.sleep(0.002) 
        
        try:
            low_count = 0
            high_count = 0
            num_reads = 500
            threshold = num_reads // 2
            
            for _ in range(num_reads):
                if GPIO.input(pin) == GPIO.LOW:
                    low_count += 1
                    if low_count > threshold:
                        self.emergency_stop()
                        break
                else:
                    high_count += 1
                    if high_count > threshold:
                        break
                        
        except RuntimeError:
            pass

    def is_estop_pressed(self) -> bool:
        """Возвращает True, если E-Stop кнопка зажата"""
        if GPIO.getmode() is None:
            return False
        try:
            low_count = 0
            high_count = 0
            num_reads = 500
            threshold = num_reads // 2
            
            for _ in range(num_reads):
                if GPIO.input(STOP_BUTTON_PIN) == GPIO.LOW:
                    low_count += 1
                    if low_count > threshold:
                        return True
                else:
                    high_count += 1
                    if high_count > threshold:
                        return False
            return False
        except Exception:
            return False

    def init_global_hardware(self):
        """Инициализация глобальных пинов (Стоп-кнопка) при старте сервера"""
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
        # Настройка E-STOP (подтяжка 3.3v аппаратная, поэтому pull_up_down не нужен)
        try:
            GPIO.setup(STOP_BUTTON_PIN, GPIO.IN)
            GPIO.remove_event_detect(STOP_BUTTON_PIN)
        except Exception:
            pass
            
        # Ловим перепад из HIGH в LOW (момент нажатия)
        GPIO.add_event_detect(STOP_BUTTON_PIN, GPIO.FALLING, callback=self._estop_callback, bouncetime=200)

        # При старте сервера уводим моторы в сон (зажимаем тормоза)
        self.sleep_steppers(['X', 'Y', 'Z'])
            
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
        
        if max_speed <= V_MIN:
            # Если целевая скорость ниже минимальной стартовой, 
            # едем на константной медленной скорости без разгона/торможения
            s_accel = 0
            s_decel = steps
            v_current = max_speed
        else:
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
        self.set_motor_state(False)
        self.saved_motor_state = False
        

        self.wake_up_steppers(['X', 'Y', 'Z'])
        self.sleep_mode = False
        
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
        if max_speed <= V_MIN:
            s_accel = 0
            s_decel = max_delta
            v_current = max_speed
        else:
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
                    if self.red_crab_triggered:
                        # Успели коснуться ровно между проверками! Возвращаем координату штатно.
                        return self.pos['Z'] / self.spm['Z']
                    if self.pause:
                        raise RuntimeError("Z probing interrupted by Pause")
                    
                    raise RuntimeError("Z probing step failed due to unknown reason")
                
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

        # --- ВЫКЛЮЧАЕМ МОТОР ПЕРЕД ПРОБИНГОМ (ОПАСНО ИСКАТЬ ЗЕМЛЮ ВРАЩАЮЩЕЙСЯ ФРЕЗОЙ) ---
        self.set_motor_state(False)
        self.saved_motor_state = False
        
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

    def set_motor_state(self, is_on: bool):
        """Включает (True) или выключает (False) мотор"""
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
        self.motor_is_on = bool(is_on)
        
        # Если станок на паузе, мы просто запоминаем состояние, но физически мотор не крутим
        if self.pause:
            self.saved_motor_state = self.motor_is_on
            return
            
        try:
            # Гарантируем, что пин - это ВЫХОД, даже если его сбросил стоп
            GPIO.setup(M_MOTOR, GPIO.OUT)
            GPIO.output(M_MOTOR, GPIO.LOW if self.motor_is_on else GPIO.HIGH)
        except Exception:
            pass

    def pause_motor(self):
        """Сохраняет состояние и выключает мотор при паузе"""
        self.saved_motor_state = self.motor_is_on
        try:
            GPIO.setup(M_MOTOR, GPIO.OUT)
            GPIO.output(M_MOTOR, GPIO.HIGH) # HIGH - это ВЫКЛ
        except Exception:
            pass

    def resume_motor(self):
        """Восстанавливает работу мотора после снятия с паузы"""
        self.motor_is_on = self.saved_motor_state
        try:
            GPIO.setup(M_MOTOR, GPIO.OUT)
            GPIO.output(M_MOTOR, GPIO.LOW if self.motor_is_on else GPIO.HIGH)
        except Exception:
            pass

    def set_light_state(self, is_on: bool):
        """Управление подсветкой (LOW - включена, HIGH - выключена)"""
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
        self.light_is_on = bool(is_on)
        try:
            GPIO.setup(LED_PIN, GPIO.OUT)
            GPIO.output(LED_PIN, GPIO.LOW if self.light_is_on else GPIO.HIGH)
            logger.info(f"Подсветка {'включена' if self.light_is_on else 'выключена'}")
        except Exception as e:
            logger.error(f"Ошибка управления светом: {e}")

    def set_brakes(self, axes, engage: bool):
        """Управление тормозами для списка осей или одной оси."""
        if isinstance(axes, str):
            axes = [axes]
            
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
        for ax in axes:
            pin = BRAKE_PINS.get(ax)
            if pin:
                try:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.HIGH if engage else GPIO.LOW)
                except Exception as e:
                    logger.error(f"Ошибка тормоза оси {ax}: {e}")

    def wake_up_steppers(self, axes=['X', 'Y', 'Z']):
        """Вывод из сна: подаем ток удержания -> ждем -> отпускаем тормоза -> ждем реле"""
        if isinstance(axes, str): axes = [axes]
        if GPIO.getmode() is None: GPIO.setmode(GPIO.BOARD)
            
        # 1. Подаем ток на драйвера (LOW = ВКЛ)
        for ax in axes:
            if ax in AXES_CONFIG:
                try:
                    GPIO.setup(AXES_CONFIG[ax]['en'], GPIO.OUT)
                    GPIO.output(AXES_CONFIG[ax]['en'], GPIO.LOW)
                except Exception:
                    pass
                    
        time.sleep(0.1) # Быстрая задержка для инициализации драйвера
        
        # 2. Отпускаем механические тормоза (LOW = отпустить)
        self.set_brakes(axes, engage=False)
        time.sleep(0.5)  # Серьезная задержка для механического реле

    def sleep_steppers(self, axes=['X', 'Y', 'Z']):
        """Перевод в сон: зажимаем тормоза -> ждем реле -> снимаем ток удержания"""
        if isinstance(axes, str): axes = [axes]
            
        # 1. Зажимаем механические тормоза (HIGH = зажать)
        self.set_brakes(axes, engage=True)
        time.sleep(0.5) # Ждем, пока реле щелкнет и заблокирует ось
        
        # 2. Снимаем ток с драйверов (HIGH = ВЫКЛ)
        for ax in axes:
            if ax in AXES_CONFIG:
                try:
                    GPIO.setup(AXES_CONFIG[ax]['en'], GPIO.OUT)
                    GPIO.output(AXES_CONFIG[ax]['en'], GPIO.HIGH)
                except Exception:
                    pass

    # ================= ИСПОЛНЕНИЕ G-CODE =================
    def run_gcode(self, commands, v_max=None, a_max=None, is_resume=False):
        if self.pause: raise RuntimeError("Сперва выполните resume")
        if not self.is_homed: raise RuntimeError("Сперва выполните паркинг")

        if not is_resume:
            self.gcode_commands = commands
            self.gcode_index = 0
            self.current_feedrate = 600.0

        self.active_operation = 'gcode'
        v_rapid = v_max if v_max is not None else V_MAX
        a_limit = a_max if a_max is not None else A_MAX

        try:
            while self.gcode_index < len(self.gcode_commands):
                if self.pause: raise RuntimeError("G-code paused")
                
                line = self.gcode_commands[self.gcode_index].strip().upper()
                self.gcode_index += 1
                
                if not line or line.startswith(';') or line.startswith('('):
                    continue

                parts = line.split()
                cmd = parts[0]
                args = {p[0]: float(p[1:]) for p in parts[1:] if p[0] in 'XYZF'}

                if 'F' in args:
                    self.current_feedrate = args['F']

                if cmd == 'G0':
                    x = args.get('X')
                    y = args.get('Y')
                    z = args.get('Z')
                    if x is not None or y is not None or z is not None:
                        self.move_absolute(x=x, y=y, z=z, diagonal=True, v_max=v_rapid, a_max=a_limit)

                elif cmd == 'G1':
                    x = args.get('X')
                    y = args.get('Y')
                    z = args.get('Z')
                    if x is not None or y is not None or z is not None:
                        moving_axes = []
                        if x is not None and abs(x - self.pos['X']/self.spm['X']) > 0.001: moving_axes.append('X')
                        if y is not None and abs(y - self.pos['Y']/self.spm['Y']) > 0.001: moving_axes.append('Y')
                        if z is not None and abs(z - self.pos['Z']/self.spm['Z']) > 0.001: moving_axes.append('Z')

                        if moving_axes:
                            max_spm = max(self.spm[ax] for ax in moving_axes)
                            v_feed = (self.current_feedrate / 60.0) * max_spm
                            if v_feed < V_MIN:
                                v_feed = min(v_feed, v_rapid) # Защита от нулевой скорости
                            else:
                                v_feed = max(v_feed, V_MIN) 
                            self.move_absolute(x=x, y=y, z=z, diagonal=True, v_max=v_feed, a_max=a_limit)

                elif cmd == 'M3':
                    self.set_motor_state(True)
                elif cmd == 'M5':
                    self.set_motor_state(False)

        except Exception:
            if self.pause:
                self.interrupted_operation = 'gcode'
                if self.gcode_index > 0:
                    self.gcode_index -= 1
            raise
        finally:
            if not self.pause:
                self.active_operation = None