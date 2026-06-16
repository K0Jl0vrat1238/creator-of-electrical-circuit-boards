import cv2
import time
from ultralytics import YOLO

# === НАСТРОЙКИ ===
MODEL_PATH = "x-large.pt"  # или .engine / .onnx
CAM_SOURCE = "output.mp4"                                     # 0 = веб-камера, или путь к видео/RTSP
IMG_SIZE = 1280                                    # Разрешение инференса
CONF_THRES = 0.45                                  # Порог уверенности
IOU_THRES = 0.45                                   # Подавление дублей

# Загрузка модели (CUDA подхватится автоматически)
print("⏳ Загрузка модели...")
model = YOLO(MODEL_PATH)
print("✅ Модель готова. Запуск камеры...")

# Инициализация камеры
cap = cv2.VideoCapture(CAM_SOURCE)
if not cap.isOpened():
    print("❌ Не удалось открыть камеру/поток. Проверь CAM_SOURCE.")
    exit()

# Опционально: форсируем разрешение камеры (не влияет на imgsz модели)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Счётчик FPS
fps_start = time.time()
fps = 0
frame_count = 0

print("🎥 Инференс запущен. Нажми 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Не удалось захватить кадр. Выход...")
        break

    # 🔍 Инференс
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False,      # Отключаем спам в консоль
        device="cpu"            # Принудительно GPU
    )

    # 🎨 Отрисовка масок, боксов, подписей (встроенный плоттер Ultralytics)
    annotated_frame = results[0].plot()

    # ⏱️ Расчёт FPS
    frame_count += 1
    if time.time() - fps_start >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_start = time.time()

    #  Оверлей FPS
    cv2.putText(annotated_frame, f"FPS: {fps}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # 🖥️ Вывод на экран
    cv2.imshow("YOLO26m-Seg | CNC Realtime Test", annotated_frame)

    # ❌ Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Остановка по команде пользователя.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Сессия завершена.")