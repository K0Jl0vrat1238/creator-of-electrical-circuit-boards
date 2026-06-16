import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    # 🔧 НАСТРОЙКИ
    MODEL_PATH = "C:/Users/fedor/Desktop/mini-x-large.pt"       # Имя файла твоей обученной модели (должно лежать рядом со скриптом)
    CONF_THRESHOLD = 0.25        # Порог уверенности (0.0 - 1.0)
    SCRIPT_DIR = Path(__file__).parent

    # 1️⃣ Загрузка модели
    model_file = SCRIPT_DIR / MODEL_PATH
    if not model_file.exists():
        print(f"❌ Модель не найдена: {model_file}")
        print("Укажи правильное имя файла в переменной MODEL_PATH")
        return
    
    print(f"⏳ Загрузка модели из {MODEL_PATH}...")
    model = YOLO(str(model_file))
    print("✅ Модель загружена.")

    # 2️⃣ Поиск картинок рядом со скриптом
    img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    img_paths = []
    for ext in img_exts:
        img_paths.extend(list(SCRIPT_DIR.glob(ext)))
    img_paths.sort()  # Сортируем для постоянного порядка

    if not img_paths:
        print("❌ Картинки с поддерживаемыми расширениями не найдены в папке со скриптом.")
        return
    print(f"📷 Найдено изображений: {len(img_paths)}")

    # 3️⃣ Подготовка окна
    win_name = f"YOLO Viewer | d/Space: след., a: пред., q: выход"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    idx = 0
    while True:
        current_path = img_paths[idx]
        print(f"\r🔍 Обработка {idx+1}/{len(img_paths)}: {current_path.name}", end="")

        try:
            # Инференс без вывода логов в консоль
            results = model(str(current_path), conf=CONF_THRESHOLD, verbose=False)
            # results[0].plot() возвращает numpy-массив (BGR) с отрисованными боксами
            annotated_img = results[0].plot()
        except Exception as e:
            print(f"\n⚠️ Ошибка при обработке {current_path.name}: {e}")
            idx = (idx + 1) % len(img_paths)
            continue

        # Адаптивное масштабирование для комфортного просмотра
        h, w = annotated_img.shape[:2]
        max_dim = 1280
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            annotated_img = cv2.resize(annotated_img, (int(w * scale), int(h * scale)))

        cv2.imshow(win_name, annotated_img)
        
        # Ждём нажатия клавиши
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('q'), 27]:          # Q или ESC -> выход
            print("\n👋 Выход...")
            break
        elif key == ord('d') or key == 32: # D или Пробел -> следующая
            idx = (idx + 1) % len(img_paths)
        elif key == ord('a'):              # A -> предыдущая
            idx = (idx - 1) % len(img_paths)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()