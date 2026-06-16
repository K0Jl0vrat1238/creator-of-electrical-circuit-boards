from ultralytics import YOLO
def train_model():
    # === КОНФИГУРАЦИЯ ===
    MODEL_NAME = "yolo26n-seg.pt"
    DATA_YAML = "cnc_data.yaml"
    IMG_SIZE = 640          # Максимум для детализации кромок
    EPOCHS = 300
    BATCH_SIZE = -1           # Уменьшен для VRAM + сильной регуляризации
    DEVICE = 0               # GPU индекс

    # Загрузка предобученной модели
    model = YOLO(MODEL_NAME)

    # === ЗАПУСК ОБУЧЕНИЯ ===
    # 💡 СОВЕТ: Просто установи albumentations (pip install albumentations), 
    # и YOLO сама добавит CLAHE-контраст и Blur без изменения кода ниже!
    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        
        # 🎨 АУГМЕНТАЦИИ (выкручены на максимум для эксперимента)
        flipud=0.5,          # Вертикальный флип/миррор
        fliplr=0.5,          # Горизонтальный флип/миррор
        degrees=25,          # Ротация ±25°
        translate=0.2,       # Дрожание/сдвиг по X/Y
        scale=0.6,           # Масштабирование 0.4x–1.6x
        perspective=0.002,   # Перспективные искажения (имитация наклона камеры)
        hsv_h=0.02,          # Сдвиг оттенка (блики, СОЖ, окислы)
        hsv_s=0.5,           # Насыщенность
        hsv_v=0.5,           # Яркость (V в HSV — спасает от бликов и темноты)
        bgr=0.15,            # Баланс белого (вероятность путаницы каналов для симуляции разных камер)
        mosaic=0.6,          # Mosaic-аугментация
        mixup=0.1,           # Mixup (осторожно: может мылить маски)
        copy_paste=0.4,      # Копипаст объектов на новые фоны
        erasing=0.3,         # Random Erasing (стружка, капли, грязь)
        overlap_mask=True,   # Корректная обработка перекрытий
        close_mosaic=30,     # Отключаем mosaic за 15 эпох до конца → четкие границы
        
        # ⚖️ Веса лоссов (seg удалён, Ultralytics балансирует маску автоматически)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 🛠️ ОПТИМИЗАЦИЯ
        optimizer="AdamW",
        lr0=0.0002,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=5,
        patience=0,         # Ранняя остановка по val/seg_loss
        plots=True,
        val=True,
        cache="ram",
        augment=True,
        deterministic=False, # Разрешаем недетерминизм для аугментаций
        seed=42,
        verbose=True
    )

    print("✅ Обучение завершено. Лучшая модель: runs/segment/train/weights/best.pt")

if __name__ == '__main__':
    # Для Windows: предотвращает рекурсивный запуск процессов
    # torch.multiprocessing.freeze_support()  # Раскомментируй, если планируешь .exe
    
    # Запуск обучения
    train_model()