import os
os.environ['YOLO_AUTOINSTALL'] = '0'  # отключаем назойливые проверки, если нужно

from ultralytics import YOLO

model = YOLO('nano_last.pt')

# Нативный экспорт в NCNN
model.export(
    half=True,
    format='ncnn',
    simplify=True,      # оптимизирует ONNX-граф перед конвертацией
    opset=11            # NCNN стабильнее всего с opset 11
)