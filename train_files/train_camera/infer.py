import torch
import torch.nn as nn
import numpy as np

# Та самая крутая архитектура
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU()
        )
        
    def forward(self, x):
        return x + self.block(x)

class LensCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, 128), nn.SiLU())
        self.core = nn.Sequential(ResBlock(128), ResBlock(128), ResBlock(128))
        self.head = nn.Linear(128, 2)
        
    def forward(self, x):
        return self.head(self.core(self.embed(x)))

# Грузим сохраненные данные
checkpoint = torch.load('calibrator.pth', weights_only=False)
model = LensCalibrator()
model.load_state_dict(checkpoint['model_state'])
model.eval()

X_mean = checkpoint['x_mean']
X_std = checkpoint['x_std']

def pixels_to_mm(px_x, px_y):
    # Нормализуем вход по тем же параметрам, что и при обучении
    pt = np.array([px_x, px_y])
    norm_pt = (pt - X_mean) / X_std
    
    inputs = torch.FloatTensor([norm_pt])
    
    with torch.no_grad():
        preds = model(inputs)
        
    return preds[0][0].item(), preds[0][1].item()

# Тестируем на рандомных пикселях
test_px_x = 420
test_px_y = 167

mm_x, mm_y = pixels_to_mm(test_px_x, test_px_y)

print(f"Входящие пиксели: X={test_px_x}, Y={test_px_y}")
print(f"Реальные координаты: X={mm_x:.2f} мм, Y={mm_y:.2f} мм")