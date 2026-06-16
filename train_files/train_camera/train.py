import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform

# Достаем центры точек из картинки
img = cv2.imread('reddotss.png', cv2.IMREAD_UNCHANGED)
if img.shape[2] == 4:
    mask = img[:, :, 3] > 0
else:
    mask = img[:, :, 2] > 50

mask = mask.astype(np.uint8) * 255
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
points = centroids[1:] 

# Оцениваем базовый шаг сетки в пикселях для умного разделения строк
dists = squareform(pdist(points))
np.fill_diagonal(dists, np.inf)
approx_step = np.median(np.min(dists, axis=1))

# Железобетонная сортировка по сетке (дисторсия больше ничего не сломает!)
sort_y_idx = np.argsort(points[:, 1])
sorted_by_y = points[sort_y_idx]

rows = []
current_row = [sorted_by_y[0]]
for p in sorted_by_y[1:]:
    if p[1] - current_row[-1][1] > approx_step * 0.5:
        rows.append(current_row)
        current_row = [p]
    else:
        current_row.append(p)
rows.append(current_row)

grid_points = []
for j, row in enumerate(rows):
    row = sorted(row, key=lambda p: p[0])
    for i, p in enumerate(row):
        grid_points.append((p[0], p[1], i, j))

grid_points = np.array(grid_points)
points = grid_points[:, :2]
cols_indices = grid_points[:, 2]
rows_indices = grid_points[:, 3]

# Переводим индексы в миллиметры с учетом твоих осей и шага 10 мм
# X идет вправо (+), Y идет вверх (в пикселях вниз — значит минус)
# Левая верхняя точка (i=0, j=0) имеет координаты X:0.0, Y:-5.0
targets_x = cols_indices * 10.0
targets_y = -5.0 - rows_indices * 10.0
targets = np.column_stack((targets_x, targets_y))

# Нормализуем входы
X_mean = points.mean(axis=0)
X_std = points.std(axis=0)
inputs_norm = (points - X_mean) / X_std

inputs_t = torch.FloatTensor(inputs_norm)
targets_t = torch.FloatTensor(targets)

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
        # Тот самый скип-коннекшен, чтобы градиенты не затухали
        return x + self.block(x)

class LensCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        # Расширяем признаки
        self.embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU()
        )
        # Глубокое ядро для сложной нелинейности линзы
        self.core = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )
        # Сужаем обратно в координаты X, Y
        self.head = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.core(x)
        return self.head(x)

model = LensCalibrator()
criterion = nn.HuberLoss(delta=1.0) 
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Обучаем
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(inputs_t)
    loss = criterion(preds, targets_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Эпоха {epoch}, Loss: {loss.item():.4f}")

# Сохраняем модель и стейт нормализации
torch.save({
    'model_state': model.state_dict(),
    'x_mean': X_mean,
    'x_std': X_std
}, 'calibrator.pth')
print("Готово! Модель сохранена.")