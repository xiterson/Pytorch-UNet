"""诊断 NaN 来源"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from pathlib import Path
from utils.dice_score import dice_loss
import logging

logging.basicConfig(level=logging.INFO)

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')

# 1. 检查数据集
print('=== 数据集检查 ===')
try:
    ds = CarvanaDataset(dir_img, dir_mask, scale=0.5)
    print(f'CarvanaDataset OK, len={len(ds)}, mask_values={ds.mask_values}')
except Exception as e:
    print(f'CarvanaDataset 失败: {e}')
    try:
        ds = BasicDataset(dir_img, dir_mask, scale=0.5, mask_suffix='_mask')
        print(f'BasicDataset OK, len={len(ds)}, mask_values={ds.mask_values}')
    except Exception as e2:
        print(f'BasicDataset 也失败: {e2}')

# 2. 取一个batch检查数值
print('\n=== 数据数值检查 ===')
sample = ds[0]
img = sample['image']
mask = sample['mask']
print(f'image: shape={img.shape}, dtype={img.dtype}, min={img.min():.4f}, max={img.max():.4f}, has_nan={torch.isnan(img).any()}')
print(f'mask: shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}, has_nan={torch.isnan(mask).any()}')
print(f'mask unique values: {torch.unique(mask).tolist()}')

# 3. 单步前向+反向传播测试
print('\n=== 前向/反向传播测试 ===')
device = 'cuda'
model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
model.train()

x = img.unsqueeze(0).to(device)
y = mask.unsqueeze(0).to(device)

print('输入无NaN:', not torch.isnan(x).any())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)

with torch.amp.autocast(device, enabled=True):
    out = model(x)
    print(f'output: has_nan={torch.isnan(out).any()}, min={out.min().item():.4f}, max={out.max().item():.4f}')

    loss = criterion(out, y)
    print(f'ce_loss: {loss.item():.6f}, has_nan={torch.isnan(loss).item()}')

    d_loss = dice_loss(F.softmax(out, dim=1).float(), F.one_hot(y, 2).permute(0, 3, 1, 2).float(), multiclass=True)
    print(f'dice_loss: {d_loss.item():.6f}, has_nan={torch.isnan(d_loss).item()}')

    total_loss = loss + d_loss
    print(f'total_loss: {total_loss.item():.6f}')

optimizer.zero_grad(set_to_none=True)
grad_scaler = torch.amp.GradScaler(enabled=True)
grad_scaler.scale(total_loss).backward()

# 检查梯度
has_nan_grad = False
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f'NaN gradient in {name}!')
        has_nan_grad = True
if not has_nan_grad:
    print('所有梯度正常，无NaN')

grad_scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
grad_scaler.step(optimizer)
grad_scaler.update()

# 检查更新后权重
first_w = list(model.parameters())[0]
print(f'\n更新后第一个权重: mean={first_w.mean():.6f}, has_nan={torch.isnan(first_w).any()}')
