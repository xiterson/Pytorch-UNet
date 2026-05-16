"""诊断脚本：定位训练退出原因"""
import torch
import gc
import logging

logging.basicConfig(level=logging.INFO)

print('=== 环境诊断 ===')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'显存总量: {round(total_mem, 1)} GB')

# 测试模型+单batch的显存占用
print('\n=== 显存测试 ===')
x = torch.randn(1, 3, 240, 160, device='cuda', dtype=torch.float32)
y = torch.randint(0, 2, (1, 240, 160), device='cuda', dtype=torch.long)
print(f'输入后: {round(torch.cuda.memory_allocated() / 1024**2, 0)} MB')

from unet import UNet
model = UNet(n_channels=3, n_classes=2).to('cuda').float()
print(f'模型后: {round(torch.cuda.memory_allocated() / 1024**2, 0)} MB')

with torch.amp.autocast('cuda', enabled=True):
    out = model(x)
    print(f'前向传播后: {round(torch.cuda.memory_reserved() / 1024**2, 0)} MB')
    loss = torch.nn.CrossEntropyLoss()(out, y)
    loss.backward()
    print(f'反向传播后: {round(torch.cuda.memory_reserved() / 1024**2, 0)} MB')

del x, y, out, loss
gc.collect()
torch.cuda.empty_cache()
peak_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f'\n清理后: {round(torch.cuda.memory_allocated() / 1024**2, 0)} MB')
print(f'峰值显存: {round(peak_mem, 0)} MB / {round(total_mem * 1024, 0)} MB')
print(f'显存使用率: {round(peak_mem / (total_mem * 1024) * 100, 1)}%')
