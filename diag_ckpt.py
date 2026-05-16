import torch

print('=== 作者的模型 ===')
sd1 = torch.load('/code/Pytorch-UNet/model/unet_carvana_scale0.5_epoch2.pth', map_location='cpu', weights_only=False)
print('mask_values:', sd1.get('mask_values'))

print('\n=== 自己训练 epoch4 ===')
sd2 = torch.load('/code/Pytorch-UNet/checkpoints/checkpoint_epoch4.pth', map_location='cpu', weights_only=False)
print('mask_values:', sd2.get('mask_values'))

# 检查epoch之间是否有差异
print('\n=== Epoch间差异 ===')
keys = [k for k in sd1.keys() if k != 'mask_values']
for i in [1, 2, 3, 4]:
    sd = torch.load(f'/code/Pytorch-UNet/checkpoints/checkpoint_epoch{i}.pth', map_location='cpu', weights_only=False)
    w = sd[keys[0]]
    print(f'Epoch{i} {keys[0][:30]}: mean={w.mean():.6f} std={w.std():.6f}')

w1 = sd1[keys[0]]
print(f'Author  {keys[0][:30]}: mean={w1.mean():.6f} std={w1.std():.6f}')
