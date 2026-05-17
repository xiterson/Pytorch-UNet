import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from torchinfo import summary
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 20,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.25,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    debug_limit = None  # 设为数字限制数据量，设为None使用全量数据
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, limit=debug_limit)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='_mask', limit=debug_limit)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders (num_workers=0 避免multiprocessing冲突)
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # TensorBoard 日志初始化
    writer = SummaryWriter(log_dir='./runs/unet')
    logging.info(f'TensorBoard 日志目录: ./runs/unet')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 记录超参数到 TensorBoard
    writer.add_hparams({
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': learning_rate,
        'val_percent': val_percent,
        'img_scale': img_scale,
        'amp': str(amp),
        'n_train': n_train,
        'n_val': n_val,
    }, metric_dict={})

    # 4. Set up optimizer, scheduler, loss scaler
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # NaN 检测：跳过异常 batch，防止 NaN 级联扩散
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f'Step {global_step}: 检测到 NaN/Inf loss，跳过该 batch')
                    optimizer.zero_grad(set_to_none=True)
                    pbar.update(images.shape[0])
                    global_step += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # TensorBoard: 记录训练 loss
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    try:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {:.4f}'.format(val_score))

                        # TensorBoard: 记录验证指标
                        writer.add_scalar('val/dice', val_score, global_step)
                        writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], global_step)

                        # TensorBoard: 记录预测样本图
                        writer.add_images('val/images', images, global_step)
                        writer.add_images('val/masks_true', true_masks.unsqueeze(1).float(), global_step)
                        pred_mask = masks_pred.argmax(dim=1).unsqueeze(1).float()
                        writer.add_images('val/masks_pred', pred_mask, global_step)

                    except Exception as e:
                        import traceback
                        logging.error(f'Validation round FAILED:\n{traceback.format_exc()}')
                        raise

        # Epoch 级别统计
        avg_epoch_loss = epoch_loss / max(n_train, 1)
        writer.add_scalar('epoch/avg_loss', avg_epoch_loss, epoch)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    writer.close()
    logging.info('TensorBoard writer closed')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 打印详细的模型结构 (torchinfo)
    batch_size = args.batch_size
    input_size = (batch_size, 3, int(480 * args.scale), int(320 * args.scale))
    summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params"], verbose=0)

    # 打印并保存模型结构文本（方便AI分析层尺寸）
    model_text_path = Path('./model_structure.txt')
    with open(model_text_path, 'w') as f:
        f.write(f'=== UNet Model Structure ===\n')
        f.write(f'n_channels={model.n_channels}, n_classes={model.n_classes}, bilinear={model.bilinear}\n')
        f.write(f'Input size: {input_size}\n\n')

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f'Total params: {total_params:,}\n')
        f.write(f'Trainable: {trainable:,}\n\n')
        f.write('--- Layer Details ---\n')

        for name, module in model.named_modules():
            prefix = len(name.split('.')) - 1
            indent = '  ' * max(prefix, 0)
            module_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters())
            f.write(f'{indent}{name or "root"} ({module_type})\n')
            if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                k = getattr(module, 'kernel_size', None)
                s = getattr(module, 'stride', None)
                p = getattr(module, 'padding', None)
                extra = f'  [in={module.in_channels}, out={module.out_channels}'
                if k is not None: extra += f', k={k}'
                if s is not None: extra += f', s={s}'
                if p is not None: extra += f', p={p}'
                extra += ']'
                f.write(f'{indent}  {extra}\n')
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                f.write(f'{indent}  [features={module.num_features}]\n')
            if params > 0:
                f.write(f'{indent}  params: {params:,}\n')
            f.write('\n')

    logging.info(f'Model structure saved to {model_text_path}')
    logging.info(model)
    print(model_text_path.read_text())

    # 使用 torchview 显示 model graph 并保存为文件
    try:
        from torchview import draw_graph
        dummy_input = torch.randn(*input_size)
        graph = draw_graph(model, input_data=dummy_input,
                           expand_nested=True, depth=4, save_graph=True)
        graph_path = str(Path('./model_graph'))
        graph.visual_graph.render(graph_path, format='png', cleanup=True)
        logging.info(f'Model graph saved to {graph_path}.png')
    except Exception as e:
        logging.warning(f'torchview 可视化失败（不影响训练）: {e}')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except Exception as e:
        import traceback
        logging.error(f'=== TRAINING CRASHED ===\n{traceback.format_exc()}')
        raise
