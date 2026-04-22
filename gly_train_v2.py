"""
Deepfake检测模型训练脚本 v2
使用 GlyFusionModelV2 (整合模型，支持直接加载output目录权重)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import argparse
import json
from datetime import datetime

from model.gly_model_v2 import create_model


# ==================== 日志工具 ====================

class Logger:
    """同时输出到终端和日志文件的记录器"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.terminal = sys.stdout
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
            self.file = open(log_file, 'a', encoding='utf-8')
        else:
            self.file = None

    def write(self, message):
        self.terminal.write(message)
        if self.file and message.strip():
            self.file.write(message)
            self.file.flush()

    def flush(self):
        self.terminal.flush()
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# ==================== 数据集定义 ====================

class DeepfakeDataset(Dataset):
    """
    Deepfake数据集
    数据格式：
    - deepfake/0_real/  真实图像 (label=0)
    - deepfake/1_fake/  伪造图像 (label=1)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # 加载真实图像 (label=0)
        real_dir = os.path.join(data_dir, '0_real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((os.path.join(real_dir, img_name), 0))

        # 加载伪造图像 (label=1)
        fake_dir = os.path.join(data_dir, '1_fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((os.path.join(fake_dir, img_name), 1))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"  Real: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Fake: {sum(1 for _, label in self.samples if label == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 数据预处理 ====================

def get_transforms(img_size=224, is_train=True):
    """获取数据预处理变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


# ==================== 评估指标计算 ====================

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==================== 训练函数 ====================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch=0):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = running_loss / len(dataloader)

    return metrics


# ==================== 验证函数 ====================

def validate(model, dataloader, criterion, device, epoch=0):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = running_loss / len(dataloader)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics, all_labels, all_preds, all_probs


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model v2')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='deepfake',
                        help='数据目录路径（包含0_real和1_fake子目录）')

    # 模型参数 - 新的加载方式
    parser.add_argument('--pretrained_path', type=str, default="./output/run_20260413_133506/checkpoints/best_model.pth",
                        help='预训练模型路径（从output目录加载完整模型权重）')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练：加载包含optimizer状态的checkpoint')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像尺寸')

    # 模型结构参数
    parser.add_argument('--freq_dim', type=int, default=512,
                        help='FreqNet输出维度')
    parser.add_argument('--vit_dim', type=int, default=768,
                        help='ViT输出维度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='融合层隐藏维度')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Cross-Attention头数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output2',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备')
    parser.add_argument('--log_file', type=str, default=None,
                        help='日志文件路径')

    # 训练策略
    parser.add_argument('--freeze_backbones', action='store_true',
                        help='冻结backbone参数（只训练融合和分类层）')
    parser.add_argument('--unfreeze_epoch', type=int, default=None,
                        help='在第N个epoch后解冻backbone')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--save_best_only', action='store_true',
                        help='只保存最佳模型')

    args = parser.parse_args()

    # 设置日志记录器
    if args.log_file:
        sys.stdout = Logger(args.log_file)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    print('\n' + '='*50)
    print('Loading datasets...')
    print('='*50)
    full_dataset = DeepfakeDataset(
        args.data_dir,
        transform=get_transforms(args.img_size, is_train=True)
    )

    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 为验证集设置测试变换
    val_dataset.dataset.transform = get_transforms(args.img_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'\nTrain size: {len(train_dataset)}, Val size: {len(val_dataset)}')

    # 创建模型
    print('\n' + '='*50)
    print('Creating model...')
    print('='*50)

    model_kwargs = {
        'freq_dim': args.freq_dim,
        'vit_dim': args.vit_dim,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_classes': 2,
        'dropout': args.dropout,
    }

    # 加载预训练权重（如果提供）
    if args.pretrained_path is not None:
        print(f'Loading pretrained model from: {args.pretrained_path}')
        model = create_model(
            pretrained_path=args.pretrained_path,
            **model_kwargs
        ).to(device)
    else:
        print('Creating model from scratch (no pretrained weights)')
        model = create_model(**model_kwargs).to(device)

    # 恢复训练（如果提供）
    start_epoch = 0
    best_f1 = 0.0
    if args.resume is not None:
        print(f'Resuming training from: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f'Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}')

    # 冻结backbone（可选）
    if args.freeze_backbones:
        model.freeze_backbones()

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Frozen parameters: {total_params - trainable_params:,}')

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 如果有resume，也恢复optimizer状态
    if args.resume is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Optimizer state restored')

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    print('\n' + '='*50)
    print('Starting training...')
    print('='*50)

    history = {'train': [], 'val': []}

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')
        print('-' * 50)

        # 在指定epoch解冻backbone
        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            print(f'Unfreezing backbones at epoch {epoch+1}')
            model.unfreeze_backbones()
            # 重新创建optimizer以包含所有参数
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,  # 使用较小的学习率
                weight_decay=args.weight_decay
            )

        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")

        # 验证
        val_metrics, _, _, _ = validate(model, val_loader, criterion, device, epoch)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"Confusion Matrix:\n{np.array(val_metrics['confusion_matrix'])}")

        # 记录历史
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.6f}')

        # 保存最佳模型
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            model.save_checkpoint(
                os.path.join(checkpoint_dir, 'best_model.pth'),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                metrics=val_metrics
            )
            print(f'Saved best model (F1: {best_f1:.4f})')

        # 保存最新模型（如果不是只保存最佳）
        if not args.save_best_only:
            model.save_checkpoint(
                os.path.join(checkpoint_dir, 'latest_model.pth'),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                metrics=val_metrics
            )

    # 保存训练历史
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print('\n' + '='*50)
    print('Training completed!')
    print(f'Best F1: {best_f1:.4f}')
    print(f'Output directory: {output_dir}')
    print('='*50)

    # 关闭日志文件
    if isinstance(sys.stdout, Logger):
        sys.stdout.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 发生异常时也要关闭日志文件
        if isinstance(sys.stdout, Logger):
            sys.stdout.close()
        raise
