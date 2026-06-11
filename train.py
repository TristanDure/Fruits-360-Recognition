"""
Fruits-360 水果识别 — 训练脚本
两阶段训练：冻结训练 → 微调
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # 非GUI模式
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import get_model, unfreeze_layers

# ===== 配置 =====
DATA_DIR = r'E:\code\周记\人工智能\Fruits-360\data\fruits-360_100x100\fruits-360'
BATCH_SIZE = 64
NUM_CLASSES = 260
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# ===== 数据预处理 =====
# 训练集：数据增强（模拟不同拍摄角度/光照）
train_transform = transforms.Compose([
    transforms.Resize(256),                    # 先放大
    transforms.RandomResizedCrop(224),         # 随机裁剪（最重要的增强）
    transforms.RandomHorizontalFlip(),         # 水平翻转
    transforms.RandomRotation(20),             # 随机旋转 ±20°
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # 颜色抖动
    transforms.ToTensor(),                     # 转Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试集：只做基础处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== 定义训练函数 =====
def train_one_epoch(model, loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()          # 清空梯度
        outputs = model(images)        # ① 前向：猜答案
        loss = criterion(outputs, labels)  # ② 算差距
        loss.backward()                # ③ 反向：算梯度
        optimizer.step()               # ④ 更新参数

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
    """验证"""
    model.eval()
    running_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels


def plot_training(history, save_path='training_curve.png'):
    """画训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', color='#e74c3c')
    ax1.plot(history['val_loss'], label='Val Loss', color='#3498db')
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss')
    ax1.set_title('Loss 曲线'), ax1.legend(), ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train Acc', color='#e74c3c')
    ax2.plot(history['val_acc'], label='Val Acc', color='#3498db')
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy 曲线'), ax2.legend(), ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'训练曲线已保存: {save_path}')


# ===== 主流程 =====
def main():
    # 1. 加载数据
    print('\n===== 加载数据 =====')
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'Training'), transform=train_transform)
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'Test'), transform=test_transform)

    # 类别名映射
    class_names = train_dataset.classes
    print(f'训练集: {len(train_dataset)} 张, {len(class_names)} 类')
    print(f'测试集: {len(test_dataset)} 张')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 2. ===== 第一阶段：冻结训练（只训练分类头） =====
    print('\n===== 第一阶段：冻结训练 =====')
    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1:2d}/10 | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_stage1.pth')

    print(f'\n第一阶段最优准确率: {best_acc:.2f}%')

    # 3. ===== 第二阶段：微调 =====
    print('\n===== 第二阶段：微调 =====')
    model = unfreeze_layers(model, num_layers_to_unfreeze=2)  # 解冻 layer3+layer4

    # 小学习率（不能破坏预训练好的权重）
    optimizer = optim.Adam(
        [{'params': model.fc.parameters(), 'lr': 0.0005},
         {'params': model.layer4.parameters(), 'lr': 0.0001},
         {'params': model.layer3.parameters(), 'lr': 0.00005}]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(15):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1:2d}/15 | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'\n🎉 最终最优准确率: {best_acc:.2f}%')
    print(f'模型已保存: best_model.pth')

    # 4. ===== 评估 =====
    print('\n===== 最终评估 =====')
    model.load_state_dict(torch.load('best_model.pth'))
    _, test_acc, all_preds, all_labels = validate(model, test_loader, criterion)

    print(f'\n测试集准确率: {test_acc:.2f}%')
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # 保存训练曲线
    plot_training(history, 'training_curve.png')

    # 保存类别映射
    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for i, name in enumerate(class_names):
            f.write(f'{i}: {name}\n')

    print('完成！')


if __name__ == '__main__':
    main()
