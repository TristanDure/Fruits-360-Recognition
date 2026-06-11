"""
Fruits-360 水果识别 — 从 Stage1 checkpoint 继续 Stage2 微调
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from model import get_model, unfreeze_layers

# ===== 配置 =====
DATA_DIR = r'E:\code\周记\人工智能\Fruits-360\data\fruits-360_100x100\fruits-360'
BATCH_SIZE = 64
NUM_CLASSES = 260
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# ===== 数据预处理 =====
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train Loss', color='#e74c3c')
    ax1.plot(history['val_loss'], label='Val Loss', color='#3498db')
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss')
    ax1.set_title('Loss 曲线 (Stage2 微调)'), ax1.legend(), ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_acc'], label='Train Acc', color='#e74c3c')
    ax2.plot(history['val_acc'], label='Val Acc', color='#3498db')
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy 曲线 (Stage2 微调)'), ax2.legend(), ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'训练曲线已保存: {save_path}', flush=True)


def main():
    print('\n===== 加载数据 =====', flush=True)
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'Training'), transform=train_transform)
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'Test'), transform=test_transform)
    class_names = train_dataset.classes
    print(f'训练集: {len(train_dataset)} 张, {len(test_dataset)} 张, {len(class_names)} 类', flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # ===== 加载 Stage1 最优模型 =====
    print('\n===== 加载 Stage1 模型 =====', flush=True)
    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
    state = torch.load('best_model_stage1.pth', map_location=DEVICE)
    model.load_state_dict(state)
    print(f'已加载 best_model_stage1.pth', flush=True)

    # 先验证一下加载的模型
    criterion = nn.CrossEntropyLoss()
    _, stage1_acc, _, _ = validate(model, test_loader, criterion)
    print(f'Stage1 模型验证准确率: {stage1_acc:.2f}%', flush=True)

    # ===== 第二阶段：微调 =====
    print('\n===== 第二阶段：微调 (15 epochs) =====', flush=True)
    model = unfreeze_layers(model, num_layers_to_unfreeze=2)

    optimizer = optim.Adam(
        [{'params': model.fc.parameters(), 'lr': 0.0005},
         {'params': model.layer4.parameters(), 'lr': 0.0001},
         {'params': model.layer3.parameters(), 'lr': 0.00005}]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = stage1_acc

    for epoch in range(15):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        improved = '(best!)' if val_acc > best_acc else ''
        print(f'Epoch {epoch+1:2d}/15 | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% {improved}', flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  → 保存新最优模型: best_model.pth ({best_acc:.2f}%)', flush=True)

    print(f'\n🎉 最终最优准确率: {best_acc:.2f}%', flush=True)
    print(f'模型已保存: best_model.pth', flush=True)

    # ===== 最终评估 =====
    print('\n===== 最终评估 =====', flush=True)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    _, test_acc, all_preds, all_labels = validate(model, test_loader, criterion)

    print(f'\n测试集准确率: {test_acc:.2f}%', flush=True)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    plot_training(history, 'training_curve.png')

    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for i, name in enumerate(class_names):
            f.write(f'{i}: {name}\n')

    print('完成！', flush=True)


if __name__ == '__main__':
    main()
