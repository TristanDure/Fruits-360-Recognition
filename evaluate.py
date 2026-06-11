"""
Fruits-360 水果识别 — 模型评估
混淆矩阵 + Grad-CAM 热力图
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from model import get_model

# ===== 配置 =====
DATA_DIR = r'E:\code\周记\人工智能\Fruits-360\data\fruits-360_100x100\fruits-360'
MODEL_PATH = 'best_model.pth'
BATCH_SIZE = 64
NUM_CLASSES = 260
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def plot_confusion_matrix(y_true, y_pred, class_names, top_k=30, save_path='confusion_matrix.png'):
    """画混淆矩阵（只显示最容易混淆的 top_k 类）"""
    cm = confusion_matrix(y_true, y_pred)

    # 找错误最多的 top_k 个类别
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                errors.append((cm[i, j], i, j))
    errors.sort(reverse=True)

    # 提取出现的类别
    top_indices = set()
    for _, i, j in errors[:top_k]:
        top_indices.add(i)
        top_indices.add(j)
    top_indices = sorted(list(top_indices))[:top_k]

    # 截取子矩阵
    sub_cm = cm[np.ix_(top_indices, top_indices)]
    sub_names = [class_names[i] for i in top_indices]

    plt.figure(figsize=(18, 14))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=sub_names, yticklabels=sub_names,
                linewidths=0.5, cbar_kws={'label': '样本数'})
    plt.title(f'混淆矩阵 — 最容易混淆的{len(top_indices)}个类别', fontsize=16)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f'混淆矩阵已保存: {save_path}')


def plot_top_errors(y_true, y_pred, class_names, top_n=15, save_path='top_errors.png'):
    """画最容易混淆的类别对"""
    cm = confusion_matrix(y_true, y_pred)
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                errors.append((cm[i, j], class_names[i], class_names[j]))
    errors.sort(reverse=True)

    top = errors[:top_n]
    labels = [f'{t}→{p}' for _, t, p in top]
    values = [v for v, _, _ in top]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(labels)), values, color='#e74c3c', alpha=0.8)
    plt.yticks(range(len(labels)), labels, fontsize=10)
    plt.xlabel('错误样本数', fontsize=12)
    plt.title('最容易混淆的类别对 (Top 15)', fontsize=14)
    plt.gca().invert_yaxis()
    for bar, val in zip(bars, values):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'易混淆类别已保存: {save_path}')


def main():
    print(f'使用设备: {DEVICE}')

    # 加载测试数据
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'Test'), transform=test_transform)
    class_names = test_dataset.classes
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 加载模型
    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f'模型已加载: {MODEL_PATH}')

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 评估报告
    print('\n' + '=' * 60)
    print('分类报告')
    print('=' * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # 画图
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_top_errors(all_labels, all_preds, class_names)

    # 各类准确率排行
    from collections import Counter
    class_correct = Counter()
    class_total = Counter()
    for t, p in zip(all_labels, all_preds):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    accuracies = {class_names[i]: class_correct[i] / class_total[i] * 100
                  for i in range(len(class_names))}
    sorted_acc = sorted(accuracies.items(), key=lambda x: x[1])

    print('\n最容易出错的5类:')
    for name, acc in sorted_acc[:5]:
        print(f'  {name}: {acc:.1f}%')

    print('\n识别最好的5类:')
    for name, acc in sorted_acc[-5:]:
        print(f'  {name}: {acc:.1f}%')

    print('\n评估完成！')


if __name__ == '__main__':
    main()
