"""
Fruits-360 水果识别 — 模型定义
基于 ResNet-18 迁移学习
"""
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=131, freeze_backbone=True):
    """
    加载 ImageNet 预训练的 ResNet-18，替换最后一层适配水果分类

    参数:
        num_classes: 水果类别数 (Fruits-360 = 131)
        freeze_backbone: 是否冻结卷积层 (True=只训练分类头, False=全模型可训练)

    返回:
        model: 修改后的模型
    """
    # 1. 加载预训练模型（已经有 ImageNet 的权重）
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        # 2. 冻结所有卷积层参数（不计算梯度 = 不更新）
        for param in model.parameters():
            param.requires_grad = False

    # 3. 替换最后的全连接层
    # ResNet-18 的 fc 层输入是 512 维，输出原来是 1000 类(ImageNet)
    # 我们改成 131 类(水果)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_layers(model, num_layers_to_unfreeze=1):
    """
    解冻最后几层，用于微调阶段

    ResNet-18 结构: conv1 → bn1 → relu → maxpool
                    → layer1 (2个BasicBlock)
                    → layer2 (2个BasicBlock)
                    → layer3 (2个BasicBlock)
                    → layer4 (2个BasicBlock)
                    → avgpool → fc

    num_layers_to_unfreeze=1: 解冻 layer4 + fc
    num_layers_to_unfreeze=2: 解冻 layer3 + layer4 + fc
    """
    layers_to_unfreeze = []
    if num_layers_to_unfreeze >= 1:
        layers_to_unfreeze.append(model.layer4)
    if num_layers_to_unfreeze >= 2:
        layers_to_unfreeze.append(model.layer3)
    if num_layers_to_unfreeze >= 3:
        layers_to_unfreeze.append(model.layer2)
    if num_layers_to_unfreeze >= 4:
        layers_to_unfreeze.append(model.layer1)

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    # fc 层始终可训练
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
