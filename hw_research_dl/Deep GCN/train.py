import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import torch.nn.functional as F

from utils import load_data
from utils import plot_loss_with_acc
from utils import accuracy
from model import GCN
from utils import anchor_point
from utils import focal_loss
from utils import loss3

device = "cuda" if torch.cuda.is_available() else "cpu"


# 训练框架
def train(model, optimizer, loss_f,
          tensor_x, tensor_adjacency,
          train_mask, tensor_y, kl_point,
          alpha=0.1, w1=0.01, gamma=2,
          penalty=False, w_gap=1):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(tensor_x, tensor_adjacency, alpha)  # 自动调用前向传播？
    # 损失函数直接在这里定义更改就可
    if loss_f == 'focal_loss':
        loss_train = focal_loss(logits[train_mask], tensor_y[train_mask].to(device),
                                kl_point[train_mask], w1=w1, gamma=gamma,
                                penalty=penalty, w_gap=w_gap)
    else:
        # nn对象不能直接调用，需要先声明定义，之后再调用
        criterion = nn.CrossEntropyLoss()
        loss_train = criterion(logits[train_mask], tensor_y[train_mask])
    acc_train = accuracy(logits[train_mask], tensor_y[train_mask].to(device))
    loss_train.backward()  # 反向传播计算参数的梯度
    optimizer.step()  # 使用优化方法进行梯度更新
    return loss_train.item(), acc_train.item()


# 验证函数
def validate(model, loss_f,
             tensor_x, tensor_adjacency,
             val_mask, tensor_y, kl_point,
             alpha=0.1, w1=0.01, gamma=2,
             penalty=False, w_gap=1):
    model.eval()
    with torch.no_grad():
        # 这个特征这里的错误可他妈坑死我了
        logits, _ = model(tensor_x, tensor_adjacency, alpha)
        if loss_f == 'focal_loss':
            loss_val = focal_loss(logits[val_mask], tensor_y[val_mask].to(device),
                                  kl_point[val_mask], w1=w1, gamma=gamma,
                                penalty=penalty, w_gap=w_gap)
        else:
            criterion = nn.CrossEntropyLoss()
            loss_val = criterion(logits[val_mask], tensor_y[val_mask])
        acc_val = accuracy(logits[val_mask], tensor_y[val_mask].to(device))
        return loss_val.item(), acc_val.item()


# 测试函数
def test(model, best_model, tensor_x, tensor_adjacency, test_mask, tensor_y, alpha):
    # 使用在验证集上最好的模型进行测试
    model.load_state_dict(torch.load(best_model))
    model.eval()
    with torch.no_grad():
        logits, x_repr = model(tensor_x, tensor_adjacency, alpha)
        test_mask_logits = logits[test_mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[test_mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[test_mask].cpu().numpy(), x_repr
