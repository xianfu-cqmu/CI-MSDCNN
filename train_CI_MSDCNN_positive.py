import os
import random
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from openpyxl import Workbook
import torch.nn.functional as F
import sys
sys.path.append('../fingerprint')
from PreAccCal import process_prediction
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.PreAccCal import process_prediction


def SetSeed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# 数据集定义 - 添加 RT / predicted CCS / predicted basic pKa 联合约束支持
# ========================
class SparseSpectralDataset(Dataset):
    def __init__(self, h5_file, label_file):
        self.h5_file = h5py.File(h5_file, 'r')
        labels = pd.read_excel(label_file)

        # ========================
        # 联合约束使用的理化属性
        # 注意：列名必须和 Excel 中完全一致
        # ========================
        self.physchem_cols = ['RT', 'predicted CCS', 'predicted basic pKa']

        missing_cols = [
            col for col in self.physchem_cols
            if col not in labels.columns
        ]

        if missing_cols:
            raise ValueError(
                f"标签文件中缺少以下列: {missing_cols}. "
                f"当前表格列名为: {list(labels.columns)}"
            )

        def parse_morgan_bits(bits_str):
            bit_indices = eval(bits_str)
            indices = torch.tensor(bit_indices, dtype=torch.long) - 1
            values = torch.ones(len(bit_indices), dtype=torch.float32)
            return torch.sparse_coo_tensor(
                indices.unsqueeze(0),
                values,
                (2048,),
                dtype=torch.float32
            )

        self.labels = {
            str(row['CAS No.']): parse_morgan_bits(row['Morgan_Bits'])
            for _, row in labels.iterrows()
        }

        # ========================
        # 提取 RT / predicted CCS / predicted basic pKa，并分别标准化
        # 避免 RT、predicted CCS、predicted basic pKa 数值尺度不同导致某一项主导约束
        # ========================
        valid_labels = labels.dropna(subset=self.physchem_cols).copy()

        self.physchem_mean = {}
        self.physchem_std = {}

        for col in self.physchem_cols:
            values = valid_labels[col].astype(float).values

            mean = np.mean(values) if len(values) > 0 else 0.0
            std = np.std(values) if len(values) > 0 else 1.0

            if std < 1e-8:
                std = 1.0

            self.physchem_mean[col] = mean
            self.physchem_std[col] = std

        # 每个 CAS 对应标准化后的 [RT, predicted CCS, predicted basic pKa]
        self.physchem_values = {}

        for _, row in valid_labels.iterrows():
            cas_no = str(row['CAS No.'])
            values = []

            for col in self.physchem_cols:
                normalized_value = (
                    float(row[col]) - self.physchem_mean[col]
                ) / self.physchem_std[col]
                values.append(normalized_value)

            self.physchem_values[cas_no] = values

        # 只保留 h5 中存在，并且 RT / predicted CCS / predicted basic pKa 都存在的样本
        self.cas_nos = [
            key for key in self.h5_file.keys()
            if key in self.physchem_values
        ]

        self.samples = {key: self.h5_file[key][:] for key in self.cas_nos}

    def __len__(self):
        return len(self.cas_nos)

    def __getitem__(self, idx):
        cas_no = self.cas_nos[idx]
        sample = self.samples[cas_no]  # 原始模型 A：形状应为 [1, 104000] 或 [104000,]

        # 确保样本是二维的 [1, 104000]
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # 添加数据标准化
        min_val = np.min(sample)
        max_val = np.max(sample)
        if max_val > min_val:
            sample = (sample - min_val) / (max_val - min_val)

        row_idx, col_idx = np.nonzero(sample)

        if len(row_idx) == 0:
            sparse_sample = torch.zeros(sample.shape, dtype=torch.float32)
        else:
            indices = np.vstack((row_idx, col_idx))
            values = sample[row_idx, col_idx]
            sparse_sample = torch.sparse_coo_tensor(
                indices,
                values,
                sample.shape,
                dtype=torch.float32
            )

        label = self.labels[cas_no]

        # shape: [3]
        # 顺序为 [RT, predicted CCS, predicted basic pKa]
        physchem = torch.tensor(
            self.physchem_values[cas_no],
            dtype=torch.float32
        )

        return sparse_sample, label.to_dense(), physchem, cas_no


# ========================
# 批处理函数 - 添加 RT / predicted CCS / predicted basic pKa 支持
# ========================
def sparse_collate_fn(batch):
    samples, labels, physchems, cas_nos = zip(*batch)

    dense_samples = torch.stack([
        x.to_dense() if x.is_sparse else x
        for x in samples
    ])

    dense_labels = torch.stack(labels)

    # shape: [batch_size, 3]
    # 三列依次为 [RT, predicted CCS, predicted basic pKa]
    dense_physchems = torch.stack(physchems)

    return dense_samples, dense_labels, dense_physchems, list(cas_nos)


# ========================
# RT / predicted CCS / predicted basic pKa 联合约束加权损失函数
#
# 保持模型 A 原来的 RTWeightedLoss 逻辑：
# 1. 先计算 BCE 基础损失
# 2. 计算 batch 内样本两两之间的理化属性差异
# 3. 差异转相似性
# 4. 相似性转权重
# 5. batch_factor = weights.mean()
# 6. weighted_loss = base_loss * batch_factor
#
# 关键修复：
# property_weights 必须和 physchems 在同一个 device 上
# ========================
class RTCCSpKaWeightedLoss(nn.Module):
    def __init__(
        self,
        base_loss_fn,
        physchem_tolerance=0.6,
        similarity_scale=1.0,
        property_weights=None
    ):
        super().__init__()
        self.base_loss = base_loss_fn
        self.physchem_tolerance = physchem_tolerance
        self.similarity_scale = similarity_scale

        # property_weights 对应 [RT, predicted CCS, predicted basic pKa]，本模型使用 1:2:0
        if property_weights is None:
            property_weights = [1.0, 2.0, 0.0]

        property_weights = torch.tensor(property_weights, dtype=torch.float32)
        property_weights = property_weights / property_weights.sum()

        # register_buffer 会让 property_weights 随 criterion.to(device) 一起移动
        self.register_buffer("property_weights", property_weights)

    def forward(self, preds, targets, physchems):
        # 基础损失（标量）
        base_loss = self.base_loss(preds, targets)

        # physchems: [batch_size, 3]
        # 计算 RT / predicted CCS / predicted basic pKa 差异矩阵: [B, B, 3]
        physchem_diff = self.calculate_physchem_diff(physchems)

        # 第一步：差异 -> 相似性
        # 属性越接近，相似性越大
        physchem_similarity = torch.exp(
            -physchem_diff / self.similarity_scale
        )

        # ========================
        # 关键修复：
        # 确保 property_weights 和 physchems 在同一个设备上
        # ========================
        property_weights = self.property_weights.to(physchems.device)

        # 第二步：RT、predicted CCS 与 predicted basic pKa 的相似性按 1:2:0 加权融合
        # joint_similarity: [B, B]
        joint_similarity = torch.sum(
            physchem_similarity * property_weights.view(1, 1, -1),
            dim=-1
        )

        # 第三步：相似性 -> 权重
        # 保持模型 A 原来的方向：
        # 越相似 -> joint_similarity 越大
        # 越相似 -> 1 - joint_similarity 越小
        # 越相似 -> weights 越大
        weights = torch.exp(
            -(1.0 - joint_similarity) / self.physchem_tolerance
        )

        # batch 级整体调制
        batch_factor = weights.mean()
        weighted_loss = base_loss * batch_factor

        return weighted_loss

    def calculate_physchem_diff(self, physchems):
        # physchems: [batch_size, 3]
        # 输出: [batch_size, batch_size, 3]
        physchem_diff = torch.abs(
            physchems.unsqueeze(1) - physchems.unsqueeze(0)
        )
        return physchem_diff


# ========================
# 计算类别权重
# ========================
def calculate_class_weights(label_file):
    labels_df = pd.read_excel(label_file)
    all_bits = []

    for _, row in labels_df.iterrows():
        bits = eval(row['Morgan_Bits'])
        all_bits.extend(bits)

    # 计算每个位的频率
    bit_counts = np.bincount(all_bits, minlength=2049)[1:]
    class_weights = 1.0 / (bit_counts + 1e-6)
    class_weights = class_weights / np.sum(class_weights)

    return torch.tensor(class_weights, dtype=torch.float32)


# ========================
# 评估指标计算
# ========================
def calculate_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()

    # 计算准确率、精确率、召回率、F1
    tp = (preds_binary * targets).sum(dim=1)
    tn = ((1 - preds_binary) * (1 - targets)).sum(dim=1)
    fp = (preds_binary * (1 - targets)).sum(dim=1)
    fn = ((1 - preds_binary) * targets).sum(dim=1)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'accuracy': accuracy.mean(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean()
    }


# ========================
# 多尺度和空洞卷积网络
# ========================
class MultiScaleDilatedCNN(nn.Module):
    def __init__(self, fingerprint_dim=2048):
        super(MultiScaleDilatedCNN, self).__init__()

        # 多尺度卷积层 - 捕获不同尺度的特征
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv1_3 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)

        # 空洞卷积层 - 扩大感受野而不增加参数量
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(96, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=8, dilation=8)
        ])

        # 下采样层
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 正则化
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(64)

        # ========================
        # 保持模型 A 原始 input 长度：104000
        # 不改成 104001
        # ========================
        self._to_linear = None
        self._get_conv_output((1, 1, 104000))

        # 全连接层
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, fingerprint_dim)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def _get_conv_output(self, shape):
        input = torch.rand(shape)
        output = self.forward_features(input)
        self._to_linear = int(torch.numel(output) / output.size(0))

    def forward_features(self, x):
        # 多尺度卷积
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x))
        x3 = F.relu(self.conv1_3(x))

        # 拼接多尺度特征
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn1(x)
        x = self.pool1(x)

        # 空洞卷积序列
        for i, conv in enumerate(self.dilated_convs):
            x = F.relu(conv(x))
            if i == 0:
                x = self.bn2(x)
                x = self.pool2(x)
            elif i == 1:
                x = self.pool3(x)

        return x

    def forward(self, x):
        x = x.to_dense() if x.is_sparse else x

        # 特征提取
        x = self.forward_features(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':

    start = time.time()
    SEED = 100
    SetSeed(SEED=SEED)

    # ========================
    # 模型保存路径
    # ========================
    save_dir = './OfflineModel'
    os.makedirs(save_dir, exist_ok=True)

    h5_file = "../data/targetC18pos204060.h5"

    # 这个 Excel 中需要包含 RT、predicted CCS、predicted basic pKa 三列
    label_file = "../fingerprint/MorganC18posPP.xlsx"

    # ========================
    # 构建数据加载器
    # ========================
    dataset = SparseSpectralDataset(h5_file, label_file)

    # 使用全部样本进行训练，不再划分额外的验证集
    train_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        generator=torch.Generator().manual_seed(SEED)
    )

    # ========================
    # 初始化模型
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model = MultiScaleDilatedCNN(fingerprint_dim=2048).to(device)

    # 使用 RT + predicted CCS + predicted basic pKa（1:2:0）联合约束的 BCE 损失
    base_criterion = nn.BCEWithLogitsLoss()

    # ========================
    # 关键修复：
    # criterion 加 .to(device)
    # 让 register_buffer 里的 property_weights 移动到 GPU
    # ========================
    criterion = RTCCSpKaWeightedLoss(
        base_criterion,
        physchem_tolerance=0.6,
        similarity_scale=1.0,
        property_weights=[1.0, 2.0, 0.0]  # 对应 [RT, predicted CCS, predicted basic pKa]，比例 1:2:0
    ).to(device)

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # ========================
    # 训练主循环
    # ========================
    epochs = 200

    # 添加调试信息
    print(f"Training samples: {len(train_set)}")

    print(f"PhysChem columns: {dataset.physchem_cols}")
    for col in dataset.physchem_cols:
        print(
            f"{col} mean: {dataset.physchem_mean[col]}, "
            f"{col} std: {dataset.physchem_std[col]}"
        )

    # 创建保存模型的目录
    os.makedirs('../Result/models', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (x, y, physchem, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            physchem = physchem.to(device)

            # 检查输入数据
            if i == 0 and epoch == 0:
                print(f"Input shape: {x.shape}")
                print(f"Input non-zero elements: {torch.sum(x != 0)}")
                print(f"Label shape: {y.shape}")
                print(f"Label non-zero elements: {torch.sum(y != 0)}")
                print(f"PhysChem shape: {physchem.shape}")
                print(f"PhysChem columns: {dataset.physchem_cols}")
                print(f"PhysChem values: {physchem[:5]}")

            optimizer.zero_grad()

            output = model(x)

            # RT + predicted CCS + predicted basic pKa（1:2:0）联合约束
            loss = criterion(output, y, physchem)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 启用 scheduler
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 只保存最终的 epoch 200 模型
        if (epoch + 1) == epochs:
            model_path = (
                f'../Result/models/'
                f'2026C18posRT_pCCS_pKa_1to2to0_premodel_epoch_{epoch + 1}.pth'
            )

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'physchem_cols': dataset.physchem_cols,
                'physchem_mean': dataset.physchem_mean,
                'physchem_std': dataset.physchem_std,
            }, model_path)

            print(f"💾 Model saved at epoch {epoch + 1}")

    print("Training completed.")

    end = time.time()
    print(f"程序运行时间：{(end - start) / 60: .2f}分钟")