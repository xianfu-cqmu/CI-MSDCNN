import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
from openpyxl import Workbook
from train_CI_MSDCNN_negative import MultiScaleDilatedCNN   # 确保导入你的模型定义

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.PreAccCal import process_prediction


class SparseSpectralDataset(Dataset):
    def __init__(self, h5_file, label_file=None, mode='train'):
        """
        mode: 'train' 或 'test'
        如果 mode='train'，需要传入 label_file 并返回 (sample, label, cas_no)
        如果 mode='test'，忽略 label_file，只返回 (sample, cas_no)
        """
        self.mode = mode
        self.h5_file = h5py.File(h5_file, 'r')
        self.cas_nos = list(self.h5_file.keys())
        # 只在 train 模式下读取并解析 labels
        if self.mode == 'train':
            assert label_file is not None, "Train mode requires label_file"
            labels_df = pd.read_excel(label_file)
            def parse_morgan_bits(bits_str):
                bit_indices = eval(bits_str)
                indices = torch.tensor([bit_indices], dtype=torch.long) - 1
                values = torch.ones(len(bit_indices), dtype=torch.float32)
                return torch.sparse_coo_tensor(indices, values, (2048,), dtype=torch.float32)
            self.labels = {
                str(row['CAS No.']): parse_morgan_bits(row['Morgan_Bits'])
                for _, row in labels_df.iterrows()
            }

    def __len__(self):
        return len(self.cas_nos)

    def __getitem__(self, idx):
        cas_no = self.cas_nos[idx]
        sample_np = self.h5_file[cas_no][:]

        # 构造稀疏张量
        non_zero = np.nonzero(sample_np)
        indices = torch.tensor(np.vstack(non_zero), dtype=torch.int64)
        values  = torch.tensor(sample_np[non_zero], dtype=torch.float32)
        sparse_sample = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=sample_np.shape,
            dtype=torch.float32
        )

        if self.mode == 'train':
            label = self.labels[cas_no].to_dense()
            return sparse_sample, label, cas_no
        else:  # test 模式
            return sparse_sample, cas_no


def sparse_collate_fn(batch):
    """
    A collate function that handles both train (sample, label, cas_no)
    and test (sample, cas_no) modes automatically.
    """
    # Detect mode by tuple size
    first = batch[0]
    if len(first) == 3:
        # —— TRAIN 模式 —— batch entries are (sample, label, cas_no)
        sparse_samples = []
        dense_labels = []
        cas_nos = []

        for sample, label, cas_no in batch:
            sparse_samples.append(sample)
            dense_labels.append(label)
            cas_nos.append(cas_no)

        # 将稀疏张量转换为稠密并堆叠
        sparse_batch = torch.stack([x.to_dense() for x in sparse_samples])
        dense_labels = torch.stack(dense_labels)

        return sparse_batch, dense_labels, cas_nos

    elif len(first) == 2:
        # —— TEST 模式 —— batch entries are (sample, cas_no)
        sparse_samples = []
        cas_nos = []

        for sample, cas_no in batch:
            sparse_samples.append(sample)
            cas_nos.append(cas_no)

        # 如果需要，可以将测试样本也转换为一个大 batch 的稠密张量：
        sparse_batch = torch.stack([x.to_dense() for x in sparse_samples])
        return sparse_batch, cas_nos

    else:
        raise RuntimeError(f"Unexpected batch element size: {len(first)}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 测试HDF5文件所在路径，并进行读取
    TestFile = "../data/qtofnegspikeplasma94.h5"
    test_ds = SparseSpectralDataset(TestFile, mode='test')
    Test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=sparse_collate_fn)
    print('数据加载完毕！')

    # 1. 创建模型实例
    model = MultiScaleDilatedCNN().to(device)

    # 2. 加载权重（以最优模型为例）
    checkpoint = torch.load("../Result/models/CI-MSDCNN_negative.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('模型加载完毕！')
    model.eval()  # 切换到推理模式
    results = []
    with torch.no_grad():
        for samples, cas_nos in Test_loader:
            samples = samples.to(device)
            outputs = model(samples)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).int().cpu().numpy()

            for cas_no, output in zip(cas_nos, outputs):
                # 将预测结果（1的位置转换为索引列表）并加1恢复到原索引编号
                non_zero_indices = np.where(output == 1)[0] + 1
                results.append([cas_no, str(non_zero_indices.tolist())])

    # 将预测结果保存为Excel文件
    save_path = '../Result/qtofnegspikeplasma94.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.title = "GBMpos"   # 这里是新增的最小改动
    ws.append(["CAS No.", "Predicted Morgan Fingerprint"])
    for result in results:
        ws.append(result)
    wb.save(save_path)

    result = process_prediction("../Result/qtofnegspikeplasma94.xlsx")
    # print(f"百分比：{result['percentage']: .4f}, 分母：{result['denominator']: .4f}, 分子：{result['numerator']: .4f}")

    # 3. （可选）如果要继续训练，也可以恢复 optimizer 状态
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']