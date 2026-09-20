import os
import random
import time
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SparseSpectralDataset(Dataset):
    """Processed MS/MS spectra with 2,048-bit Morgan fingerprint labels."""

    def __init__(self, h5_file, label_file):
        self.h5_file = h5py.File(h5_file, "r")
        labels = pd.read_excel(label_file)

        def parse_morgan_bits(bits_str):
            bit_indices = eval(bits_str)
            indices = torch.tensor(bit_indices, dtype=torch.long) - 1
            values = torch.ones(len(bit_indices), dtype=torch.float32)
            return torch.sparse_coo_tensor(
                indices.unsqueeze(0),
                values,
                (2048,),
                dtype=torch.float32,
            )

        self.labels = {
            str(row["CAS No."]): parse_morgan_bits(row["Morgan_Bits"])
            for _, row in labels.iterrows()
        }

        # Keep only samples present in both the HDF5 file and label table.
        self.cas_nos = [
            key for key in self.h5_file.keys()
            if key in self.labels
        ]

        self.samples = {key: self.h5_file[key][:] for key in self.cas_nos}

    def __len__(self):
        return len(self.cas_nos)

    def __getitem__(self, idx):
        cas_no = self.cas_nos[idx]
        sample = self.samples[cas_no]

        # Expected shape: [1, 104000] or [104000,].
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # Min-max normalization.
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
                dtype=torch.float32,
            )

        label = self.labels[cas_no]
        return sparse_sample, label.to_dense(), cas_no


def sparse_collate_fn(batch):
    samples, labels, cas_nos = zip(*batch)

    dense_samples = torch.stack([
        x.to_dense() if x.is_sparse else x
        for x in samples
    ])
    dense_labels = torch.stack(labels)

    return dense_samples, dense_labels, list(cas_nos)


class MultiScaleDilatedCNN(nn.Module):
    """Multi-scale dilated CNN without physicochemical constraints."""

    def __init__(self, fingerprint_dim=2048):
        super().__init__()

        # Multi-scale convolution.
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv1_3 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)

        # Dilated convolution.
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(96, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=8, dilation=8),
        ])

        # Downsampling.
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Normalization.
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(64)

        # Input length: 104000.
        self._to_linear = None
        self._get_conv_output((1, 1, 104000))

        # Fully connected layers.
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, fingerprint_dim)

        self.dropout = nn.Dropout(0.3)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = self.forward_features(x)
        self._to_linear = int(torch.numel(x) / x.size(0))

    def forward_features(self, x):
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x))
        x3 = F.relu(self.conv1_3(x))

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn1(x)
        x = self.pool1(x)

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

        x = self.forward_features(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def main():
    start = time.time()

    # Negative-ion MSDCNN settings.
    SEED = 90
    EPOCHS = 200
    BATCH_SIZE = 32

    set_seed(SEED)

    # Negative-ion training data.
    h5_file = "../data/targetC18neg204060revise.h5"
    label_file = "../fingerprint/MorganC18neg.xlsx"

    dataset = SparseSpectralDataset(h5_file, label_file)

    # Use the complete negative-ion training dataset.
    # No internal 300-sample validation subset is created.
    # Independent validation data should be evaluated separately.
    train_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        generator=torch.Generator().manual_seed(SEED),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {SEED}")
    print(f"Training samples: {len(train_set)}")

    model = MultiScaleDilatedCNN(fingerprint_dim=2048).to(device)

    # MSDCNN does not use physicochemical-property constraints.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    os.makedirs("../Result/models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if i == 0 and epoch == 0:
                print(f"Input shape: {x.shape}")
                print(f"Input non-zero elements: {torch.sum(x != 0)}")
                print(f"Label shape: {y.shape}")
                print(f"Label non-zero elements: {torch.sum(y != 0)}")

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save only the final epoch-200 model.
    model_path = "../Result/models/MSDCNN_negative_epoch200.pth"

    torch.save(
        {
            "epoch": EPOCHS,
            "seed": SEED,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        model_path,
    )

    print(f"Model saved: {model_path}")
    print("Training completed.")
    print("Independent validation should be performed with a separate validation dataset.")

    end = time.time()
    print(f"Runtime: {(end - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
