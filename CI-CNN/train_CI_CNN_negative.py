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
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SparseSpectralDataset(Dataset):
    """Processed MS/MS spectra with Morgan fingerprints and physicochemical properties."""

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

        # Physicochemical properties used by the CI-CNN training constraint.
        self.physchem_cols = [
            "RT",
            "predicted CCS",
            "predicted basic pKa",
        ]

        missing_cols = [
            col for col in self.physchem_cols
            if col not in labels.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing physicochemical columns: {missing_cols}. "
                f"Available columns: {list(labels.columns)}"
            )

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

        self.physchem_values = {}

        for _, row in valid_labels.iterrows():
            cas_no = str(row["CAS No."])
            values = []

            for col in self.physchem_cols:
                normalized_value = (
                    float(row[col]) - self.physchem_mean[col]
                ) / self.physchem_std[col]
                values.append(normalized_value)

            self.physchem_values[cas_no] = values

        # Keep samples that have spectral data, Morgan fingerprints,
        # and complete physicochemical metadata.
        self.cas_nos = [
            key for key in self.h5_file.keys()
            if key in self.labels and key in self.physchem_values
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
        physchem = torch.tensor(
            self.physchem_values[cas_no],
            dtype=torch.float32,
        )

        return sparse_sample, label.to_dense(), physchem, cas_no


def sparse_collate_fn(batch):
    samples, labels, physchems, cas_nos = zip(*batch)

    dense_samples = torch.stack([
        x.to_dense() if x.is_sparse else x
        for x in samples
    ])
    dense_labels = torch.stack(labels)
    dense_physchems = torch.stack(physchems)

    return dense_samples, dense_labels, dense_physchems, list(cas_nos)


class RTCCSpKaWeightedLoss(nn.Module):
    """Chemistry-informed weighted BCE loss for the CI-CNN model."""

    def __init__(
        self,
        base_loss_fn,
        physchem_tolerance=0.65,
        similarity_scale=1.0,
        property_weights=None,
    ):
        super().__init__()

        self.base_loss = base_loss_fn
        self.physchem_tolerance = physchem_tolerance
        self.similarity_scale = similarity_scale

        # Relative weights correspond to
        # [RT, predicted CCS, predicted basic pKa].
        if property_weights is None:
            property_weights = [0.0, 1.0, 3.0]

        property_weights = torch.tensor(
            property_weights,
            dtype=torch.float32,
        )
        property_weights = property_weights / property_weights.sum()
        self.register_buffer("property_weights", property_weights)

    def forward(self, preds, targets, physchems):
        base_loss = self.base_loss(preds, targets)

        physchem_similarity = self.calculate_physchem_similarity(
            physchems
        )

        weights = torch.exp(
            -physchem_similarity / self.physchem_tolerance
        )

        weighted_loss = base_loss * weights
        return weighted_loss.mean()

    def calculate_physchem_similarity(self, physchems):
        physchem_diff = torch.abs(
            physchems.unsqueeze(1) - physchems.unsqueeze(0)
        )

        physchem_similarity_each = torch.exp(
            -physchem_diff / self.similarity_scale
        )

        property_weights = self.property_weights.to(
            physchems.device
        )

        physchem_similarity = torch.sum(
            physchem_similarity_each
            * property_weights.view(1, 1, -1),
            dim=-1,
        )

        return physchem_similarity


class SimpleSpectralCNN(nn.Module):
    """Conventional 1D CNN backbone used in the CI-CNN ablation model."""

    def __init__(self, fingerprint_dim=2048):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, 32, kernel_size=7, stride=2, padding=3
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            32, 64, kernel_size=5, stride=2, padding=2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            64, 128, kernel_size=3, stride=2, padding=1
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self._to_linear = None
        self._get_conv_output((1, 1, 104000))

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, fingerprint_dim)
        self.dropout = nn.Dropout(0.3)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        self._to_linear = int(torch.numel(x) / x.size(0))

    def forward(self, x):
        x = x.to_dense() if x.is_sparse else x

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def main():
    start = time.time()

    # Negative-ion CI-CNN settings.
    SEED = 90
    EPOCHS = 200
    BATCH_SIZE = 32

    set_seed(SEED)

    # Negative-ion training data.
    h5_file = "../data/targetC18neg204060revise.h5"
    label_file = "../fingerprint/MorganC18negPP.xlsx"

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Random seed: {SEED}")
    print(f"Training samples: {len(train_set)}")
    print(f"Physicochemical columns: {dataset.physchem_cols}")

    model = SimpleSpectralCNN(fingerprint_dim=2048).to(device)

    # Relative weights:
    # RT : predicted CCS : predicted basic pKa = 0 : 1 : 3
    base_criterion = nn.BCEWithLogitsLoss()
    criterion = RTCCSpKaWeightedLoss(
        base_criterion,
        physchem_tolerance=0.65,
        similarity_scale=1.0,
        property_weights=[0.0, 1.0, 3.0],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (x, y, physchem, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            physchem = physchem.to(device)

            if i == 0 and epoch == 0:
                print(f"Input shape: {x.shape}")
                print(
                    f"Input non-zero elements: "
                    f"{torch.sum(x != 0)}"
                )
                print(f"Label shape: {y.shape}")
                print(
                    f"Label non-zero elements: "
                    f"{torch.sum(y != 0)}"
                )
                print(f"PhysChem shape: {physchem.shape}")

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y, physchem)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {avg_loss:.4f}"
        )

    # Save only the final epoch-200 model.
    save_dir = "./OfflineModel"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(
        save_dir,
        "CI_CNN_negative_epoch200.pth",
    )

    torch.save(
        {
            "epoch": EPOCHS,
            "seed": SEED,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "physchem_cols": dataset.physchem_cols,
            "physchem_mean": dataset.physchem_mean,
            "physchem_std": dataset.physchem_std,
            "property_weights": [0.0, 1.0, 3.0],
            "physchem_tolerance": 0.65,
            "similarity_scale": 1.0,
        },
        model_path,
    )

    print(f"Model saved: {model_path}")
    print("Training completed.")
    print(
        "Independent validation should be performed "
        "with a separate validation dataset."
    )

    end = time.time()
    print(f"Runtime: {(end - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
