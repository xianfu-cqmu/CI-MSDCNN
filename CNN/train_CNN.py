import os
import random
import argparse
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
    """Load processed MS/MS spectra and 2,048-bit Morgan fingerprints."""

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

        # Keep samples that are present in both the HDF5 file and label table.
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

        # Expected input shape: [1, 104000] or [104000,].
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
        x.to_dense() if x.is_sparse else x for x in samples
    ])
    dense_labels = torch.stack(labels)
    return dense_samples, dense_labels, list(cas_nos)


class SimpleSpectralCNN(nn.Module):
    """Conventional 1D CNN used as the CNN ablation model."""

    def __init__(self, fingerprint_dim=2048):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
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
    parser = argparse.ArgumentParser(
        description="Train the conventional CNN ablation model."
    )
    parser.add_argument(
        "--ion-mode",
        choices=["positive", "negative"],
        default="positive",
        help="Ion mode to train. The architecture is identical; only the training dataset and seed differ.",
    )
    args = parser.parse_args()

    # Positive- and negative-ion CNN models use the same architecture and
    # training settings. Only the training dataset and random seed differ.
    config = {
        "positive": {
            "seed": 100,
            "h5_file": "../data/targetC18pos204060.h5",
            "label_file": "../fingerprint/MorganC18poswithoutRT.xlsx",
        },
        "negative": {
            "seed": 90,
            "h5_file": "../data/targetC18neg204060revise.h5",
            "label_file": "../fingerprint/MorganC18negPP.xlsx",
        },
    }

    ion_mode = args.ion_mode
    seed = config[ion_mode]["seed"]
    h5_file = config[ion_mode]["h5_file"]
    label_file = config[ion_mode]["label_file"]

    set_seed(seed)

    save_dir = "./OfflineModel"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Ion mode: {ion_mode}")
    print(f"Random seed: {seed}")
    print(f"H5 file: {h5_file}")
    print(f"Label file: {label_file}")

    dataset = SparseSpectralDataset(h5_file, label_file)

    # Use the complete training dataset.
    # Independent validation data are evaluated separately and are not included
    # in this training script.
    train_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleSpectralCNN(fingerprint_dim=2048).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    print(f"Training samples: {len(train_set)}")

    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save only the final epoch-200 model.
    model_path = os.path.join(
        save_dir,
        f"CNN_{ion_mode}_epoch200.pth",
    )

    torch.save(
        {
            "epoch": epochs,
            "ion_mode": ion_mode,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        model_path,
    )

    print(f"Model saved: {model_path}")
    print("Training completed.")


if __name__ == "__main__":
    main()
