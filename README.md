# CI-MSDCNN

**Chemistry-informed multi-scale dilated convolutional neural network for metabolite identification from MS/MS spectra**

CI-MSDCNN is a deep-learning framework for metabolite identification from tandem mass spectrometry (MS/MS) data. The model learns fragmentation patterns from MS/MS spectra to predict molecular fingerprints and incorporates complementary physicochemical properties as chemistry-informed constraints during training. Physicochemical properties are used during training only; model inference requires MS/MS spectra alone.

## Repository structure

This repository contains the full CI-MSDCNN model together with three ablation models used to evaluate the contributions of the network architecture and chemistry-informed training strategy.

```text
CI-MSDCNN/
├── README.md
│
├── CI-MSDCNN/
│   ├── README.md
│   ├── train_CI_MSDCNN_positive.py
│   ├── train_CI_MSDCNN_negative.py
│   ├── test_CI_MSDCNN_positive.py
│   └── test_CI_MSDCNN_negative.py
│
├── MSDCNN/
│   ├── README.md
│   ├── train_MSDCNN_positive.py
│   └── train_MSDCNN_negative.py
│
├── CNN/
│   ├── README.md
│   └── train_CNN.py
│
└── CI-CNN/
    ├── README.md
    ├── train_CI_CNN_positive.py
    └── train_CI_CNN_negative.py
```

### Models

- **`CI-MSDCNN/`** — full chemistry-informed multi-scale dilated CNN.
- **`MSDCNN/`** — multi-scale dilated CNN without chemistry-informed physicochemical constraints.
- **`CNN/`** — conventional CNN baseline without chemistry-informed constraints.
- **`CI-CNN/`** — conventional CNN with chemistry-informed physicochemical constraints.

Each model directory contains a dedicated README describing its architecture, training configuration and usage.

## CI-MSDCNN

The full CI-MSDCNN model combines:

- multi-scale one-dimensional convolution;
- dilated convolution for enlarged receptive fields;
- prediction of 2,048-bit Morgan molecular fingerprints;
- chemistry-informed physicochemical constraints during training.

The physicochemical properties used by the final CI-MSDCNN models are:

| Ion mode | RT | predicted CCS | predicted basic pKa |
|---|---:|---:|---:|
| Positive-ion mode | 1 | 2 | 0 |
| Negative-ion mode | 0 | 1 | 3 |

The physicochemical information is incorporated during model training only and is not required during inference.

## Training and validation

Positive- and negative-ion models are trained separately.

For the final models:

- positive-ion random seed: **100**;
- negative-ion random seed: **90**;
- batch size: **32**;
- training epochs: **200**;
- optimizer: **Adam**;
- initial learning rate: **1 × 10⁻³**;
- weight decay: **1 × 10⁻⁵** for CI-MSDCNN and MSDCNN;
- output fingerprint dimension: **2,048**.

The complete corresponding training dataset is used for model fitting. No 300-sample internal validation subset is created. Independent validation datasets are kept separate from training and are evaluated independently after model training.

Only the final **epoch-200** checkpoint is retained for the final training workflow.

## Pretrained CI-MSDCNN models

Pretrained positive- and negative-ion CI-MSDCNN models are available on Hugging Face:

https://huggingface.co/Xianfu1993/CI-MSDCNN

Persistent DOI:

https://doi.org/10.57967/hf/10530

## Environment

CI-MSDCNN was implemented in:

- Python **3.12.2**
- PyTorch **2.5.1**
- CUDA **11.8**

Training and evaluation were performed on a GPU-enabled high-performance computing cluster managed by Slurm.

Main Python dependencies include:

```text
torch
numpy
pandas
h5py
openpyxl
```

## Usage

For the full model, enter the `CI-MSDCNN/` directory and run the corresponding positive- or negative-ion training/testing script.

Example:

```bash
cd CI-MSDCNN
python train_CI_MSDCNN_positive.py
```

or

```bash
python train_CI_MSDCNN_negative.py
```

Detailed instructions for each model are provided in the README within the corresponding subdirectory.

## Source code

Repository:

https://github.com/xianfu-cqmu/CI-MSDCNN

## Citation

Citation information will be updated upon publication.
