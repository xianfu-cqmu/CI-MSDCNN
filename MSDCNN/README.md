# MSDCNN

This directory contains the **multi-scale dilated convolutional neural network (MSDCNN)** used in the ablation experiments of the CI-MSDCNN study.

MSDCNN retains the multi-scale and dilated convolutional architecture of CI-MSDCNN but **does not use chemistry-informed physicochemical-property constraints during training**. It is used to evaluate the contribution of the chemistry-informed component of CI-MSDCNN.

## Files

```text
MSDCNN/
├── train_MSDCNN_positive.py
├── train_MSDCNN_negative.py
└── README.md
```

## Model architecture

MSDCNN uses the same multi-scale dilated CNN backbone as CI-MSDCNN.

The main components are:

- three parallel one-dimensional convolutional branches with kernel sizes of 3, 5 and 7;
- concatenation of the multi-scale convolutional features;
- three sequential dilated convolutional layers with dilation rates of 2, 4 and 8;
- batch normalization and max-pooling layers;
- fully connected layers with 1,024 and 512 hidden units;
- dropout with a rate of 0.3;
- a final output layer predicting a **2,048-bit Morgan molecular fingerprint**.

Unlike CI-MSDCNN, MSDCNN uses the standard `BCEWithLogitsLoss` and does not incorporate RT, predicted CCS or predicted basic pKa as training constraints.

## Positive- and negative-ion models

The positive- and negative-ion MSDCNN models use the **same network architecture and training procedure**.

The differences between the two ion modes are limited to the corresponding training datasets and random seeds:

| Ion mode | Random seed | Epochs |
|---|---:|---:|
| Positive-ion mode | 100 | 200 |
| Negative-ion mode | 90 | 200 |

For both ion modes, the **complete corresponding training dataset** is used for model fitting.

No 300-sample internal validation subset is created. Independent validation datasets are kept separate from training and are evaluated independently after model training.

## Training settings

The main training settings are:

- batch size: **32**;
- number of epochs: **200**;
- optimizer: **Adam**;
- initial learning rate: **1 × 10⁻³**;
- weight decay: **1 × 10⁻⁵**;
- loss function: **BCEWithLogitsLoss**;
- dropout: **0.3**;
- output fingerprint dimension: **2,048**.

A `ReduceLROnPlateau` scheduler is used with:

- mode: `min`;
- factor: `0.5`;
- patience: `5`.

Only the final **epoch-200** model is saved for each ion mode.

## Usage

### Positive-ion MSDCNN

```bash
python train_MSDCNN_positive.py
```

The positive-ion model uses random seed **100** and the complete positive-ion training dataset.

The final checkpoint is saved as:

```text
MSDCNN_positive_epoch200.pth
```

### Negative-ion MSDCNN

```bash
python train_MSDCNN_negative.py
```

The negative-ion model uses random seed **90** and the complete negative-ion training dataset.

The final checkpoint is saved as:

```text
MSDCNN_negative_epoch200.pth
```

## Model evaluation

The MSDCNN models are evaluated using the same testing and downstream candidate-ranking workflow as the corresponding CI-MSDCNN models.

Therefore, a separate MSDCNN-specific testing script is not required in this directory. During evaluation, the MSDCNN architecture and the corresponding MSDCNN checkpoint are loaded in place of the CI-MSDCNN architecture and checkpoint, while the remaining evaluation workflow is unchanged.

Independent validation datasets are kept completely separate from the training datasets.

## Model checkpoints

The MSDCNN checkpoints are generated specifically for ablation analysis and are not distributed as part of the pretrained CI-MSDCNN model release.

The training scripts provided here reproduce the final positive- and negative-ion MSDCNN models.

The pretrained CI-MSDCNN models are available separately at:

https://huggingface.co/Xianfu1993/CI-MSDCNN

DOI:

https://doi.org/10.57967/hf/10530
