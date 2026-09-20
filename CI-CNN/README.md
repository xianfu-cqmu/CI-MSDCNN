# CI-CNN

This directory contains the **chemistry-informed convolutional neural network (CI-CNN)** used in the ablation experiments of the CI-MSDCNN study.

CI-CNN uses the same conventional CNN backbone as the CNN ablation model, while incorporating physicochemical-property constraints during training. It is used to evaluate the contribution of the chemistry-informed training strategy independently of the multi-scale dilated CNN architecture.

## Files

```text
CI-CNN/
├── train_CI_CNN_positive.py
├── train_CI_CNN_negative.py
└── README.md
```

## Model architecture

CI-CNN uses a conventional one-dimensional CNN consisting of three sequential convolutional layers followed by max-pooling and fully connected layers.

The convolutional layers use:

- 32 channels with a kernel size of 7;
- 64 channels with a kernel size of 5;
- 128 channels with a kernel size of 3.

The extracted features are passed through a fully connected layer with 512 units and then mapped to a **2,048-bit Morgan molecular fingerprint**.

A dropout rate of **0.3** is used before the final output layer.

Unlike CI-MSDCNN, CI-CNN does not use multi-scale parallel convolution or dilated convolution.

## Chemistry-informed physicochemical constraints

Three physicochemical properties are considered during model training:

- retention time (**RT**);
- predicted collision cross section (**predicted CCS**);
- predicted basic pKa.

The physicochemical properties are standardized before being incorporated into the training constraint.

The relative physicochemical-property weights used in the final CI-CNN models are:

| Ion mode | RT | predicted CCS | predicted basic pKa |
|---|---:|---:|---:|
| Positive-ion mode | 1 | 2 | 0 |
| Negative-ion mode | 0 | 1 | 3 |

Thus:

- the positive-ion CI-CNN uses RT and predicted CCS as the chemistry-informed constraints;
- the negative-ion CI-CNN uses predicted CCS and predicted basic pKa as the chemistry-informed constraints.

These physicochemical properties are used during model training and are not required during inference.

## Positive- and negative-ion models

The positive- and negative-ion CI-CNN models use the **same CNN architecture and training procedure**.

The differences between the two ion modes are the corresponding training datasets, random seeds and physicochemical-property weights:

| Ion mode | Random seed | RT : predicted CCS : predicted basic pKa | Epochs |
|---|---:|---:|---:|
| Positive-ion mode | 100 | 1 : 2 : 0 | 200 |
| Negative-ion mode | 90 | 0 : 1 : 3 | 200 |

For both ion modes, the **complete corresponding training dataset** is used for model fitting.

No 300-sample internal validation subset is created. Independent validation datasets are kept separate from training and are evaluated independently after model training.

## Training settings

The main training settings are:

- batch size: **32**;
- number of epochs: **200**;
- optimizer: **Adam**;
- initial learning rate: **1 × 10⁻³**;
- base loss function: **BCEWithLogitsLoss**;
- dropout: **0.3**;
- output fingerprint dimension: **2,048**.

For the chemistry-informed loss:

### Positive-ion mode

- physicochemical tolerance: **0.6**;
- similarity scale: **1.0**;
- property weights: **1 : 2 : 0** for RT : predicted CCS : predicted basic pKa.

### Negative-ion mode

- physicochemical tolerance: **0.65**;
- similarity scale: **1.0**;
- property weights: **0 : 1 : 3** for RT : predicted CCS : predicted basic pKa.

Only the final **epoch-200** model is saved for each ion mode.

## Usage

### Positive-ion CI-CNN

```bash
python train_CI_CNN_positive.py
```

The positive-ion model uses random seed **100** and the complete positive-ion training dataset.

The final checkpoint is saved as:

```text
CI_CNN_positive_epoch200.pth
```

### Negative-ion CI-CNN

```bash
python train_CI_CNN_negative.py
```

The negative-ion model uses random seed **90** and the complete negative-ion training dataset.

The final checkpoint is saved as:

```text
CI_CNN_negative_epoch200.pth
```

## Model evaluation

The CI-CNN models are evaluated using the same testing and downstream candidate-ranking workflow as the corresponding CI-MSDCNN models.

Therefore, a separate CI-CNN-specific testing script is not provided in this directory. During evaluation, the CI-CNN architecture and the corresponding CI-CNN checkpoint are loaded in place of the CI-MSDCNN architecture and checkpoint, while the remaining evaluation workflow is unchanged.

Independent validation datasets are kept completely separate from the training datasets.

## Model checkpoints

The CI-CNN checkpoints are generated specifically for ablation analysis and are not distributed as part of the pretrained CI-MSDCNN model release.

The training scripts provided here reproduce the final positive- and negative-ion CI-CNN models.

The pretrained CI-MSDCNN models are available separately at:

https://huggingface.co/Xianfu1993/CI-MSDCNN

DOI:

https://doi.org/10.57967/hf/10530
