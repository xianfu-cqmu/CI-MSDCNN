# CNN

This directory contains the conventional convolutional neural network (CNN) used in the ablation experiments of the CI-MSDCNN study.

The CNN model serves as a structural baseline for evaluating the contribution of the multi-scale and dilated convolutional architecture used in CI-MSDCNN.

## Model architecture

The model consists of three sequential one-dimensional convolutional layers followed by max-pooling and fully connected layers.

The convolutional layers use:

- 32 channels with a kernel size of 7;
- 64 channels with a kernel size of 5;
- 128 channels with a kernel size of 3.

The extracted spectral features are passed through a fully connected layer with 512 units and then mapped to a **2,048-bit Morgan molecular fingerprint**.

Unlike CI-MSDCNN, this CNN model does not use:

- multi-scale parallel convolution;
- dilated convolution;
- chemistry-informed physicochemical constraints.

## Positive- and negative-ion models

The same CNN architecture and training procedure are used for positive- and negative-ion modes.

Only the corresponding training dataset and random seed differ between the two ion modes:

| Ion mode | Random seed |
|---|---:|
| Positive-ion mode | 100 |
| Negative-ion mode | 90 |

The complete training dataset for each ion mode is used for model fitting. No internal validation subset is split from the training data.

Independent validation datasets are evaluated separately and are not included in model training.

## Training settings

The main training settings are:

- batch size: **32**;
- number of epochs: **200**;
- optimizer: **Adam**;
- initial learning rate: **1 × 10⁻³**;
- loss function: **BCEWithLogitsLoss**;
- dropout: **0.3**;
- output fingerprint dimension: **2,048**.

Only the final **epoch-200** model is saved for each ion mode.

## Usage

### Positive-ion model

```bash
python train_CNN.py --ion-mode positive
```

This uses random seed **100** and the positive-ion training dataset.

### Negative-ion model

```bash
python train_CNN.py --ion-mode negative
```

This uses random seed **90** and the negative-ion training dataset.

The positive- and negative-ion models share exactly the same network architecture and training settings; only the training dataset and random seed differ.

## Model evaluation

The CNN ablation models are evaluated using the same testing and downstream candidate-ranking workflow as the corresponding CI-MSDCNN models.

Therefore, a separate CNN-specific testing script is not provided in this directory.

During evaluation, the CNN architecture and the corresponding CNN checkpoint are loaded in place of the CI-MSDCNN architecture and checkpoint, while the remaining testing and candidate-ranking procedure is unchanged.

Independent validation datasets are kept completely separate from the training datasets.

## Model checkpoints

The CNN checkpoints are generated specifically for ablation analysis and are not distributed as part of the pretrained CI-MSDCNN model release.

The training script provided here reproduces the corresponding positive- and negative-ion CNN models.

The generated checkpoint names are:

```text
CNN_positive_epoch200.pth
CNN_negative_epoch200.pth
```

The pretrained CI-MSDCNN models are available separately at:

https://huggingface.co/Xianfu1993/CI-MSDCNN

DOI:

https://doi.org/10.57967/hf/10530
