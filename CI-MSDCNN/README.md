# CI-MSDCNN

**CI-MSDCNN** (chemistry-informed multi-scale dilated convolutional neural network) is a deep-learning framework for metabolite identification from tandem mass spectrometry (MS/MS) data.

CI-MSDCNN learns fragmentation patterns from MS/MS spectra to predict molecular fingerprints and incorporates complementary physicochemical properties as chemistry-informed constraints during model training. These physicochemical properties are used during training only; model inference requires MS/MS data only.

This repository provides the training and testing code for the positive- and negative-ion CI-MSDCNN models.

## Repository structure

```text
CI-MSDCNN/
├── train_CI_MSDCNN_positive.py
├── train_CI_MSDCNN_negative.py
├── test_CI_MSDCNN_positive.py
├── test_CI_MSDCNN_negative.py
└── README.md
```

### Training scripts

- `train_CI_MSDCNN_positive.py`  
  Training script for the positive-ion CI-MSDCNN model.

- `train_CI_MSDCNN_negative.py`  
  Training script for the negative-ion CI-MSDCNN model.

### Testing scripts

- `test_CI_MSDCNN_positive.py`  
  Testing script for the positive-ion CI-MSDCNN model.

- `test_CI_MSDCNN_negative.py`  
  Testing script for the negative-ion CI-MSDCNN model.

## Model overview

CI-MSDCNN uses multi-scale one-dimensional convolution and dilated convolution to learn fragmentation patterns from processed MS/MS spectra.

The model predicts a **2,048-bit Morgan molecular fingerprint** for each input spectrum. Separate models are trained for positive- and negative-ion data.

The main network components include:

- parallel one-dimensional convolutional layers with multiple kernel sizes;
- dilated convolutional layers for multi-scale spectral feature extraction;
- batch-normalization and max-pooling layers;
- fully connected layers for molecular fingerprint prediction.

## Chemistry-informed physicochemical constraints

Three physicochemical properties are considered during model training:

- retention time (**RT**);
- predicted collision cross section (**predicted CCS**);
- predicted basic pKa.

The physicochemical properties are standardized before being incorporated into the training constraint.

The relative weights used in the final models are:

| Ion mode | RT | predicted CCS | predicted basic pKa |
|---|---:|---:|---:|
| Positive-ion mode | 1 | 2 | 0 |
| Negative-ion mode | 0 | 1 | 3 |

Thus, the final positive-ion model uses RT and predicted CCS as physicochemical constraints, whereas the final negative-ion model uses predicted CCS and predicted basic pKa.

These physicochemical properties are used **only during model training** and are **not required during inference**.

## Model training

The positive- and negative-ion CI-MSDCNN models were trained separately.

For the final models, the complete corresponding training datasets were used for model fitting. Independent validation datasets were used separately for model evaluation and were not included in model training.

Main training settings:

- batch size: **32**;
- number of epochs: **200**;
- optimizer: **Adam**;
- initial learning rate: **1 × 10⁻³**;
- weight decay: **1 × 10⁻⁵**;
- output fingerprint dimension: **2,048**.

The trained models are saved as PyTorch `.pth` files.

## Model evaluation

Independent validation datasets were used to evaluate the trained models.

The validation datasets were kept separate from the training datasets and were not used for model optimization.

During inference, only the processed MS/MS spectrum is required. The physicochemical properties used as training constraints are not required for prediction.

## Pretrained models

The pretrained CI-MSDCNN model weights are publicly available on Hugging Face:

https://huggingface.co/Xianfu1993/CI-MSDCNN

The repository contains:

```text
CI-MSDCNN_positive.pth
CI-MSDCNN_negative.pth
```

corresponding to the positive- and negative-ion CI-MSDCNN models, respectively.

### Model DOI

The pretrained models are archived with a persistent Digital Object Identifier (DOI):

**https://doi.org/10.57967/hf/10530**

DOI: `10.57967/hf/10530`

## Computational environment

CI-MSDCNN was implemented and evaluated on a GPU-enabled high-performance computing cluster managed by Slurm.

The main software environment was:

- **Python 3.12.2**
- **PyTorch 2.5.1**
- **CUDA 11.8**

Model training and evaluation were performed in an Anaconda environment with one GPU allocated per Slurm job.

## Dependencies

The main Python packages used by the current scripts include:

```text
torch
numpy
pandas
h5py
openpyxl
```

The scripts also use project-specific utilities for molecular fingerprint processing and candidate evaluation. Input file paths, auxiliary utilities and local directory structures should be configured according to the corresponding scripts.

## Usage

### Train the positive-ion model

```bash
python train_CI_MSDCNN_positive.py
```

### Train the negative-ion model

```bash
python train_CI_MSDCNN_negative.py
```

### Test the positive-ion model

```bash
python test_CI_MSDCNN_positive.py
```

### Test the negative-ion model

```bash
python test_CI_MSDCNN_negative.py
```

The paths to processed MS/MS data, molecular fingerprint/physicochemical-property tables, pretrained model weights and evaluation datasets should be adjusted according to the local directory structure.

## Source code

The CI-MSDCNN source code is available at:

https://github.com/xianfu-cqmu/CI-MSDCNN

## Model weights

Pretrained positive- and negative-ion model weights:

https://huggingface.co/Xianfu1993/CI-MSDCNN

Permanent DOI:

https://doi.org/10.57967/hf/10530

## Citation

If you use CI-MSDCNN or the pretrained models in your research, please cite the corresponding publication and the model repository.

Citation information for the associated manuscript will be added after publication.

Pretrained model repository DOI:

`10.57967/hf/10530`

## Contact

For questions regarding CI-MSDCNN, please contact the authors through the GitHub repository.
