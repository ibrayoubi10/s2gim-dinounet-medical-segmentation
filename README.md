# S2GIM and DINO U-Net for Medical Image Segmentation

This repository contains the implementation of a deep learning pipeline for medical image segmentation, developed as part of a Master 2 research internship.

The project focuses on the study and optimization of medical image segmentation models using modern deep learning architectures and advanced data augmentation techniques.

## Overview

Medical image segmentation is a key task in computer-aided diagnosis, treatment planning and clinical image analysis. However, it remains challenging due to:

- limited availability of annotated medical data;
- anatomical variability between patients;
- class imbalance between foreground structures and background;
- difficulty in accurately segmenting fine boundaries;
- variability across imaging modalities.

This project investigates how modern segmentation architectures and structured data augmentation methods can improve segmentation robustness and generalization.

## Main Contributions

The project includes two main contributions:

### 1. S2GIM: Superpixel and Saliency-Guided Image Mixing

S2GIM is a data augmentation method designed for medical image segmentation.

It combines two medical images and their corresponding masks using:

- superpixel segmentation to preserve local image structures;
- saliency maps to guide the mixing process toward informative regions;
- pixel-wise mixing rules to ensure consistency between the augmented image and its segmentation mask.

The goal of S2GIM is to generate more realistic and structurally coherent augmented samples compared to standard augmentation methods such as CutOut, CutMix and MixUp.

### 2. DINO U-Net

DINO U-Net is implemented and evaluated as a modern segmentation architecture based on a visual foundation model.

The architecture uses:

- a DINOv3 Vision Transformer encoder;
- dense feature extraction from multiple transformer blocks;
- a U-Net-like decoder;
- skip connections for multi-scale feature fusion;
- a final segmentation head for pixel-wise prediction.

The model is adapted to a 2D medical image segmentation pipeline and compared with other segmentation architectures.

## Evaluated Architectures

The pipeline is designed to evaluate several segmentation models, including:

- U-Net
- Attention U-Net
- UNeXt
- EfficientUNet
- DeepLabV3+
- TransUNet
- HiFormer
- DINO U-Net

## Datasets

Experiments are designed for public medical image segmentation datasets, including:

| Dataset | Modality | Task |
|---|---|---|
| Synapse | CT scans | Multi-organ abdominal segmentation |
| ISIC 2017 | Dermoscopy | Skin lesion segmentation |
| GLaS | Histology | Gland segmentation |
| MoNuSeg | Histology | Nuclei segmentation |

> Note: The datasets are not included in this repository. They must be downloaded from their official sources and placed in the appropriate data directory.

## Evaluation Metrics

The models are evaluated using standard medical image segmentation metrics:

- Dice Score
- Jaccard Index / Intersection over Union
- HD95: 95th percentile Hausdorff Distance

These metrics allow the evaluation of both global region overlap and boundary quality.


## Results

request via email: ayoubi192003@gmail.com


## Author

**Ibrahim Al Ayoubi**
Master 2 Artificial Intelligence and Data Science
Université de Montpellier

## Academic Context

This work was developed as part of a Master 2 research internship on medical image segmentation using deep learning.

Internship topic:

**Study and optimization of medical image segmentation using deep learning: impact of modern architectures and data augmentation techniques.**

## License

This repository is intended for academic and research purposes.