# Object Centric Learning for Uncertainty Aware Medical Image Segmentation

This project applies slot attention to medical image segmentation tasks, featuring optional ResNet and DINO ViT encoders, along with additive CNN and transformer decoders.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [License](#license)

## Overview
This has been developed as part of research into using object representation methods for data efficient medical image segmentation tasks.

## Features
- Fixed probabilistic slot attention mechanism
- Optional encoders:
  - ResNet
  - DINO ViT (Vision Transformer)
- Decoders:
  - Additive CNN
  - Additive ViT
  - Autoregressive ViT

## Installation
```bash
git clone https://github.com/bcreganprogs/semi_supervised_uncertainty.git
cd semi_supervised_uncertainty
pip install -r requirements.txt
```

## Usage
Please update model configuration in a bash script located in bash_scripts and run:

```bash
bash (script_name).sh
```

## Model Architecture
Describe your model's architecture, including:
- Slot attention mechanism
- Encoder options (ResNet, DINO ViT)
- Decoder configurations (Additive CNN, Additive Transformer, Autoregressive Transformer)

## License
This work is released under an MIT License.
