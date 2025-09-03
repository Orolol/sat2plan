# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAT2PLAN is a computer vision project that converts satellite images to maps using Generative Adversarial Networks (GANs). The project supports multiple GAN architectures and includes a Streamlit web interface for interactive map generation from GPS coordinates or addresses.

## Development Commands

### Package Installation
```bash
# Install package in development mode
pip install -e .

# Reinstall package (useful after code changes)
make reinstall_package
```

### Training Models
The project supports multiple model architectures:
```bash
# Train U-Net model
make run_train_unet

# Train UCV-GAN model  
make run_train_ucvgan

# Train SAM-GAN model
make run_train_sam_gan

# Train Vision Transformer model
make run_train_vit

# Train Diffusion model
make run_train_diffusion
```

### Running Applications
```bash
# Run prediction interface
make run_pred

# Test API functionality
make run_api

# Start Streamlit interface directly
streamlit run sat2plan/interface/interface.py
```

### Environment Setup
```bash
# Copy environment template
cp .env.sample .env

# Run full installation script (Linux/Ubuntu)
./install.sh
```

## Architecture Overview

### Package Structure
- `sat2plan/logic/models/` - Contains different GAN implementations:
  - `unet/` - U-Net architecture for image-to-image translation
  - `ucvgan/` - UNet-Vision Transformer GAN implementation
  - `samgan/` - Style-Aware Mapping GAN
  - `basegan/` - Basic DCGAN implementation
  - `diffusion/` - Diffusion model implementation
  - `vit/` - Vision Transformer model
- `sat2plan/logic/preproc/` - Data preprocessing and dataset management
- `sat2plan/logic/configuration/` - Model and training configurations
- `sat2plan/interface/` - Streamlit web interface and training orchestration
- `sat2plan/api/` - Google Maps API integration for image retrieval

### Model Training Architecture
- Multi-GPU training support via PyTorch's `torch.multiprocessing.spawn()`
- Automatic data downloading from Google Cloud Storage buckets
- MLflow integration for experiment tracking
- Configuration-driven hyperparameters via `Global_Configuration` class

### Data Pipeline
- Images are downloaded from Google Cloud Storage buckets
- Dataset is automatically split into train/validation (95%/5% split)
- Supports various image sizes (default 256x256) and batch processing
- Albumentations library used for data augmentation

### Key Configuration Parameters
Located in `sat2plan/logic/configuration/config.py`:
- GPU/CPU device selection
- Batch size, epochs, learning rates
- Image dimensions and channels
- Model checkpointing settings
- Data paths and bucket configurations

## Environment Variables

Required environment variables (see `.env.sample`):
- `DATA_SIZE` - Dataset size identifier
- `GCP_PROJECT`, `GCP_REGION` - Google Cloud settings
- `BUCKET_NAME` - Cloud Storage bucket for datasets
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` - MLflow tracking
- `MASTER_ADDR`, `MASTER_PORT` - Distributed training settings

## Dependencies

Core dependencies include:
- PyTorch ≥2.4.0 with CUDA support
- Streamlit for web interface
- Google Cloud Storage client
- MLflow for experiment tracking
- Albumentations for image augmentation
- NumPy, Pandas, Matplotlib for data processing