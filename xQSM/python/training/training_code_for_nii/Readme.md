# xQSM Transfer Learning Training

This directory contains code for training xQSM (octave-convolution based Quantitative Susceptibility Mapping) models using transfer learning on head and neck datasets.

## Overview

The xQSM model is an octave U-Net architecture that processes 3D medical imaging data to predict quantitative susceptibility maps. This implementation supports transfer learning, allowing you to fine-tune pre-trained models on new datasets.

## Architecture

- **Model**: xQSM - Octave U-Net with encoding/decoding layers
- **Input**: 3D local field maps (NIfTI format)
- **Output**: 3D susceptibility maps (NIfTI format)
- **Transfer Learning**: Freeze encoding layers, train decoding layers

## File Structure

```
training_code_for_nii/
├── README.md                           # This file
├── xQSM.py                            # Main model architecture
├── xQSM_blocks.py                     # Octave convolution blocks
├── TrainingDataLoadHN.py              # Head & Neck dataset loader
├── TrainPh2QSMNet_TransferLearning.py # Transfer learning training script
├── TrainPh2QSMNet.py                  # Original training script
├── model_summary.py                   # Model analysis tool
├── test_dataloader.py                 # Dataset testing utility
└── transfer_learning_checkpoints/     # Saved model checkpoints
```

## Requirements

### Dependencies
```bash
pip install torch torchvision
pip install nibabel numpy
pip install torchinfo  # For model summary
```

### Data Structure
Our data loader loads data with the following structure:
```
QSM_data/
├── sub-01/
│   └── ses-01/
│       └── qsm/
│           ├── sub-01_ses-01_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield.nii.gz    # Input
│           └── sub-01_ses-01_unwrapped-SEGUE_mask-nfe_bfr-PDF_susc-autoNDI_Chimap.nii.gz  # Target
├── sub-02/
│   └── ses-01/
│       └── qsm/
│           ├── sub-02_ses-01_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield.nii.gz
│           └── sub-02_ses-01_unwrapped-SEGUE_mask-nfe_bfr-PDF_susc-autoNDI_Chimap.nii.gz
└── ...
```

## Workflow Summary

1. **Setup**: Install dependencies, organize data
2. **Test**: Run CPU test mode to verify everything works
3. **Analyze**: Generate model summary to understand architecture
4. **Configure**: Set appropriate parameters for your dataset/hardware
5. **Train**: Run full transfer learning training
6. **Monitor**: Track progress and adjust parameters as needed
7. **Evaluate**: Test trained model on validation data

## Quick Start

### 1. Test CPU Training (Recommended First)
Test that everything works with minimal resources
```bash
python TrainPh2QSMNet_TransferLearning.py
```
This runs in test mode by default:
- Uses CPU only
- Processes 2 batches per epoch
- Runs for 1 epoch
- Skips checkpoint saving
- Verifies data loading and model forward/backward passes

### 2. Analyze Model Structure
Generate detailed model summary
```bash
python model_summary.py
```
This creates a text file with:
- Layer-by-layer parameter counts
- Freezing recommendations
- Transfer learning impact analysis

### 3. Test Dataset Loading
Verify your dataset is correctly formatted
```bash
python test_dataloader.py
```

### 4. Full Transfer Learning Training
Edit `TrainPh2QSMNet_TransferLearning.py`:
```python
# Change these settings for full training
TEST_MODE = False  # Enable full training
USE_GPU = True     # Use GPU if available
EPOCHS = 50        # Number of training epochs
BATCH_SIZE = 4     # Batch size
```

Then run:
```bash
python TrainPh2QSMNet_TransferLearning.py
```

## Configuration

### Key Parameters in `TrainPh2QSMNet_TransferLearning.py`:

```python
# Data
DATA_DIRECTORY = '/path/to/your/QSM_data'  # Update this path
PRETRAINED_PATH = './ChiNet_Latest.pth'    # Pre-trained weights (optional)

# Training Mode
TEST_MODE = True/False          # Test mode vs full training
USE_GPU = True/False           # GPU usage

# Training Parameters
LEARNING_RATE = 0.0001         # Lower for transfer learning
BATCH_SIZE = 4                 # Adjust based on GPU memory
EPOCHS = 50                    # Fewer epochs needed for transfer learning

# Test Mode
max_test_batches = 2           # Batches per epoch in test mode
max_test_epochs = 1            # Epochs in test mode
```

### Model Parameters:
```python
# In xQSM model initialization
EncodingDepth = 2              # Number of encoding layers
ini_chNo = 32                  # Initial number of channels
```

## Transfer Learning Strategy

### Frozen Layers:
- `InputOct`: Initial octave convolution layer
- `EncodeConv1`: First encoding layer
- `EncodeConv2`: Second encoding layer

### Trainable Layers:
- `MidConv1`: Middle processing layer
- `DecodeConv1`: First decoding layer  
- `DecodeConv2`: Second decoding layer
- `FinalOct`: Final output layer

## Training Process

### 1. Data Loading
- Scans directory for paired input/target files
- Splits into train/validation (80/20)
- Applies noise augmentation (20% probability)
- Loads as 3D tensors with channel dimension

### 2. Model Setup
- Creates xQSM model with specified architecture
- Loads pre-trained weights (if available)
- Freezes encoding layers for transfer learning
- Sets up optimizer for trainable parameters only

### 3. Training Loop
Each epoch:
1. **Training Phase**: Forward pass, loss computation, backpropagation
2. **Validation Phase**: Evaluate on held-out data
3. **Checkpointing**: Save best model and periodic checkpoints
4. **Learning Rate Scheduling**: Reduce LR at milestones

### 4. Monitoring
- Real-time loss tracking
- Input/output shape verification
- Memory usage monitoring
- Training time estimation

## Output Files

### Checkpoints (saved in `transfer_learning_checkpoints/`):
- `xQSM_TransferLearning_Latest.pth`: Most recent model
- `xQSM_TransferLearning_Best.pth`: Best validation loss model
- `xQSM_TransferLearning_epoch_X.pth`: Epoch-specific checkpoints

### Logs:
- Console output with detailed training progress
- Model summary files (if generated)
- Dataset verification logs

## Common Issues:

**1. CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 2  # or 1

# Or use CPU for testing
USE_GPU = False

# Use Data Chunking
**Feature to be added**
```

**2. Dataset Not Found**
```bash
#Check data directory path
ls /path/to/your/QSM_data

#Verify file naming convention matches expected pattern
```

**3. Model Loading Errors**
```python
# Check pre-trained weights path
PRETRAINED_PATH = None  # Skip pre-trained weights
```

**4. Memory Issues**
```python
# Enable test mode for debugging
TEST_MODE = True
max_test_batches = 1
```

## Model Analysis

Use the model summary tool to understand your model:

```bash
python model_summary.py
```

## Contact & Support

For questions about:
- **Model architecture**: Check `xQSM.py` and `xQSM_blocks.py`
- **Data loading**: See `TrainingDataLoadHN.py` and `test_dataloader.py`
- **Training issues**: Review console output and adjust parameters
- **Performance**: Use model summary and monitoring tools

Happy training! 🚀