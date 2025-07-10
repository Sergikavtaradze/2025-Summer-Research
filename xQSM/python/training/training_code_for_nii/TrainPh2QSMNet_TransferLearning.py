################### Transfer Learning Training for xQSM #####################
#########  Network Training with Frozen Encoding Layers #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
import os
from xQSM import * 
from TrainingDataLoadHN import QSMDataSet
from torch.utils import data

#########  Section 1: DataSet Load #############
def DataLoad(data_directory, batch_size=8, test_split=0.2):
    """
    Load the head and neck QSM dataset
    
    Args:
        data_directory (str): Path to QSM data directory
        batch_size (int): Batch size for training
        test_split (float): Fraction of data to use for validation
    """
    # Create full dataset
    full_dataset = QSMDataSet(data_directory, include_noise=True)
    print(f'Total dataset length: {len(full_dataset)}')
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create dataloaders
    trainloader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    valloader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    return trainloader, valloader

def freeze_encoding_layers(model):
    """
    Freeze the encoding layers of the xQSM model for transfer learning
    
    Args:
        model: xQSM model instance
    """
    print("Freezing encoding layers...")
    
    # Freeze input octave layer
    for param in model.InputOct.parameters():
        param.requires_grad = False
    print("  - InputOct layer frozen")
    
    # Freeze all encoding convolution layers
    for i, encode_conv in enumerate(model.EncodeConvs):
        for param in encode_conv.parameters():
            param.requires_grad = False
        print(f"  - EncodeConv{i+1} layer frozen")
    
    # Optionally freeze middle layer (uncomment if you want to freeze it)
    # for param in model.MidConv1.parameters():
    #     param.requires_grad = False
    # print("  - MidConv1 layer frozen")
    
    # Count trainable vs total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    return model

def load_pretrained_weights(model, pretrained_path):
    """
    Load pretrained weights into the model
    
    Args:
        model: xQSM model instance
        pretrained_path: Path to pretrained weights file
    """
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        try:
            # Load the state dict
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Handle DataParallel wrapper if present
            if 'module.' in list(pretrained_dict.keys())[0]:
                # Remove 'module.' prefix from keys
                pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            
            # Load the weights
            model.load_state_dict(pretrained_dict, strict=True)
            print("Successfully loaded pretrained weights!")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with random initialization...")
            model.apply(weights_init)
    else:
        print(f"Pretrained weights file not found: {pretrained_path}")
        print("Proceeding with random initialization...")
        model.apply(weights_init)
    
    return model

def validate_model(model, valloader, criterion, device):
    """
    Validate the model on validation set
    
    Args:
        model: xQSM model
        valloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, names in valloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def SaveNet(model, epoch, save_dir='./transfer_learning_checkpoints', best_loss=None):
    """
    Save network checkpoints
    
    Args:
        model: Model to save
        epoch: Current epoch
        save_dir: Directory to save checkpoints
        best_loss: Best validation loss (if this is the best model)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Always save latest checkpoint
    latest_path = os.path.join(save_dir, 'xQSM_TransferLearning_Latest.pth')
    torch.save(model.state_dict(), latest_path)
    
    # Save epoch-specific checkpoint
    epoch_path = os.path.join(save_dir, f'xQSM_TransferLearning_epoch_{epoch}.pth')
    torch.save(model.state_dict(), epoch_path)
    
    # Save best model if specified
    if best_loss is not None:
        best_path = os.path.join(save_dir, 'xQSM_TransferLearning_Best.pth')
        torch.save(model.state_dict(), best_path)
        print(f'New best model saved with validation loss: {best_loss:.6f}')

def TrainTransferLearning(data_directory, pretrained_path=None, LR=0.001, Batchsize=8, 
                         Epoches=50, useGPU=True, save_dir='./transfer_learning_checkpoints'):
    """
    Train xQSM model with transfer learning approach
    
    Args:
        data_directory: Path to head and neck QSM data
        pretrained_path: Path to pretrained model weights
        LR: Learning rate
        Batchsize: Batch size
        Epoches: Number of epochs
        useGPU: Whether to use GPU
        save_dir: Directory to save checkpoints
    """
    print('='*80)
    print('TRANSFER LEARNING TRAINING FOR HEAD AND NECK QSM')
    print('='*80)
    
    # Create model
    Chi_Net = xQSM(EncodingDepth=2, ini_chNo=32)  # Using same config as original
    
    # Load pretrained weights if available
    Chi_Net = load_pretrained_weights(Chi_Net, pretrained_path)
    
    # Freeze encoding layers
    Chi_Net = freeze_encoding_layers(Chi_Net)
    
    # Set model to training mode
    Chi_Net.train()
    
    print('\nDataLoader setting begins')
    trainloader, valloader = DataLoad(data_directory, Batchsize)
    print('Dataloader setting end')

    print('\nTraining Begins')
    criterion = nn.MSELoss(reduction='mean')  # Using mean for better comparison

    # Only optimize parameters that require gradients (unfrozen layers)
    trainable_params = [p for p in Chi_Net.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)

    # More aggressive learning rate schedule for transfer learning
    scheduler = LS.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    ## start the timer. 
    time_start = time.time()
    
    if useGPU:
        if torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            Chi_Net = nn.DataParallel(Chi_Net)
            Chi_Net.to(device)

            for epoch in range(1, Epoches + 1):
                # Training phase
                Chi_Net.train()
                epoch_train_loss = 0.0
                num_train_batches = 0
                
                for i, (inputs, targets, names) in enumerate(trainloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    ## zero the gradient buffers 
                    optimizer.zero_grad()
                    
                    ## forward: 
                    pred_targets = Chi_Net(inputs)
                    
                    ## loss
                    loss = criterion(pred_targets, targets)
                    
                    ## backward
                    loss.backward()
                    
                    ## optimization step
                    optimizer.step()
                    
                    # Accumulate loss
                    epoch_train_loss += loss.item()
                    num_train_batches += 1
                    
                    ## print statistical information 
                    if i % 10 == 0:  # Print every 10 batches
                        time_end = time.time()
                        print(f'Epoch: {epoch}, Batch: {i + 1}, Loss: {loss.item():.6f}, '
                              f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                              f'Time: {time_end - time_start:.0f}s')
                
                # Calculate average training loss
                avg_train_loss = epoch_train_loss / num_train_batches
                
                # Validation phase
                val_loss = validate_model(Chi_Net, valloader, criterion, device)
                
                # Learning rate scheduler step
                scheduler.step()
                
                # Print epoch summary
                time_end = time.time()
                print(f'\nEpoch {epoch} Summary:')
                print(f'  Train Loss: {avg_train_loss:.6f}')
                print(f'  Val Loss: {val_loss:.6f}')
                print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                print(f'  Total Time: {time_end - time_start:.0f}s')
                print('-' * 50)
                
                # Save checkpoints
                if epoch % 10 == 0:
                    SaveNet(Chi_Net, epoch, save_dir)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    SaveNet(Chi_Net, epoch, save_dir, best_val_loss)
                
        else:
            print('No CUDA Device Available!')
            return
            
    print('Training Complete!')
    SaveNet(Chi_Net, Epoches, save_dir)
    print(f'Best validation loss achieved: {best_val_loss:.6f}')

if __name__ == '__main__':
    # Configuration
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'  # Update this path
    PRETRAINED_PATH = './ChiNet_Latest.pth'  # Path to pretrained weights (optional)
    
    # Training parameters - adjusted for transfer learning
    LEARNING_RATE = 0.0001  # Lower learning rate for transfer learning
    BATCH_SIZE = 4  # Smaller batch size due to potentially limited data
    EPOCHS = 50  # Fewer epochs needed for transfer learning
    
    print("Starting Transfer Learning Training...")
    print(f"Data Directory: {DATA_DIRECTORY}")
    print(f"Pretrained Weights: {PRETRAINED_PATH}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    ## Start transfer learning training
    TrainTransferLearning(
        data_directory=DATA_DIRECTORY,
        pretrained_path=PRETRAINED_PATH,
        LR=LEARNING_RATE,
        Batchsize=BATCH_SIZE, 
        Epoches=EPOCHS,
        useGPU=True
    ) 