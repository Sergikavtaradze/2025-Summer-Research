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
import argparse

def freeze_encoding_layers(model):
    """
    Freeze the encoding layers of the xQSM model for transfer learning
    
    Args:
        model: xQSM model instance
    """
    # Freeze input octave layer
    for param in model.InputOct.parameters():
        param.requires_grad = False
    
    # Freeze all encoding convolution layers
    for i, encode_conv in enumerate(model.EncodeConvs):
        for param in encode_conv.parameters():
            param.requires_grad = False
    
    # Count trainable vs total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Transfer Learning Setup: {trainable_params:,}/{total_params:,} trainable parameters ({trainable_params/total_params*100:.1f}%)")
    
    return model

def load_pretrained_weights(model, pretrained_path):
    """
    Load pretrained weights into the model
    
    Args:
        model: xQSM model instance
        pretrained_path: Path to pretrained weights file
    """
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            # Load the state dict
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Handle DataParallel wrapper if present
            if 'module.' in list(pretrained_dict.keys())[0]:
                # Remove 'module.' prefix from keys
                pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            
            # Load the weights
            model.load_state_dict(pretrained_dict, strict=True)
            print(f"Loaded pretrained weights from: {pretrained_path}")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using random initialization...")
            model.apply(weights_init)
    else:
        print("Using random initialization...")
        model.apply(weights_init)
    
    return model

def validate_model(model, val_loader, criterion, device, test_mode=False, max_batches=None):
    """
    Validate the model on validation set
    
    Args:
        model: xQSM model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        test_mode: If True, limit validation for testing
        max_batches: Maximum number of batches to validate (for testing)
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, names in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Early termination for testing
            if test_mode and max_batches and num_batches >= max_batches:
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def SaveNet(model, epoch, snapshot_path='./transfer_learning_checkpoints', best_loss=None, test_mode=False):
    """
    Save network checkpoints
    
    Args:
        model: Model to save
        epoch: Current epoch
        snapshot_path: Directory to save checkpoints
        best_loss: Best validation loss (if this is the best model)
        test_mode: If True, skip saving in test mode
    """
    if test_mode:
        # No saving in test mode
        return
        
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Always save latest checkpoint
    latest_path = os.path.join(snapshot_path, 'xQSM_TransferLearning_Latest.pth')
    torch.save(model.state_dict(), latest_path)
    
    # Save epoch-specific checkpoint
    epoch_path = os.path.join(snapshot_path, f'xQSM_TransferLearning_epoch_{epoch}.pth')
    torch.save(model.state_dict(), epoch_path)
    
    # Save best model if specified
    if best_loss is not None:
        best_path = os.path.join(snapshot_path, 'xQSM_TransferLearning_Best.pth')
        torch.save(model.state_dict(), best_path)

def TrainTransferLearning(data_directory, pretrained_path=None, LR=0.001, batch_size=8, 
                         Epoches=50, useGPU=True, snapshot_path='./transfer_learning_checkpoints',
                         test_mode=False, max_test_batches=3, max_test_epochs=2):
    """
    Train xQSM model with transfer learning approach
    
    Args:
        data_directory: Path to head and neck QSM data
        pretrained_path: Path to pretrained model weights
        LR: Learning rate
        Batchsize: Batch size
        Epoches: Number of epochs
        useGPU: Whether to use GPU
        snapshot_path: Directory to save checkpoints
        test_mode: If True, run in test mode with early termination
        max_test_batches: Maximum batches per epoch in test mode
        max_test_epochs: Maximum epochs in test mode
    """
    print('='*80)
    if test_mode:
        print('TRANSFER LEARNING TRAINING - TEST MODE')
    else:
        print('TRANSFER LEARNING TRAINING FOR HEAD AND NECK QSM')
    print('='*80)
    
    # Data Loading
    train_dataset = QSMDataSet(data_directory, split_type='train')
    val_dataset = QSMDataSet(data_directory, split_type='val')
        
    print(f'Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples')
    
    # Create dataloaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Create model
    Chi_Net = xQSM(EncodingDepth=2, ini_chNo=32)
    
    # Load pretrained weights if available
    Chi_Net = load_pretrained_weights(Chi_Net, pretrained_path)
    
    # Freeze encoding layers
    Chi_Net = freeze_encoding_layers(Chi_Net)
    
    # Set model to training mode
    Chi_Net.train()

    criterion = nn.MSELoss(reduction='mean')

    # Only optimize unfrozen layers
    trainable_params = [p for p in Chi_Net.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)

    # More aggressive learning rate schedule for transfer learning
    scheduler = LS.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Limit epochs in test mode
    if test_mode:
        Epoches = min(Epoches, max_test_epochs)
    
    ## start the timer. 
    time_start = time.time()
    
    # Device selection
    if useGPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        Chi_Net = nn.DataParallel(Chi_Net)
        Chi_Net.to(device)
        print(f"Using GPU: {torch.cuda.device_count()} devices")
    else:
        device = torch.device("cpu")
        Chi_Net.to(device)
        print("Using CPU")

    print(f"Training for {Epoches} epochs...")
    
    for epoch in range(1, Epoches + 1):
        # Training phase
        Chi_Net.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for i, (inputs, targets, names) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # zero the gradient buffers 
            optimizer.zero_grad()
            
            # forward pass
            outputs = Chi_Net(inputs)
            loss = criterion(outputs, targets)
            
            # backward pass
            loss.backward()                  
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            # Early termination for testing
            if test_mode and i >= max_test_batches - 1:
                break
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / num_train_batches
        
        # Validation phase
        val_loss = validate_model(Chi_Net, val_loader, criterion, device, 
                                test_mode=test_mode, max_batches=max_test_batches)
        
        # Learning rate scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # Calculate timing
        time_end = time.time()
        epoch_time = (time_end - time_start) / epoch
        
        # Print epoch summary with all requested information
        print(f'Epoch [{epoch:3d}/{Epoches}] train_loss: {avg_train_loss:.6f} | val_loss: {val_loss:.6f} | best_val_loss: {best_val_loss:.6f} (epoch {best_epoch}) | Time: {epoch_time:.0f}s')
        
        # Save checkpoints periodically
        if epoch % 10 == 0 and not test_mode:
            print(f"  → Checkpoint saved (epoch {epoch})")
            SaveNet(Chi_Net, epoch, snapshot_path, test_mode=test_mode)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if not test_mode:
                SaveNet(Chi_Net, epoch, snapshot_path, best_val_loss, test_mode=test_mode)
            print(f"New best model! Val: {best_val_loss:.6f} (epoch {best_epoch})")
        
        # Early termination for test mode
        if test_mode and epoch >= max_test_epochs:
            break
            
    # Final summary
    total_time = time.time() - time_start
    print('='*80)
    print('TRAINING COMPLETE')
    print(f'Best validation loss: {best_val_loss:.6f} (achieved at epoch {best_epoch})')
    print(f'Total training time: {total_time:.0f}s ({total_time/60:.1f}min)')
    
    if not test_mode:
        SaveNet(Chi_Net, Epoches, snapshot_path)
        print(f"Final model saved to: {snapshot_path}")
    print('='*80)
    
    if test_mode:
        print("Test mode completed successfully - ready for full training")

if __name__ == '__main__':
    # Configuration
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_directory", required=True, type=str)
    parser.add_argument("--pretrained_path", required=True, type=str)
    parser.add_argument("--snapshot_path", required=True, type=str)

    parser.add_argument("-lr", "--learning_rate", default=4e-4, type=float)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("-ep", "--epochs", default=50, type=int)
    parser.add_argument("--test_mode", action="store_true", help="Default is False, Run in test mode,")
    parser.add_argument("--use_gpu", action="store_true", help="Default is False, Use GPU for training,")
    
    #parser.add_argument("--num_worker", default=6, type=int)
    #parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()

    # Data parameters
    data_directory = args.data_directory
    pretrained_path = args.pretrained_path
    snapshot_path = args.snapshot_path
    
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    # Training mode
    test_mode = args.test_mode
    use_gpu = args.use_gpu
    
    if test_mode:
        print("="*80)
        print("RUNNING IN TEST MODE")
        print("="*80)
        print("This will:")
        print("- Use CPU only")
        print("- Run for 2 epochs maximum")
        print("- Process 3 batches per epoch")
        print("- Skip checkpoint saving")
        print("- Verify everything works before full training")
        print("="*80)
        
        # Test parameters
        learning_rate = 0.001  # Higher LR for quick testing
        batch_size = 3  # Small batch size for CPU
        epochs = 2  # Just test a couple epochs
        use_gpu = False  # CPU only for testing
        
    else:
        # Full training parameters
        use_gpu = True  # Use GPU for full training
    
    print("Starting Transfer Learning Training...")
    print(f"Data Directory: {data_directory}")
    print(f"Pretrained Weights: {pretrained_path}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"epochs: {epochs}")
    print(f"Use GPU: {use_gpu}")
    print(f"Test Mode: {test_mode}")
    
    ## Start transfer learning training
    TrainTransferLearning(
        data_directory=data_directory,
        pretrained_path=pretrained_path,
        LR=learning_rate,
        batch_size=batch_size, 
        Epoches=epochs,
        useGPU=use_gpu,
        snapshot_path=snapshot_path,
        test_mode=test_mode,
        max_test_batches=2,
        max_test_epochs=1
    ) 