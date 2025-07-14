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
from torch.cuda.amp import autocast, GradScaler # adding mixed precision training

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
    if pretrained_path and os.path.exists(pretrained_path):
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

def validate_model(model, valloader, criterion, device, test_mode=False, max_batches=None):
    """
    Validate the model on validation set
    
    Args:
        model: xQSM model
        valloader: Validation dataloader
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
        for inputs, targets, names in valloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Early termination for testing
            if test_mode and max_batches and num_batches >= max_batches:
                print(f"  [TEST MODE] Validated {num_batches} batches only")
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def SaveNet(model, epoch, save_dir='./transfer_learning_checkpoints', best_loss=None, test_mode=False):
    """
    Save network checkpoints
    
    Args:
        model: Model to save
        epoch: Current epoch
        save_dir: Directory to save checkpoints
        best_loss: Best validation loss (if this is the best model)
        test_mode: If True, skip saving in test mode
    """
    if test_mode:
        print(f"  [TEST MODE] Skipping checkpoint save for epoch {epoch}")
        return
        
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
                         Epoches=50, useGPU=True, save_dir='./transfer_learning_checkpoints',
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
        save_dir: Directory to save checkpoints
        test_mode: If True, run in test mode with early termination
        max_test_batches: Maximum batches per epoch in test mode
        max_test_epochs: Maximum epochs in test mode
    """
    print('='*80)
    if test_mode:
        print('TRANSFER LEARNING TRAINING - CPU TEST MODE')
        print(f'Test Mode: Max {max_test_epochs} epochs, {max_test_batches} batches per epoch')
    else:
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
    
    # Limit epochs in test mode
    if test_mode:
        Epoches = min(Epoches, max_test_epochs)
        print(f"[TEST MODE] Limited to {Epoches} epochs")
    
    ## start the timer. 
    time_start = time.time()
    
    # Device selection
    if useGPU and torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} Available GPUs!")
        device = torch.device("cuda:0")
        Chi_Net = nn.DataParallel(Chi_Net)
        Chi_Net.to(device)
        print("Using GPU for training")
    else:
        device = torch.device("cpu")
        Chi_Net.to(device)
        print("Using CPU for training")
        if test_mode:
            print("[TEST MODE] Running on CPU - this is expected for testing")
        elif useGPU:
            print("Warning: GPU requested but not available, falling back to CPU")

    for epoch in range(1, Epoches + 1):
        print(f"\n{'='*60}")
        print(f"STARTING EPOCH {epoch}/{Epoches}")
        print(f"{'='*60}")
        
        # Training phase
        Chi_Net.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        # Get total number of batches for progress tracking
        total_batches = len(trainloader)
        if test_mode:
            total_batches = min(total_batches, max_test_batches)
        print(f"Total batches to process this epoch: {total_batches}")
        
        scaler = GradScaler()

        for i, (inputs, targets, names) in enumerate(trainloader):
            print(f"\n--- BATCH {i + 1}/{total_batches} (Epoch {epoch}) ---")
            
            # Print detailed input information
            print(f"INPUT DATA:")
            print(f"   - Input shape: {inputs.shape}")
            print(f"   - Target shape: {targets.shape}")
            print(f"   - Sample names: {names}")
            print(f"   - Input dtype: {inputs.dtype}")
            print(f"   - Target dtype: {targets.dtype}")
            print(f"   - Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
            print(f"   - Target range: [{targets.min():.4f}, {targets.max():.4f}]")
            
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(f"   - Data moved to device: {device}")
            
            print(f"TRAINING STEP:")
            print(f"   - Zeroing gradients...")
            ## zero the gradient buffers 
            optimizer.zero_grad()
            
            print(f"   - Forward pass...")
            ## forward: 

            with autocast():                             
                outputs = Chi_Net(inputs)
                loss = criterion(outputs, targets)
            print(f"   - Prediction shape: {outputs.shape}")
            print(f"   - Prediction range: [{outputs.min():.4f}, {outputs.max():.4f}]")
            
            print(f"   - Computing loss...")
            ## loss
            print(f"   - Loss value: {loss.item():.6f}")
            
            print(f"   - Backward pass...")
            ## backward
            scaler.scale(loss).backward()                  
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            # Progress and timing information
            time_end = time.time()
            elapsed_time = time_end - time_start
            
            print(f"PROGRESS:")
            print(f"   - Current batch loss: {loss.item():.6f}")
            print(f"   - Average loss so far: {epoch_train_loss/num_train_batches:.6f}")
            print(f"   - Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"   - Elapsed time: {elapsed_time:.0f}s")
            print(f"   - Batch progress: {i+1}/{total_batches} ({(i+1)/total_batches*100:.1f}%)")
            
            # Memory usage (if CUDA)
            if torch.cuda.is_available() and device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
                print(f"   - GPU Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")
            
            # Early termination for testing
            if test_mode and i >= max_test_batches - 1:
                print(f"\n[TEST MODE] Completed {i + 1} batches, moving to validation")
                break
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} TRAINING PHASE COMPLETE")
        print(f"{'='*60}")
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / num_train_batches
        print(f"EPOCH {epoch} TRAINING RESULTS:")
        print(f"   - Batches processed: {num_train_batches}")
        print(f"   - Average training loss: {avg_train_loss:.6f}")
        
        print(f"\nSTARTING VALIDATION PHASE...")
        # Validation phase
        val_loss = validate_model(Chi_Net, valloader, criterion, device, 
                                test_mode=test_mode, max_batches=max_test_batches)
        print(f"Validation complete - Loss: {val_loss:.6f}")
        
        # Learning rate scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate updated: {old_lr:.6f} → {new_lr:.6f}")
        
        # Print epoch summary
        time_end = time.time()
        total_time = time_end - time_start
        epoch_time = total_time if epoch == 1 else total_time / epoch
        
        print(f'\n{"EPOCH " + str(epoch) + " SUMMARY":=^60}')
        print(f'   Train Loss: {avg_train_loss:.6f}')
        print(f'   Val Loss: {val_loss:.6f}')
        print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'   Epoch Time: {epoch_time:.0f}s')
        print(f'   Total Time: {total_time:.0f}s')
        print(f'   Est. Remaining: {epoch_time * (Epoches - epoch):.0f}s')
        print('=' * 60)
        
        # Save checkpoints (skip in test mode)
        if epoch % 10 == 0:
            print(f"Saving checkpoint for epoch {epoch}...")
            SaveNet(Chi_Net, epoch, save_dir, test_mode=test_mode)
        
        # Save best model
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            print(f"NEW BEST MODEL! Validation loss improved by {improvement:.6f}")
            SaveNet(Chi_Net, epoch, save_dir, best_val_loss, test_mode=test_mode)
        
        # Early termination message for test mode
        if test_mode and epoch >= max_test_epochs:
            print(f"\n[TEST MODE] Completed {epoch} epochs - stopping early")
            break
            
    print('Training Complete!')
    if not test_mode:
        SaveNet(Chi_Net, Epoches, save_dir)
    print(f'Best validation loss achieved: {best_val_loss:.6f}')
    
    if test_mode:
        print("\n" + "="*80)
        print("TEST MODE COMPLETE")
        print("="*80)
        print("CPU training test successful!")
        print("Data loading works correctly")
        print("Model forward/backward pass works")
        print("Loss calculation and optimization working")
        print("\nYou can now run full training with:")
        print("- useGPU=True for GPU training")
        print("- test_mode=False for full training")
        print("- Increase epochs and batch size as needed")

if __name__ == '__main__':
    # Configuration
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'  # Update this path
    PRETRAINED_PATH = './ChiNet_Latest.pth'  # Path to pretrained weights (optional)
    
    # Test mode parameters
    TEST_MODE = True  # Set to True for CPU testing, False for full training
    
    if TEST_MODE:
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
        LEARNING_RATE = 0.001  # Higher LR for quick testing
        BATCH_SIZE = 1  # Small batch size for CPU
        EPOCHS = 1  # Just test a couple epochs
        USE_GPU = False  # CPU only for testing
        
    else:
        # Full training parameters
        LEARNING_RATE = 0.0001  # Lower learning rate for transfer learning
        BATCH_SIZE = 4  # Smaller batch size due to potentially limited data
        EPOCHS = 50  # Fewer epochs needed for transfer learning
        USE_GPU = True  # Use GPU for full training
    
    print("Starting Transfer Learning Training...")
    print(f"Data Directory: {DATA_DIRECTORY}")
    print(f"Pretrained Weights: {PRETRAINED_PATH}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Use GPU: {USE_GPU}")
    print(f"Test Mode: {TEST_MODE}")
    
    ## Start transfer learning training
    TrainTransferLearning(
        data_directory=DATA_DIRECTORY,
        pretrained_path=PRETRAINED_PATH,
        LR=LEARNING_RATE,
        Batchsize=BATCH_SIZE, 
        Epoches=EPOCHS,
        useGPU=USE_GPU,
        test_mode=TEST_MODE,
        max_test_batches=2,
        max_test_epochs=1
    ) 