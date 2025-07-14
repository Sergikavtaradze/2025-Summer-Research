import os
import numpy as np
import random
import torch
import nibabel as nib
from torch.utils import data
import glob
from functools import wraps


class QSMDataSet(data.Dataset):
    train_subjects=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08']
    val_subjects=['sub-09', 'sub-10']
    test_subjects=None
    
    def __init__(self, root, split_type='train', transform=None, include_noise=True):
        super(QSMDataSet, self).__init__()
        self.root = root
        self.split_type = split_type
        self.transform = transform
        self.include_noise = include_noise
        
        # Noise parameters (same as original)
        self.Prob = torch.tensor(0.8)   ## 20% (1 - 0.8) probability to add noise
        self.SNRs = torch.tensor([50, 40, 20, 10, 5])  # Noise SNRs
        
        # Find all available data files
        self.files = []
        self._scan_data_directory()
        
        print(f"Found {len(self.files)} data pairs for {split_type} split")

    @property
    def current_subjects(self):
        """Property decorator to get current split subjects"""
        if self.split_type == 'train':
            return self.train_subjects
        elif self.split_type == 'val':
            return self.val_subjects
        elif self.split_type == 'test':
            return self.test_subjects if self.test_subjects else []
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

    @classmethod
    def get_train_subjects(cls):
        """Class method decorator to get training subjects"""
        return cls.train_subjects

    @classmethod
    def get_val_subjects(cls):
        """Class method decorator to get validation subjects"""
        return cls.val_subjects

    @classmethod
    def get_test_subjects(cls):
        """Class method decorator to get test subjects"""
        return cls.test_subjects if cls.test_subjects else []

    @classmethod
    def get_all_split_subjects(cls):
        """Class method to get all subjects organized by split"""
        return {
            'train': cls.train_subjects,
            'val': cls.val_subjects,
            'test': cls.test_subjects if cls.test_subjects else []
        }

    def _scan_data_directory(self):
        """Scan the data directory for current split subjects only"""
        current_subjects = self.current_subjects
        if not current_subjects:
            print(f"No subjects defined for {self.split_type} split")
            return
            
        for subject in current_subjects:
            # Create pattern for this specific subject
            input_pattern = os.path.join(self.root, f"{subject}/ses-*/qsm/*_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield.nii.gz")
            input_files = glob.glob(input_pattern)
            
            for input_file in input_files:
                # Extract subject and session from filename
                filename = os.path.basename(input_file)
                # Parse sub-XX_ses-XX from filename
                parts = filename.split('_')
                if len(parts) >= 2:
                    sub_id = parts[0]  # sub-XX
                    ses_id = parts[1]  # ses-XX
                    
                    # Construct the corresponding target file path
                    target_filename = f"{sub_id}_{ses_id}_unwrapped-SEGUE_mask-nfe_bfr-PDF_susc-autoNDI_Chimap.nii.gz"
                    target_file = os.path.join(os.path.dirname(input_file), target_filename)
                    
                    # Check if both files exist
                    if os.path.exists(input_file) and os.path.exists(target_file):
                        self.files.append({
                            "input": input_file,
                            "target": target_file,
                            "subject": sub_id,
                            "session": ses_id,
                            "name": f"{sub_id}_{ses_id}"
                        })
                    else:
                        print(f"Warning: Missing pair for {filename}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Load the data
        name = datafiles["name"]
        
        # Load NIfTI files (handles .gz compression automatically)
        input_nii = nib.load(datafiles["input"])
        target_nii = nib.load(datafiles["target"])
        
        # Get data arrays
        input_data = input_nii.get_fdata()  # get_fdata() is preferred over get_data()
        target_data = target_nii.get_fdata()
        
        # Convert to numpy arrays and then to torch tensors
        input_data = np.array(input_data, dtype=np.float32)
        target_data = np.array(target_data, dtype=np.float32)
        
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        
        # Add noise augmentation (optional)
        if self.include_noise:
            tmp = torch.rand(1)
            if tmp > self.Prob:
                tmp_mask = input_tensor != 0
                tmp_idx = torch.randint(5, (1,1))
                tmp_SNR = self.SNRs[tmp_idx]
                input_tensor = AddNoise(input_tensor, tmp_SNR)
        
        # Add channel dimension (batch_size will be added by DataLoader)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        target_tensor = torch.unsqueeze(target_tensor, 0)
        
        return input_tensor, target_tensor, name

    def get_file_info(self, index):
        """Get file information for a specific index"""
        return self.files[index]

    def get_all_subjects(self):
        """Get list of all unique subjects"""
        return list(set([f["subject"] for f in self.files]))

    def get_all_sessions(self):
        """Get list of all unique sessions"""
        return list(set([f["session"] for f in self.files]))

def AddNoise(ins, SNR):
    """Add noise to input tensor based on SNR"""
    sigPower = SigPower(ins)
    noisePower = sigPower / SNR
    noise = torch.sqrt(noisePower.float()) * torch.randn(ins.size()).float()
    return ins + noise


def SigPower(ins):
    """Calculate signal power"""
    ll = torch.numel(ins)
    tmp1 = torch.sum(ins ** 2)
    return torch.div(tmp1, ll)


# Test the dataloader
if __name__ == '__main__':
    # Update this path to your QSM data directory
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    BATCH_SIZE = 2

    try:
        # Create subset datasets
        train_dataset = QSMDataSet(DATA_DIRECTORY, split_type='train')
        val_dataset = QSMDataSet(DATA_DIRECTORY, split_type='val')
        
        # Create dataloaders
        trainloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"\nCreated dataloaders:")
        print(f"  Train: {len(trainloader)} batches")
        print(f"  Val: {len(valloader)} batches")
        
        # Test loading a few batches from each split
        print(f"\nTesting train dataloader:")
        for i, (inputs, targets, names) in enumerate(trainloader):
            if i < 2:  # Test first 2 batches
                print(f"  Batch {i+1}: {names} | Shapes: {inputs.shape}, {targets.shape}")
            else:
                break
        
        print(f"\nTesting validation dataloader:")
        for i, (inputs, targets, names) in enumerate(valloader):
            if i < 2:  # Test first 2 batches
                print(f"  Batch {i+1}: {names} | Shapes: {inputs.shape}, {targets.shape}")
            else:
                break
                
    except ValueError as e:
        print(f"Error: {e}")
