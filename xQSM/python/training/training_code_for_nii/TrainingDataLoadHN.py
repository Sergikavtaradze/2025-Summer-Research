import os
import numpy as np
import random
import torch
import nibabel as nib
from torch.utils import data
import glob


class QSMDataSet(data.Dataset):
    def __init__(self, root, transform=None, include_noise=True):
        super(QSMDataSet, self).__init__()
        self.root = root
        self.transform = transform
        self.include_noise = include_noise
        
        # Noise parameters (same as original)
        self.Prob = torch.tensor(0.8)   ## 20% (1 - 0.8) probability to add noise
        self.SNRs = torch.tensor([50, 40, 20, 10, 5])  # Noise SNRs
        
        # Find all available data files
        self.files = []
        self._scan_data_directory()
        
        print(f"Found {len(self.files)} data pairs")

    def _scan_data_directory(self):
        """Scan the data directory to find all available subject/session combinations"""
        # Pattern to find all input files
        input_pattern = os.path.join(self.root, "sub-*/ses-*/qsm/*_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield.nii.gz")
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
    
    # Create dataset
    dataset = QSMDataSet(DATA_DIRECTORY, include_noise=True)
    print(f"Dataset length: {dataset.__len__()}")
    
    # Print some dataset info
    print("Available subjects:", dataset.get_all_subjects())
    print("Available sessions:", dataset.get_all_sessions())
    
    # Create dataloader
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Test loading a few batches
    for i, (inputs, targets, names) in enumerate(dataloader):
        if i < 3:  # Test first 3 batches
            print(f"\nBatch {i+1}:")
            print(f"Names: {names}")
            print(f"Input shape: {inputs.shape}")
            print(f"Target shape: {targets.shape}")
            print(f"Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
            print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        else:
            break
