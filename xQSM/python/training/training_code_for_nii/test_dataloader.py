#!/usr/bin/env python3
"""
Comprehensive test script for QSM DataLoader
"""

import sys
import os
import time
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# Add the current directory to Python path to import TrainingDataLoad
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TrainingDataLoad import QSMDataSet

class Logger:
    """Logger class to write output to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def test_basic_functionality():
    """Test basic dataloader functionality"""
    print("="*60)
    print("TESTING BASIC DATALOADER FUNCTIONALITY")
    print("="*60)
    
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    
    try:
        # Test dataset creation
        print("1. Creating dataset...")
        dataset = QSMDataSet(DATA_DIRECTORY, include_noise=False)
        print("✓ Dataset created successfully")
        print(f"✓ Found {len(dataset)} data pairs")
        
        if len(dataset) == 0:
            print("ERROR: No data found! Check your data directory path.")
            return False
            
        # Test dataset info methods
        print(f"✓ Available subjects: {dataset.get_all_subjects()}")
        print(f"✓ Available sessions: {dataset.get_all_sessions()}")
        
        # Test individual item loading
        print("\n2. Testing individual item loading...")
        item = dataset[0]
        input_tensor, target_tensor, name = item
        
        print(f"✓ Successfully loaded item 0: {name}")
        print(f"✓ Input tensor shape: {input_tensor.shape}")
        print(f"✓ Target tensor shape: {target_tensor.shape}")
        print(f"✓ Input data type: {input_tensor.dtype}")
        print(f"✓ Target data type: {target_tensor.dtype}")
        
        # Check for reasonable data ranges
        print(f"✓ Input range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
        print(f"✓ Target range: [{target_tensor.min():.6f}, {target_tensor.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"ERROR in basic functionality test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_batching():
    """Test DataLoader with batching"""
    print("\n" + "="*60)
    print("TESTING DATALOADER BATCHING")
    print("="*60)
    
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    BATCH_SIZE = 2
    
    try:
        dataset = QSMDataSet(DATA_DIRECTORY, include_noise=False)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"✓ DataLoader created with batch size {BATCH_SIZE}")
        
        # Test first few batches
        for i, (inputs, targets, names) in enumerate(dataloader):
            if i < 3:  # Test first 3 batches
                print(f"\nBatch {i+1}:")
                print(f"  Names: {names}")
                print(f"  Input batch shape: {inputs.shape}")
                print(f"  Target batch shape: {targets.shape}")
                print(f"  Input range: [{inputs.min():.6f}, {inputs.max():.6f}]")
                print(f"  Target range: [{targets.min():.6f}, {targets.max():.6f}]")
                
                # Check that batch dimension is correct
                expected_batch_size = min(BATCH_SIZE, len(dataset) - i * BATCH_SIZE)
                if inputs.shape[0] != expected_batch_size:
                    print(f"ERROR: Expected batch size {expected_batch_size}, got {inputs.shape[0]}")
                    return False
                else:
                    print(f"  ✓ Correct batch size: {inputs.shape[0]}")
            else:
                break
                
        print("✓ Batching test completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in batching test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_noise_augmentation():
    """Test noise augmentation functionality"""
    print("\n" + "="*60)
    print("TESTING NOISE AUGMENTATION")
    print("="*60)
    
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    
    try:
        # Test without noise
        dataset_no_noise = QSMDataSet(DATA_DIRECTORY, include_noise=False)
        input_no_noise, _, name = dataset_no_noise[0]
        
        # Test with noise (multiple samples to see if noise is added)
        dataset_with_noise = QSMDataSet(DATA_DIRECTORY, include_noise=True)
        
        # Load the same item multiple times to see if noise varies
        inputs_with_noise = []
        for _ in range(5):
            input_with_noise, _, _ = dataset_with_noise[0]
            inputs_with_noise.append(input_with_noise)
        
        print(f"✓ Original input range: [{input_no_noise.min():.6f}, {input_no_noise.max():.6f}]")
        
        # Check if any of the noisy versions are different from the original
        noise_added = False
        for i, noisy_input in enumerate(inputs_with_noise):
            if not torch.allclose(input_no_noise, noisy_input, atol=1e-6):
                noise_added = True
                print(f"✓ Noise detected in sample {i+1}: [{noisy_input.min():.6f}, {noisy_input.max():.6f}]")
                break
        
        if noise_added:
            print("✓ Noise augmentation is working (detected differences)")
        else:
            print("NOTE: No noise was added in this test (20% probability)")
            
        return True
        
    except Exception as e:
        print(f"ERROR in noise augmentation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test loading performance"""
    print("\n" + "="*60)
    print("TESTING LOADING PERFORMANCE")
    print("="*60)
    
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    BATCH_SIZE = 4
    
    try:
        dataset = QSMDataSet(DATA_DIRECTORY, include_noise=False)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Time loading first 10 batches
        num_batches_to_test = min(10, len(dataloader))
        
        print(f"Timing {num_batches_to_test} batches...")
        start_time = time.time()
        
        for i, (inputs, targets, names) in enumerate(dataloader):
            if i >= num_batches_to_test:
                break
            # Just access the data to ensure it's loaded
            _ = inputs.shape, targets.shape
        
        end_time = time.time()
        total_time = end_time - start_time
        time_per_batch = total_time / num_batches_to_test
        samples_per_second = (num_batches_to_test * BATCH_SIZE) / total_time
        
        print(f"✓ Total time for {num_batches_to_test} batches: {total_time:.2f} seconds")
        print(f"✓ Average time per batch: {time_per_batch:.3f} seconds")
        print(f"✓ Samples per second: {samples_per_second:.1f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in performance test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_file_paths():
    """Test that all file paths are valid"""
    print("\n" + "="*60)
    print("TESTING FILE PATHS")
    print("="*60)
    
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    
    try:
        dataset = QSMDataSet(DATA_DIRECTORY, include_noise=False)
        
        print(f"Checking file paths for {len(dataset)} items...")
        
        for i in range(len(dataset)):
            file_info = dataset.get_file_info(i)
            input_path = file_info["input"]
            target_path = file_info["target"]
            
            if not os.path.exists(input_path):
                print(f"ERROR: Input file does not exist: {input_path}")
                return False
                
            if not os.path.exists(target_path):
                print(f"ERROR: Target file does not exist: {target_path}")
                return False
                
            if i < 5:  # Print first 5 for verification with filename differences
                print(f"\n✓ Item {i}: {file_info['name']}")
                input_filename = os.path.basename(input_path)
                target_filename = os.path.basename(target_path)
                print(f"  Input:  {input_filename}")
                print(f"  Target: {target_filename}")
                
                # Show the key differences in filenames
                print(f"  Filename differences:")
                print(f"     Input contains:  'localfield.nii.gz'")
                print(f"     Target contains: 'susc-autoNDI_Chimap.nii.gz'")
                
                # Extract and show the common prefix
                if input_filename.startswith(target_filename.split('_unwrapped')[0]):
                    common_prefix = target_filename.split('_unwrapped')[0]
                    print(f"     Common prefix: '{common_prefix}'")
                    input_suffix = input_filename.replace(common_prefix + '_', '')
                    target_suffix = target_filename.replace(common_prefix + '_', '')
                    print(f"     Input suffix:  '{input_suffix}'")
                    print(f"     Target suffix: '{target_suffix}'")
        
        print(f"\n✓ All {len(dataset)} file paths are valid")
        return True
        
    except Exception as e:
        print(f"ERROR in file path test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test data consistency and shape matching"""
    print("\n" + "="*60)
    print("TESTING DATA CONSISTENCY")
    print("="*60)
    
    DATA_DIRECTORY = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    
    try:
        dataset = QSMDataSet(DATA_DIRECTORY, include_noise=False)
        
        # Check first few items for shape consistency
        shapes_input = []
        shapes_target = []
        
        num_to_check = min(5, len(dataset))
        
        for i in range(num_to_check):
            input_tensor, target_tensor, name = dataset[i]
            shapes_input.append(input_tensor.shape)
            shapes_target.append(target_tensor.shape)
            
            print(f"Item {i} ({name}):")
            print(f"  Input shape:  {input_tensor.shape}")
            print(f"  Target shape: {target_tensor.shape}")
            
            # Check that input and target have same spatial dimensions
            if input_tensor.shape[1:] != target_tensor.shape[1:]:  # Skip channel dimension
                print(f"ERROR: Spatial dimensions don't match for {name}")
                print(f"  Input spatial:  {input_tensor.shape[1:]}")
                print(f"  Target spatial: {target_tensor.shape[1:]}")
                return False
            else:
                print(f"  ✓ Spatial dimensions match: {input_tensor.shape[1:]}")
        
        # Check if all shapes are consistent across dataset
        if len(set(shapes_input)) == 1:
            print(f"✓ All input tensors have consistent shape: {shapes_input[0]}")
        else:
            print(f"WARNING: Input tensor shapes vary: {set(shapes_input)}")
            
        if len(set(shapes_target)) == 1:
            print(f"✓ All target tensors have consistent shape: {shapes_target[0]}")
        else:
            print(f"WARNING: Target tensor shapes vary: {set(shapes_target)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in data consistency test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"dataloader_test_output_{timestamp}.txt"
    
    # Set up logging to both console and file
    logger = Logger(log_filename)
    sys.stdout = logger
    
    try:
        print("Starting comprehensive QSM DataLoader tests...")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output will be saved to: {log_filename}")
        print()
        
        tests = [
            ("Basic Functionality", test_basic_functionality),
            ("DataLoader Batching", test_dataloader_batching),
            ("Noise Augmentation", test_noise_augmentation),
            ("File Paths", test_file_paths),
            ("Data Consistency", test_data_consistency),
            ("Performance", test_performance),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"CRITICAL ERROR in {test_name}: {str(e)}")
                results[test_name] = False
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = 0
        total = len(tests)
        
        for test_name, result in results.items():
            status = "PASSED" if result else "FAILED"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if passed == total:
            print("All tests passed! Your dataloader is working correctly.")
        else:
            print("Some tests failed. Please check the output above for details.")
        
        success = passed == total
        
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nTest output saved to: {log_filename}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 