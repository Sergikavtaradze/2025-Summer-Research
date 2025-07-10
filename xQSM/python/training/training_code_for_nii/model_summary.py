import torch
import torch.nn as nn
from torchinfo import summary
import sys
import os

# Add the current directory to the path to import xQSM and xQSM_blocks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xQSM import xQSM, get_parameter_number

def analyze_xqsm_model(encoding_depth=2, ini_chNo=32, input_size=(1, 1, 48, 48, 48), batch_size=1):
    """
    Analyze the xQSM model structure and provide detailed summary
    
    Args:
        encoding_depth (int): Depth of encoding layers
        ini_chNo (int): Initial number of channels
        input_size (tuple): Input tensor dimensions (batch, channels, depth, height, width)
        batch_size (int): Batch size for analysis
    """
    
    print("="*80)
    print(f"xQSM Model Analysis - Encoding Depth: {encoding_depth}, Initial Channels: {ini_chNo}")
    print("="*80)
    
    # Create the model
    model = xQSM(EncodingDepth=encoding_depth, ini_chNo=ini_chNo)
    
    # Get parameter count
    param_info = get_parameter_number(model)
    print(f"\nModel Parameter Count:")
    print(f"Total Parameters: {param_info['Total']:,}")
    print(f"Trainable Parameters: {param_info['Trainable']:,}")
    
    # Create input tensor with batch size
    full_input_size = (batch_size,) + input_size[1:]  # (batch_size, channels, depth, height, width)
    
    print(f"\nInput Shape: {full_input_size}")
    print("\n" + "="*80)
    print("DETAILED MODEL SUMMARY")
    print("="*80)
    
    # Comprehensive model summary
    try:
        model_summary = summary(
            model=model,
            input_size=full_input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
            verbose=2
        )
        
        print("\n" + "="*80)
        print("LAYER-BY-LAYER ANALYSIS FOR FREEZING")
        print("="*80)
        
        print("\nLayers available for freezing:")
        print("-" * 50)
        
        # Analyze each major component
        print("1. INPUT OCTAVE LAYER:")
        print(f"   - InputOct: {sum(p.numel() for p in model.InputOct.parameters()):,} parameters")
        
        print("\n2. ENCODING LAYERS:")
        for i, encode_conv in enumerate(model.EncodeConvs):
            params = sum(p.numel() for p in encode_conv.parameters())
            print(f"   - EncodeConv{i+1}: {params:,} parameters")
        
        print("\n3. MIDDLE LAYER:")
        mid_params = sum(p.numel() for p in model.MidConv1.parameters())
        print(f"   - MidConv1: {mid_params:,} parameters")
        
        print("\n4. DECODING LAYERS:")
        for i, decode_conv in enumerate(model.DecodeConvs):
            params = sum(p.numel() for p in decode_conv.parameters())
            print(f"   - DecodeConv{i+1}: {params:,} parameters")
        
        print("\n5. FINAL LAYER:")
        final_params = sum(p.numel() for p in model.FinalOct.parameters())
        print(f"   - FinalOct: {final_params:,} parameters")
        
        print("\n" + "="*80)
        print("FREEZING RECOMMENDATIONS")
        print("="*80)
        
        print("\nFor transfer learning on head and neck dataset:")
        print("- Consider freezing: Early encoding layers (EncodeConv1, EncodeConv2)")
        print("- Keep trainable: Later decoding layers and FinalOct for domain adaptation")
        print("- Middle layers: Can be frozen or kept trainable based on dataset similarity")
        
        total_encoding_params = sum(sum(p.numel() for p in conv.parameters()) for conv in model.EncodeConvs)
        total_decoding_params = sum(sum(p.numel() for p in conv.parameters()) for conv in model.DecodeConvs)
        
        print(f"\nParameter distribution:")
        print(f"- Encoding layers: {total_encoding_params:,} parameters ({total_encoding_params/param_info['Total']*100:.1f}%)")
        print(f"- Decoding layers: {total_decoding_params:,} parameters ({total_decoding_params/param_info['Total']*100:.1f}%)")
        print(f"- Other layers: {param_info['Total'] - total_encoding_params - total_decoding_params:,} parameters")
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        print("Falling back to basic model structure analysis...")
        
        # Basic structure analysis if torchinfo fails
        print(f"\nModel structure:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name}: {num_params} parameters")

def create_freezing_example():
    """
    Create an example of how to freeze layers
    """
    print("\n" + "="*80)
    print("EXAMPLE: HOW TO FREEZE LAYERS")
    print("="*80)
    
    print("""
# Example code to freeze specific layers:

model = xQSM(EncodingDepth=2, ini_chNo=32)

# Freeze early encoding layers (feature extractors)
for param in model.InputOct.parameters():
    param.requires_grad = False

for param in model.EncodeConvs[0].parameters():  # First encoding layer
    param.requires_grad = False

# Optional: Freeze second encoding layer too
# for param in model.EncodeConvs[1].parameters():
#     param.requires_grad = False

# Keep decoding layers trainable for domain adaptation
# (No changes needed - they remain trainable by default)

# Verify which parameters are frozen:
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} parameters")
""")

if __name__ == "__main__":
    # Analyze with different configurations
    print("Analyzing xQSM model for transfer learning...")
    
    # Default configuration from your original code
    analyze_xqsm_model(encoding_depth=2, ini_chNo=32)
    
    # Create freezing example
    create_freezing_example()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Use this information to decide which layers to freeze for your head and neck dataset.")
    print("Generally, freeze early layers (feature extractors) and fine-tune later layers.") 