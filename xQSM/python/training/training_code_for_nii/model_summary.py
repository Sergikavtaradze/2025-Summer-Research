import torch
import torch.nn as nn
from torchinfo import summary
import sys
import os
from datetime import datetime
from contextlib import redirect_stdout
import io

# Add the current directory to the path to import xQSM and xQSM_blocks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xQSM import xQSM, get_parameter_number

def analyze_and_save_model_summary():
    """
    Analyze xQSM model and save detailed summary to text file
    """
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"xQSM_model_summary_{timestamp}.txt"
    
    print(f"Analyzing xQSM model and saving to: {output_filename}")
    
    with open(output_filename, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("xQSM Model Analysis Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write("\n")
        
        # Model configuration
        encoding_depth = 2
        ini_chNo = 32
        input_size = (1, 1, 48, 48, 48)
        batch_size = 1
        
        f.write("="*80 + "\n")
        f.write(f"xQSM Model Analysis - Encoding Depth: {encoding_depth}, Initial Channels: {ini_chNo}\n")
        f.write("="*80 + "\n")
        
        # Create the model
        model = xQSM(EncodingDepth=encoding_depth, ini_chNo=ini_chNo)
        
        # Get parameter count
        param_info = get_parameter_number(model)
        f.write(f"\nModel Parameter Count:\n")
        f.write(f"Total Parameters: {param_info['Total']:,}\n")
        f.write(f"Trainable Parameters: {param_info['Trainable']:,}\n")
        
        # Create input tensor with batch size
        full_input_size = (batch_size,) + input_size[1:]
        
        f.write(f"\nInput Shape: {full_input_size}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED MODEL SUMMARY\n")
        f.write("="*80 + "\n")
        
        # Comprehensive model summary
        try:
            # Capture torchinfo summary output
            summary_buffer = io.StringIO()
            with redirect_stdout(summary_buffer):
                model_summary = summary(
                    model=model,
                    input_size=full_input_size,
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"],
                    verbose=2
                )
            
            # Write the captured summary
            summary_output = summary_buffer.getvalue()
            f.write(summary_output)
            
            f.write("\n" + "="*80 + "\n")
            f.write("LAYER-BY-LAYER ANALYSIS FOR FREEZING\n")
            f.write("="*80 + "\n")
            
            f.write("\nLayers available for freezing:\n")
            f.write("-" * 50 + "\n")
            
            # Analyze each major component
            f.write("1. INPUT OCTAVE LAYER:\n")
            input_params = sum(p.numel() for p in model.InputOct.parameters())
            f.write(f"   - InputOct: {input_params:,} parameters\n")
            
            f.write("\n2. ENCODING LAYERS:\n")
            encoding_layer_params = []
            for i, encode_conv in enumerate(model.EncodeConvs):
                params = sum(p.numel() for p in encode_conv.parameters())
                encoding_layer_params.append(params)
                f.write(f"   - EncodeConv{i+1}: {params:,} parameters\n")
            
            f.write("\n3. MIDDLE LAYER:\n")
            mid_params = sum(p.numel() for p in model.MidConv1.parameters())
            f.write(f"   - MidConv1: {mid_params:,} parameters\n")
            
            f.write("\n4. DECODING LAYERS:\n")
            decoding_layer_params = []
            for i, decode_conv in enumerate(model.DecodeConvs):
                params = sum(p.numel() for p in decode_conv.parameters())
                decoding_layer_params.append(params)
                f.write(f"   - DecodeConv{i+1}: {params:,} parameters\n")
            
            f.write("\n5. FINAL LAYER:\n")
            final_params = sum(p.numel() for p in model.FinalOct.parameters())
            f.write(f"   - FinalOct: {final_params:,} parameters\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FREEZING RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            
            f.write("\nFor transfer learning on head and neck dataset:\n")
            f.write("- Consider freezing: Early encoding layers (EncodeConv1, EncodeConv2)\n")
            f.write("- Keep trainable: Later decoding layers and FinalOct for domain adaptation\n")
            f.write("- Middle layers: Can be frozen or kept trainable based on dataset similarity\n")
            
            total_encoding_params = sum(sum(p.numel() for p in conv.parameters()) for conv in model.EncodeConvs)
            total_decoding_params = sum(sum(p.numel() for p in conv.parameters()) for conv in model.DecodeConvs)
            
            f.write(f"\nParameter distribution:\n")
            f.write(f"- Encoding layers: {total_encoding_params:,} parameters ({total_encoding_params/param_info['Total']*100:.1f}%)\n")
            f.write(f"- Decoding layers: {total_decoding_params:,} parameters ({total_decoding_params/param_info['Total']*100:.1f}%)\n")
            f.write(f"- Other layers: {param_info['Total'] - total_encoding_params - total_decoding_params:,} parameters\n")
            
            # Additional freezing analysis
            f.write(f"\nTransfer Learning Impact Analysis:\n")
            frozen_params = input_params + total_encoding_params
            trainable_params = param_info['Total'] - frozen_params
            
            f.write(f"- If freezing InputOct + Encoding layers:\n")
            f.write(f"  * Frozen parameters: {frozen_params:,} ({frozen_params/param_info['Total']*100:.1f}%)\n")
            f.write(f"  * Trainable parameters: {trainable_params:,} ({trainable_params/param_info['Total']*100:.1f}%)\n")
            f.write(f"  * Training speedup: ~{param_info['Total']/trainable_params:.1f}x faster\n")
            
        except Exception as e:
            f.write(f"Error generating summary: {e}\n")
            f.write("Falling back to basic model structure analysis...\n")
            
            # Basic structure analysis if torchinfo fails
            f.write(f"\nModel structure:\n")
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    num_params = sum(p.numel() for p in module.parameters())
                    if num_params > 0:
                        f.write(f"{name}: {num_params} parameters\n")
        
        # Add freezing example code
        f.write("\n" + "="*80 + "\n")
        f.write("EXAMPLE: HOW TO FREEZE LAYERS\n")
        f.write("="*80 + "\n")
        
        f.write("""
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
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("="*80 + "\n")
        f.write("Use this information to decide which layers to freeze for your head and neck dataset.\n")
        f.write("Generally, freeze early layers (feature extractors) and fine-tune later layers.\n")
    
    print(f"Model summary has been saved to: {output_filename}")
    print("You can now review the detailed analysis in the text file.")
    
    return output_filename

if __name__ == "__main__":
    analyze_and_save_model_summary() 