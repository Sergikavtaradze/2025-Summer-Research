#!/bin/bash

# Example usage for normal xQSM (no SE)
# python3 run_val_eval.py \
#   --data_directory "/path/to/QSM_data" \
#   --weights_path "/path/to/xQSM_weights.pth" \
#   --batch_size 1 \

# Example usage for xQSM + SE
# python3 run_val_eval.py \
#   --data_directory "/path/to/QSM_data" \
#   --weights_path "/path/to/xQSM_SE_weights.pth" \
#   --batch_size 1 \
#   --squeeze_exc

# Activate conda environment if needed
eval "$(/SAN/medic/CARES/mobarak/venvs/anaconda3/bin/conda shell.bash hook)"
conda activate 3DSAM-adapter

# Print arguments for clarity
echo "Running validation evaluation with the following arguments:"
echo "  Data directory: $1"
echo "  Weights path: $2"
echo "  Batch size: $3"
echo "  Encoding depth: $7"
echo "  Initial channels: $8"
echo "  Squeeze-and-Excitation: $9"

# Example actual run (edit as needed)
# python3 run_val_eval.py \
#   --data_directory "$1" \
#   --weights_path "$2" \
#   --batch_size "$3" \
#   $9 

python run_val_eval.py \
  --data_directory "/cluster/project7/SAMed/xQSM/2025-Summer-Research/QSM_data" \
  --weights_path "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/xQSM/ckpt/Aug8/ckpt/Aug7_bs32_ep100_lr4e-5_ps48_xQSM_SE/xQSM_TransferLearning_Best.pth" \
  --save_dir "./Aug7_bs32_ep100_lr4e-5_ps48_xQSM_SE"
