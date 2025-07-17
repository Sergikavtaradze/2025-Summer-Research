#!/bin/bash -l
 
#specify the required resources
#$-q Arya, Bran, Bronn, Brienne, Catelyn, Cersei, Daario, Daenerys, Davos, Eddard, Gendry, Jaime, Jaqen, Jeor, Joffrey, Jorah, arya, bran, bronn, brienne, catelyn, cersei, daario, daenerys, davos, eddard, gendry, jaime, jaqen, jeor, joffrey, jorah, Ellaria, Margaery, Gilly, Jon, ellaria, gilly, jon
# RAM
#$ -l mem=16G
# GPU
#$ -l gpu=1
 
# Set the job name, output file paths
#$ -N xQSM_autocast_TL_bs4_ep50_lr4e-4
#$ -o /home/zcemska/Scratch/DeepLearningQSM/2025-Summer-Research/xQSM/python/job_info
#$ -e /home/zcemska/Scratch/DeepLearningQSM/2025-Summer-Research/xQSM/python/job_info
#$ -wd /home/zcemska/Scratch/DeepLearningQSM
 
# Activate the virtual environment
# Initialize Conda
# source /share/apps/source_files/python/python-3.9.16.source
# Activate the specific environment

##################################################################################
# Need to change the command to the correct path for the conda installation #
##################################################################################
source /home/zcemska/Scratch/DeepLearningQSM/2025-Summer-Research/QSM/bin/activate
 
########################################################
## CUDA Environment Setup
########################################################
 
# Add CUDA binary directories to PATH - enables system to find and execute CUDA tools (nvidia-smi, nvcc, etc.)
export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH}
 
# Set runtime library path - tells system where to find CUDA shared libraries during program execution
# This includes both shared (/share/apps) and local (/usr/local) CUDA installations
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH}
 
# Set CUDA include directory - specifies location of CUDA header files
# Used during compilation of CUDA programs (if needed)
export CUDA_INC_DIR=/share/apps/cuda-11.8/include
 
# Set compile-time library path - tells compiler where to find libraries during linking
# Similar to LD_LIBRARY_PATH but used at build time instead of runtime
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}
 
########################################################
## Finding available GPUs
########################################################
 
nvidia-smi
 
# Get the first available GPU ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 100 {print $1}' | head -n 1| tr -d ',')
 
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No available GPU found. Exiting..."
    exit 1
fi
 
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
 
# Add CUDA blocking for better error reporting
export CUDA_LAUNCH_BLOCKING=1
 
########################################################
 
 
# Navigate to the directory containing the scripts
cd 2025-Summer-Research/xQSM/python/training/training_code_for_nii
 
python3 Train_Autocast_TL.py -bs 4 -ep 50 -lr 4e-4 \
--data_directory "QSM_data" \
--pretrained_path "xQSM/Pretrained_Checkpoints/xQSM_invivo.pth" \
--snapshot_path "xQSM/python/training/ckpt/" 