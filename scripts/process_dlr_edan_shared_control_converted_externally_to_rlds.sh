#!/bin/bash
#SBATCH --partition=tibet --qos=normal
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --job-name="process_dlr_edan_shared_control_converted_externally_to_rlds"
#SBATCH --output=sample-%j.out

# Load environment
source /sailhome/yalcintr/openx/bin/activate

# Directories
DOWNLOAD_DIR=/vision/group/jointvla/raw
CONVERSION_DIR=/vision/group/jointvla/processed_downsample10x
N_WORKERS=30
MAX_EPISODES_IN_MEMORY=210

# Process dataset
python3 ../modify_rlds_dataset.py --dataset=dlr_edan_shared_control_converted_externally_to_rlds --data_dir=$DOWNLOAD_DIR --target_dir=$CONVERSION_DIR --mods=downsample10x,resize_and_jpeg_encode --n_workers=$N_WORKERS --max_episodes_in_memory=$MAX_EPISODES_IN_MEMORY

echo "Finished processing dlr_edan_shared_control_converted_externally_to_rlds"
