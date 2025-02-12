#!/bin/bash

# List of dataset transformations
DATASET_TRANSFORMS=(
    "fractal20220817_data 0.1.0 downsample10x,resize_and_jpeg_encode"
    "bridge 0.1.0 downsample10x,resize_and_jpeg_encode"
    "kuka 0.1.0 downsample10x,resize_and_jpeg_encode,filter_success"
    "taco_play 0.1.0 downsample10x,resize_and_jpeg_encode"
    "jaco_play 0.1.0 downsample10x,resize_and_jpeg_encode"
    "berkeley_cable_routing 0.1.0 downsample10x,resize_and_jpeg_encode"
    "roboturk 0.1.0 downsample10x,resize_and_jpeg_encode"
    "viola 0.1.0 downsample10x,resize_and_jpeg_encode"
    "berkeley_autolab_ur5 0.1.0 downsample10x,resize_and_jpeg_encode,flip_wrist_image_channels"
    "toto 0.1.0 downsample10x,resize_and_jpeg_encode"
    "language_table 0.1.0 downsample10x,resize_and_jpeg_encode"
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "bc_z 0.1.0 downsample10x,resize_and_jpeg_encode"
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 downsample10x,resize_and_jpeg_encode"
    "utaustin_mutex 0.1.0 downsample10x,resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
    "berkeley_fanuc_manipulation 0.1.0 downsample10x,resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
    "cmu_stretch 0.1.0 downsample10x,resize_and_jpeg_encode"
    "dobbe 0.0.1 downsample10x,resize_and_jpeg_encode"
    "fmb 0.0.1 downsample10x,resize_and_jpeg_encode"
    "droid 1.0.0 downsample10x,resize_and_jpeg_encode"
)

# Generate 27 separate SLURM scripts
for i in {0..26}; do
  strings=(${DATASET_TRANSFORMS[i]})
  DATASET=${strings[0]}
  VERSION=${strings[1]}
  TRANSFORM=${strings[2]}

  cat <<EOT > process_${DATASET}.sh
#!/bin/bash
#SBATCH --partition=tibet --qos=normal
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --job-name="process_${DATASET}"
#SBATCH --output=sample-%j.out

# Load environment
source /sailhome/yalcintr/openx/bin/activate

# Directories
DOWNLOAD_DIR=/vision/group/jointvla/raw
CONVERSION_DIR=/vision/group/jointvla/processed_downsample10x
N_WORKERS=30
MAX_EPISODES_IN_MEMORY=210

# Process dataset
python3 modify_rlds_dataset.py --dataset=$DATASET --data_dir=\$DOWNLOAD_DIR --target_dir=\$CONVERSION_DIR --mods=$TRANSFORM --n_workers=\$N_WORKERS --max_episodes_in_memory=\$MAX_EPISODES_IN_MEMORY

echo "Finished processing $DATASET"
EOT

  chmod +x process_${DATASET}.sh
done
