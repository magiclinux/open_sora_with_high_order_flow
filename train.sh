export HF_HOME="/nyx-storage1/hanliu/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

pip install -v .

torchrun --standalone --nproc_per_node 4 scripts/train.py \
    configs/opensora-v1-2/train/stage1.py --data-path /home/mzh1800/Open-Sora/datasets/pexels_45k/pexels_45k_popular_2.csv