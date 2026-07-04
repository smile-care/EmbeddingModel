# train
torchrun --nproc_per_node=4 scripts_training/train_supcon.py
    # --resume checkpoints/supcon_models/0421/current_model.pth


# val
# python scripts_training/eval_supcon.py \
#   --config configs/supcon_config.yaml \
#   --checkpoint checkpoints/supcon_models/xxx/final_model.pth


