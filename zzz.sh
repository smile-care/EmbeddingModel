# python scripts/crop_images/generate_dataset/extract_and_patch.py \
#     --num_workers 16 \
#     --save_dir data/datasets/zhenyu_new \
#     --root_dir /media/unitx/预训练模型数据_2T-2/预训练数据_3F/K3_02 \
#     --custom_flag 3F_K3_02 \
#     --type_id A34



torchrun --nproc_per_node=4 scripts/train_supcon.py
    # --resume checkpoints/supcon_models/0421/current_model.pth
