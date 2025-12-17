import json
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from PIL import Image
from tqdm import tqdm

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


if __name__ == "__main__":
    data_dir = Path('data/datasets/zhenyu/train/R角凸起')
    output_dir = Path('data/datasets/zhenyu/val/R角凸起')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_mask_list = []
    for img_p in list(data_dir.rglob('*.png')):
        if "_mask.png" in img_p.name:
            continue
        mask_p = img_p.with_name(img_p.stem + "_mask.png")
        if not mask_p.exists():
            print(f"缺失掩码文件: {mask_p}")
            continue
        image_mask_list.append((img_p, mask_p))
    
    print(f"总图像数量: {len(image_mask_list)}")
    
    # exit()
    
    random.shuffle(image_mask_list)
    
    val_list = random.sample(image_mask_list, k=10)
    for img_p, mask_p in val_list:
        src_img_path = str(img_p)
        src_mask_path = str(mask_p)
        
        shutil.move(src_img_path, str(output_dir))
        shutil.move(src_mask_path, str(output_dir))
    
    print(f"已移动 {len(val_list)} 张图像及其掩码到验证集目录: {output_dir}")