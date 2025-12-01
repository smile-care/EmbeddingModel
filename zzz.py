import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    save_dir = "data/zhenyu_data/patches/supcon_labels"
    json_data = json.load(open("data/zhenyu_data/metadata/supcon_dataset.json", 'r', encoding='utf-8'))
    
    total_patches = json_data['total_patches']
    patches = json_data['patches']
    
    
    view2paths = defaultdict(list)
    for patch in tqdm(patches):
        label = patch['label']
        view = label.split('_')[0]
        # patch_path = patch['patch_path']
        # patch_mask_path = patch['patch_mask_path']
        
        # save_path_dir = Path(save_dir) / view / label
        # save_path_dir.mkdir(parents=True, exist_ok=True)
        # save_patch_path = save_path_dir / Path(patch_path).name
        # save_patch_mask_path = save_path_dir / Path(patch_mask_path).name
        # shutil.copy(patch_path, save_patch_path)
        # shutil.copy(patch_mask_path, save_patch_mask_path)
        
        view2paths[view].append(patch)
    
    
    for view, patches in view2paths.items():
        save_json_path = Path(save_dir) / view / "metadata.json"
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        view_json_data = {
            'total_patches': len(patches),
            'patches': patches
        }
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(view_json_data, f, ensure_ascii=False, indent=4)