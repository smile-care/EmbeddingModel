import glob
import json
import os
import os.path as osp
import shutil
from collections import defaultdict
from pathlib import Path

import fiftyone as fo
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    image_dir = "data/zhenyu_data/patches/supcon_labels/顶盖正面"
    
    image_list = list(Path(image_dir).glob("*/*.png"))
    print(f"Found {len(image_list)} images in {image_dir}")
    
    annotations = {}
    for img_path in tqdm(image_list):
        img_path = str(img_path)
        if img_path.endswith("_mask.png"):
            continue
        img_path = Path(img_path).absolute()
        mask_path = img_path.with_name(img_path.stem + "_mask.png")
        if not mask_path.exists():
            print(f"Mask not found for image: {img_path}")
            continue
        label = img_path.parent.name
        annotations[str(img_path)] = [label, mask_path]
    
    print(f"Total annotations: {len(annotations)}")
    
    # Create samples for your data
    samples = []
    for filepath, (label, mask_path) in annotations.items():
        sample = fo.Sample(filepath=filepath)

        # Store segmentation in a field name of your choice
        sample["ground_truth"] = fo.Classification(label=label)
        sample["segmentation1"] = fo.Segmentation(filepath=str(mask_path))

        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset("my-dataset")
    dataset.add_samples(samples)
    
    session = fo.launch_app(dataset)