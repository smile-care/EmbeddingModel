import json
import yaml
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import shutil
from collections import defaultdict
from pathlib import Path

import fiftyone as fo
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    data = yaml.load(open("data/zhenyu_data/patches/supcon_data_clean/similarity_config.yaml", "r"), Loader=yaml.FullLoader)
    print(data)
    
