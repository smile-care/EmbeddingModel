import json
import os
import os.path as osp
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from tqdm import tqdm

os.environ['QT_QPA_PLATFORM'] = 'offscreen'




if __name__ == "__main__":
    image_path = "data/zhenyu_data/60194/60194-CCD1-负极/顶盖正面_划伤/71ef3aa1-8867-403b-924a-10a7ba65ccd4_000_mask.png"
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((224, 224))
    plt.imshow(resized_img)
    plt.axis('off')
    plt.show()