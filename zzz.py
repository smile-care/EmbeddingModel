import json
import os
import os.path as osp
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.supcon_train.models.moco_queue import MoCoQueue

os.environ['QT_QPA_PLATFORM'] = 'offscreen'




if __name__ == "__main__":
    # 测试MoCoQueue
    queue_size = 10
    embedding_dim = 128
    moco_queue = MoCoQueue(queue_size=queue_size, embedding_dim=embedding_dim)
    
    batch_size = 2
    total_iters = 7
    
    # 模拟添加embeddings和labels
    for i in range(total_iters):
        time.sleep(0.01)  # 模拟时间间隔
        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, 10, (batch_size,))
 
        moco_queue.enqueue(embeddings, labels)
        
        print("当前队列labels:\n", moco_queue.labels)
        print("队列指针位置:", moco_queue.ptr.item())
        print("队列是否已满:", moco_queue.is_full())
        print("-" * 50)