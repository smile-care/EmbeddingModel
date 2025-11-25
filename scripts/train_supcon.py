"""
启动SupCon监督对比学习训练
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.train_supcon import main

if __name__ == '__main__':
    main()

