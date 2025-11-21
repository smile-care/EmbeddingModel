"""
启动MAE自监督训练
"""
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ssl_pretrain.train_ssl import main

if __name__ == '__main__':
    main()

