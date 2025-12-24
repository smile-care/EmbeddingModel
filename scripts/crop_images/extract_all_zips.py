import os
import os.path as osp
import zipfile
from pathlib import Path

from tqdm import tqdm

os.environ['QT_QPA_PLATFORM'] = 'offscreen'




def extract_all_zips(root_dir):
    """递归解压目录下所有zip文件"""
    root_path = Path(root_dir)
    zip_files = list(root_path.rglob("*.zip"))
    
    print(f"找到 {len(zip_files)} 个zip文件")
    
    for zip_file in tqdm(zip_files):
        try:
            extract_dir = zip_file.parent / zip_file.stem
            print(f"正在解压: {zip_file.name} -> {extract_dir}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✓ 完成: {zip_file.name}")
        except Exception as e:
            print(f"✗ 错误 {zip_file.name}: {e}")


if __name__ == "__main__":
    target_dir = "/home/unitx/workspace_custom/data/震裕/4.x/60194/4xdata"
    extract_all_zips(target_dir)