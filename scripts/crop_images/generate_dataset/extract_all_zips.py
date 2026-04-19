import os
import os.path as osp
import zipfile
from pathlib import Path

from tqdm import tqdm

os.environ['QT_QPA_PLATFORM'] = 'offscreen'




def extract_all_zips(root_dir, save_dir, type_id):
    """递归解压目录下所有zip文件"""
    root_path = Path(root_dir)
    zip_files = list(root_path.rglob("*.zip"))
    
    print(f"找到 {len(zip_files)} 个zip文件")
    
    for zip_file in tqdm(zip_files):
        if type_id not in zip_file.parent.stem:
            continue
        try:
            extract_dir = Path(save_dir) / zip_file.parent.stem / zip_file.stem
            print(f"正在解压: {zip_file.name} -> {extract_dir}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✓ 完成: {zip_file.name}")
        except Exception as e:
            print(f"✗ 错误 {zip_file.name}: {e}")


if __name__ == "__main__":
    type_id = "64201"
    target_dir = "/media/unitx/预训练模型数据_2T-2/预训练数据_3F/二代训练机-01"
    save_dir = f"data/zhenyu_data/3F-二代训练机-01/{type_id}"
    extract_all_zips(target_dir, save_dir, type_id)