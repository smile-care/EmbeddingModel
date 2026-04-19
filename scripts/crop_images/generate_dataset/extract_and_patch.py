"""
从zip文件解压并提取patch的完整流程：
1. 解压zip文件到临时目录 data/datasets/unzip_files
2. 从解压的图像和JSON标注文件提取patch
3. 可选：清理临时解压文件
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sys.path.insert(0, str(Path(__file__).parent.parent))

TEMP_UNZIP_DIR = "data/datasets/unzip_files"


# ──────────────────────────────────────────────
# 解压相关
# ──────────────────────────────────────────────

def _is_extract_complete(extract_dir: Path) -> bool:
    """通过verify文件校验解压是否完整。

    verify文件格式: {"image_count": N, ...}
    实际图像为目录中无后缀的文件（原始数据格式）。
    若无verify文件则仅检查目录非空。
    """
    verify_files = list(extract_dir.rglob("*.verify"))
    if not verify_files:
        return extract_dir.exists() and any(extract_dir.iterdir())

    for verify_file in verify_files:
        try:
            with open(verify_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            expected = meta.get("image_count", 0)
            label_dir = verify_file.parent
            # 无后缀文件即为图像
            actual = sum(1 for p in label_dir.iterdir() if p.is_file() and p.suffix == '')
            if actual < expected:
                return False
        except Exception:
            return False
    return True


def _extract_one_zip(args: tuple) -> tuple:
    """解压单个zip文件，返回 (status, zip_name, extract_dir)。供线程池调用。"""
    zip_file, extract_dir = args
    zip_file = Path(zip_file)
    extract_dir = Path(extract_dir)

    if extract_dir.exists() and _is_extract_complete(extract_dir):
        return ('skip', zip_file.name, str(extract_dir))
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return ('ok', zip_file.name, str(extract_dir))
    except Exception as e:
        return ('err', zip_file.name, str(e))


def extract_all_zips(root_dir: str, temp_dir: str, type_id: str, num_workers: int = 4) -> Path:
    """递归解压目录下所有匹配type_id的zip文件到临时目录，多线程并发，已完整解压则跳过"""
    root_path = Path(root_dir)
    zip_files = [
        f for f in root_path.rglob("*.zip")
        if type_id in f.parent.stem
    ]
    print(f"找到 {len(zip_files)} 个匹配zip文件")

    tasks = [
        (str(zf), str(Path(temp_dir) / zf.parent.stem / zf.stem))
        for zf in zip_files
    ]

    ok_count = skip_count = err_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_extract_one_zip, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="解压zip文件"):
            status, name, info = future.result()
            if status == 'skip':
                skip_count += 1
                tqdm.write(f"  ○ 跳过(已完整): {name}")
            elif status == 'ok':
                ok_count += 1
                tqdm.write(f"  ✓ {name}")
            else:
                err_count += 1
                tqdm.write(f"  ✗ 错误 {name}: {info}")

    print(f"解压完成: 新解压 {ok_count}, 跳过 {skip_count}, 失败 {err_count}")
    return Path(temp_dir)


# ──────────────────────────────────────────────
# Patch提取相关
# ──────────────────────────────────────────────

def extract_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def polygon_to_mask(polygon: List[List[float]], img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    points = np.array(polygon, dtype=np.float32)
    points[:, 0] *= img_w
    points[:, 1] *= img_h
    points = points.astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def extract_instances_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    instances = []
    for label_id in range(1, num_labels):
        instance_mask = (labels == label_id).astype(np.uint8) * 255
        instances.append(instance_mask)
    return instances


def compute_crop_region(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    crop_sizes: List[int] = [96, 160, 224]
) -> Tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    img_h, img_w = image_shape

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_max = max(bbox_w, bbox_h)
    min_required_crop_size = int(bbox_max * 2)

    if bbox_max < crop_sizes[0]:
        crop_size = max(crop_sizes[0], min_required_crop_size)
        use_expand = False
    elif bbox_max < crop_sizes[1]:
        crop_size = max(crop_sizes[1], min_required_crop_size)
        use_expand = False
    elif bbox_max < crop_sizes[2]:
        crop_size = max(crop_sizes[2], min_required_crop_size)
        use_expand = False
    else:
        crop_size = None
        use_expand = True

    if use_expand:
        min_padding = bbox_max / 2
        expand_w = int(min_padding)
        expand_h = int(min_padding)

        expanded_x_min = max(0, x_min - expand_w)
        expanded_y_min = max(0, y_min - expand_h)
        expanded_x_max = min(img_w, x_max + expand_w)
        expanded_y_max = min(img_h, y_max + expand_h)

        expanded_w = expanded_x_max - expanded_x_min
        expanded_h = expanded_y_max - expanded_y_min
        crop_size_final = max(expanded_w, expanded_h, min_required_crop_size)

        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2
        half_size = crop_size_final // 2
        crop_x_min = max(0, bbox_center_x - half_size)
        crop_y_min = max(0, bbox_center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size_final)
        crop_y_max = min(img_h, crop_y_min + crop_size_final)

        final_w = crop_x_max - crop_x_min
        final_h = crop_y_max - crop_y_min
        final_crop_size = min(final_w, final_h)
        if final_w != final_h:
            center_x = (crop_x_min + crop_x_max) // 2
            center_y = (crop_y_min + crop_y_max) // 2
            half_final = final_crop_size // 2
            crop_x_min = max(0, center_x - half_final)
            crop_y_min = max(0, center_y - half_final)
            crop_x_max = min(img_w, crop_x_min + final_crop_size)
            crop_y_max = min(img_h, crop_y_min + final_crop_size)
    else:
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        half_size = crop_size // 2
        crop_x_min = max(0, center_x - half_size)
        crop_y_min = max(0, center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size)
        crop_y_max = min(img_h, crop_y_min + crop_size)

        if crop_x_max - crop_x_min < crop_size:
            crop_x_min = max(0, crop_x_max - crop_size)
        if crop_y_max - crop_y_min < crop_size:
            crop_y_min = max(0, crop_y_max - crop_size)

        final_w = crop_x_max - crop_x_min
        final_h = crop_y_max - crop_y_min
        final_crop_size = min(final_w, final_h)
        if final_w != final_h:
            center_x = (crop_x_min + crop_x_max) // 2
            center_y = (crop_y_min + crop_y_max) // 2
            half_final = final_crop_size // 2
            crop_x_min = max(0, center_x - half_final)
            crop_y_min = max(0, center_y - half_final)
            crop_x_max = min(img_w, crop_x_min + final_crop_size)
            crop_y_max = min(img_h, crop_y_min + final_crop_size)

    return (int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max))


def extract_patches_from_image_json(
    image_path: str,
    json_path: str,
    output_dir: str,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    label: Optional[str] = None
) -> List[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    labels_data = json_data.get('labels', [])
    if len(labels_data) == 0:
        print(f"警告: {json_path} 中没有找到labels")
        return []

    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for label_data in labels_data:
        if label_data.get('isSubtract', False):
            continue
        points = label_data.get('points', [])
        if len(points) < 3:
            continue
        polygon_mask = polygon_to_mask(points, img_h, img_w)
        combined_mask = np.where(polygon_mask > 0, 255, combined_mask)

    for label_data in labels_data:
        if not label_data.get('isSubtract', False):
            continue
        points = label_data.get('points', [])
        if len(points) < 3:
            continue
        polygon_mask = polygon_to_mask(points, img_h, img_w)
        combined_mask = np.where(polygon_mask > 0, 0, combined_mask)

    instance_masks = extract_instances_from_mask(combined_mask)
    if len(instance_masks) == 0:
        print(f"警告: {json_path} 中没有找到任何instance")
        return []

    if label is None:
        json_name = Path(json_path).stem
        label = json_name.split('_')[-1] if '_' in json_name else 'defect'

    output_path = Path(output_dir) / label
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = Path(image_path).stem
    patches_info = []

    for idx, instance_mask in enumerate(instance_masks):
        bbox = extract_bbox_from_mask(instance_mask)
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        bbox_max = max(x_max - x_min, y_max - y_min)
        if bbox_max < min_size:
            continue

        crop_x_min, crop_y_min, crop_x_max, crop_y_max = compute_crop_region(
            bbox, (img_h, img_w), crop_sizes
        )

        patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        patch_mask = instance_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if patch.shape[0] != patch.shape[1]:
            max_dim = max(patch.shape[0], patch.shape[1])
            patch = cv2.resize(patch, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(patch_mask, (max_dim, max_dim), interpolation=cv2.INTER_NEAREST)

        patch_filename = f"{prefix}_{idx:03d}.png"
        patch_mask_filename = f"{prefix}_{idx:03d}_mask.png"
        Image.fromarray(patch).save(output_path / patch_filename)
        Image.fromarray(patch_mask).save(output_path / patch_mask_filename)

        patches_info.append({
            'patch_path': str(output_path / patch_filename),
            'patch_mask_path': str(output_path / patch_mask_filename),
            'original_image_path': str(image_path),
            'original_json_path': str(json_path),
            'instance_id': idx,
            'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'crop_bbox': [int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)],
            'original_size': [img_h, img_w],
            'patch_size': [patch.shape[0], patch.shape[1]]
        })

    return patches_info


def _process_one_json(args: tuple) -> tuple:
    """处理单个json/image对，返回 (patches_info, error_msg)。供进程池调用。"""
    json_path, output_dir, min_size, crop_sizes = args
    json_path = Path(json_path)
    image_path = json_path.with_suffix('')
    if not image_path.exists():
        return ([], f"找不到对应图像: {image_path}")
    label = image_path.parent.name
    try:
        patches_info = extract_patches_from_image_json(
            str(image_path), str(json_path), output_dir,
            min_size=min_size, crop_sizes=crop_sizes, label=label
        )
        return (patches_info, None)
    except Exception as e:
        return ([], f"处理 {image_path} 时出错: {e}")


def process_folder(
    input_dir: str,
    output_dir: str,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    num_workers: int = 8,
) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"输入文件夹不存在: {input_path}")

    json_files = list(input_path.rglob("*.json"))
    assert len(json_files) > 0, f"在 {input_dir} 中没有找到JSON文件"

    print(f"找到 {len(json_files)} 个JSON文件，使用 {num_workers} 个进程")

    tasks = [(str(jf), output_dir, min_size, crop_sizes) for jf in json_files]

    all_patches_info = []
    failed_count = 0
    processed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one_json, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="提取patch"):
            patches_info, err = future.result()
            if err:
                tqdm.write(f"  警告: {err}")
                failed_count += 1
            else:
                all_patches_info.extend(patches_info)
                processed_count += 1

    print(f"\n  成功: {processed_count} / {len(json_files)}, 失败: {failed_count}, 总patch数: {len(all_patches_info)}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run_pipeline(
    root_dir: str,
    save_dir: str,
    type_id: str,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    cleanup: bool = False,
    num_workers: int = 8,
) -> None:
    temp_dir = str(Path(TEMP_UNZIP_DIR) / type_id)
    print(f"\n[1/2] 解压zip文件到临时目录: {temp_dir}")
    extract_all_zips(root_dir, temp_dir, type_id, num_workers=num_workers)

    print(f"\n[2/2] 从解压文件提取patch -> {save_dir}")
    temp_path = Path(temp_dir)
    sub_dirs = [d for d in temp_path.iterdir() if d.is_dir()]

    if not sub_dirs:
        print(f"警告: 临时目录 {temp_dir} 中没有找到子文件夹")
        return

    for sub_dir in sub_dirs:
        # sub_dir.name 形如 "64201-CCD1-极柱R角"，作为输出的中间层文件夹
        sub_save_dir = str(Path(save_dir) / sub_dir.name)
        print(f"\n处理文件夹: {sub_dir} -> {sub_save_dir}")
        try:
            process_folder(str(sub_dir), sub_save_dir, min_size=min_size, crop_sizes=crop_sizes, num_workers=num_workers)
        except AssertionError as e:
            print(f"跳过: {e}")

    if cleanup:
        print(f"\n清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("清理完成")
    else:
        print(f"\n临时解压文件保留在: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description='解压zip并提取patch的完整流程')
    parser.add_argument('--root_dir', type=str,
                        default="/media/unitx/预训练模型数据_2T-2/预训练数据_3F/k3_02",
                        help='包含zip文件的根目录')
    parser.add_argument('--save_dir', type=str,
                        default="data/datasets/zhenyu_new",
                        help='patch输出目录')
    parser.add_argument('--type_id', type=str, nargs='+', default=["A34"],
                        help='zip文件过滤关键字（匹配父文件夹名），支持多个，空格分隔')
    parser.add_argument('--min_size', type=int, default=8, help='最小patch尺寸')
    parser.add_argument('--crop_sizes', type=int, nargs='+', default=[96, 160, 224],
                        help='裁剪尺寸区间')
    parser.add_argument('--cleanup', type=bool, default=True,
                        help='处理完成后删除临时解压文件')
    parser.add_argument('--custom_flag', type=str, default="3F",
                        help='自定义标识，用于输出目录命名')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='并发工作进程/线程数')

    args = parser.parse_args()

    custom_flag = args.custom_flag
    for type_id in args.type_id:
        save_dir = str(Path(args.save_dir) / f"{custom_flag}-{type_id}")
        run_pipeline(
            root_dir=args.root_dir,
            save_dir=save_dir,
            type_id=type_id,
            min_size=args.min_size,
            crop_sizes=args.crop_sizes,
            cleanup=args.cleanup,
            num_workers=args.num_workers,
        )


if __name__ == '__main__':
    main()
