import os
import sys
from pathlib import Path

import yaml


def parse_path(root_path: Path):
    if not root_path.exists():
        print(f"[错误] 目录不存在: {root_path}", file=sys.stderr)
        sys.exit(1)

    scene_dirs = sorted(d for d in root_path.iterdir() if d.is_dir())
    if not scene_dirs:
        print(f"[错误] {root_path} 下没有子目录", file=sys.stderr)
        sys.exit(1)

    rel_base = Path.cwd()
    train_entries = [
        {"name": f"{d.parts[-2]}_{d.parts[-1]}", "root": str(d.relative_to(rel_base))}
        for d in scene_dirs
    ]
    
    return train_entries


def main():
    root = 'data/datasets/zhenyu/E0_2'
    output = "data/datasets/zhenyu/E0_2/config.yaml"
    

    root = Path(root).resolve()
    
    list_dirs = os.listdir(root)
    
    all_entries = []
    for dir_name in list_dirs:
        dir_path = root / dir_name
        train_entries = parse_path(dir_path)
        all_entries.extend(train_entries)
    
    
    config = {"scenes": {"train": all_entries}, "debug_mode": False}
    yaml_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2)


    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(yaml_str, encoding="utf-8")
        print(f"已保存到: {output}\n")
    print(yaml_str)


if __name__ == "__main__":
    main()
