import sys
from pathlib import Path

import yaml


def build_entry(scene_path: Path, rel_base: Path):
    if not scene_path.exists():
        print(f"[错误] 目录不存在: {scene_path}", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted(d for d in scene_path.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"[错误] {scene_path} 下没有类别子目录", file=sys.stderr)
        sys.exit(1)

    return {
        "name": f"{scene_path.parent.name}_{scene_path.name}",
        "root": str(scene_path.relative_to(rel_base)),
    }


def main():
    root = 'data/datasets/zhenyu'
    output = "data/datasets/zhenyu/config.yaml"
    

    rel_base = Path(__file__).absolute().parents[1]
    root = (rel_base / root).absolute()
    output_path = (rel_base / output).absolute() if output else None
    
    all_entries = []
    for group_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if group_dir.name in ['2025-1201-5.x', '2025-1219', '2026-0105']:
            print(f"跳过目录: {group_dir}")
            continue
        for scene_path in sorted(d for d in group_dir.iterdir() if d.is_dir()):
            all_entries.append(build_entry(scene_path, rel_base))
    
    
    config = {"scenes": {"train": all_entries}, "debug_mode": False}
    yaml_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2)


    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_str, encoding="utf-8")
        print(f"已保存到: {output}\n")
    print(yaml_str)


if __name__ == "__main__":
    main()
