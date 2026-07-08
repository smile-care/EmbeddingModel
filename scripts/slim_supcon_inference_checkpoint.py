from pathlib import Path

import torch


INPUT = Path("checkpoints/supcon_models/0707_convnext_tiny/current_model.pth")
OUTPUT = INPUT.with_name("current_model_inference.pth")
QUERY_PREFIX = "query_encoder."
MOMENTUM_PREFIX = "momentum_encoder."


def get_state_dict(ckpt):
    for key in ("model_state_dict", "state_dict", "model"):
        state = ckpt.get(key) if isinstance(ckpt, dict) else None
        if isinstance(state, dict):
            return state
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt):
        return ckpt
    raise ValueError("未找到模型权重")


def slim_for_inference(state):
    if not any(k.startswith(QUERY_PREFIX) for k in state):
        return dict(state)

    slim = {}
    for key, value in state.items():
        if key.startswith(MOMENTUM_PREFIX):
            continue
        if key.startswith(QUERY_PREFIX):
            key = key[len(QUERY_PREFIX):]
        slim[key] = value
    return slim


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def main():
    ckpt = torch.load(INPUT, map_location="cpu", weights_only=False)
    state = slim_for_inference(get_state_dict(ckpt))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state}, OUTPUT)

    print(f"input:  {INPUT} ({size_mb(INPUT):.2f} MB)")
    print(f"output: {OUTPUT} ({size_mb(OUTPUT):.2f} MB)")
    print(f"keys:   {len(state)}")


if __name__ == "__main__":
    main()
