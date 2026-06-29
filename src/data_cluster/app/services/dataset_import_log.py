"""Real-time stdout progress for dataset import (zip extract, DB persist, crop gen)."""


def log_import(message: str) -> None:
    print(f"[dataset-import] {message}", flush=True)
