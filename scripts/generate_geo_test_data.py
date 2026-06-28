#!/usr/bin/env python3
"""Generate simple geometric shape test images under geo_data/."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw
import colorsys


IMAGE_SIZE = 960
BG_COLOR = (245, 245, 245)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def similar_color(
    rng: random.Random,
    base_h: float,
    base_s: float,
    base_v: float,
) -> tuple[int, int, int]:
    h = (base_h + rng.uniform(-18, 18)) % 360
    s = max(0.35, min(0.95, base_s + rng.uniform(-0.12, 0.12)))
    v = max(0.45, min(0.95, base_v + rng.uniform(-0.15, 0.15)))
    return hsv_to_rgb(h, s, v)


def rotate_points(
    points: list[tuple[float, float]],
    cx: float,
    cy: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rotated = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        rotated.append((rx, ry))
    return rotated


def random_center(rng: random.Random, margin: float) -> tuple[float, float]:
    lo, hi = margin, IMAGE_SIZE - margin
    return rng.uniform(lo, hi), rng.uniform(lo, hi)


def draw_triangle(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    color: tuple[int, int, int],
) -> None:
    cx, cy = random_center(rng, 50)
    radius = rng.uniform(35, 95)
    angles = sorted(rng.sample(range(360), 3))
    points = [
        (
            cx + radius * math.cos(math.radians(a)),
            cy + radius * math.sin(math.radians(a)),
        )
        for a in angles
    ]
    rotation = rng.uniform(0, 360)
    points = rotate_points(points, cx, cy, rotation)
    draw.polygon(points, fill=color)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    color: tuple[int, int, int],
) -> None:
    cx, cy = random_center(rng, 45)
    radius = rng.uniform(30, 90)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=color)


def draw_rectangle(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    color: tuple[int, int, int],
) -> None:
    cx, cy = random_center(rng, 55)
    w = rng.uniform(50, 140)
    h = rng.uniform(35, 120)
    half_w, half_h = w / 2, h / 2
    points = [
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    ]
    rotation = rng.uniform(0, 360)
    points = rotate_points(points, cx, cy, rotation)
    draw.polygon(points, fill=color)


SHAPE_CONFIG = {
    "triangle": {
        "draw": draw_triangle,
        "base_hsv": (8.0, 0.78, 0.82),   # warm red-orange family
    },
    "circle": {
        "draw": draw_circle,
        "base_hsv": (215.0, 0.72, 0.80),  # blue family
    },
    "rectangle": {
        "draw": draw_rectangle,
        "base_hsv": (125.0, 0.68, 0.78),  # green family
    },
}


def generate_folder(
    out_dir: Path,
    shape_name: str,
    count: int,
    seed: int,
) -> None:
    cfg = SHAPE_CONFIG[shape_name]
    draw_fn = cfg["draw"]
    base_h, base_s, base_v = cfg["base_hsv"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for i in range(count):
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BG_COLOR)
        draw = ImageDraw.Draw(img)
        color = similar_color(rng, base_h, base_s, base_v)
        draw_fn(draw, rng, color)
        img.save(out_dir / f"{shape_name}_{i + 1:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate geo_data test images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test_cluster_data/geo_data"),
    )
    parser.add_argument("--count", type=int, default=25, help="Images per shape folder.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for idx, shape_name in enumerate(SHAPE_CONFIG):
        generate_folder(
            args.output / shape_name,
            shape_name,
            args.count,
            seed=args.seed + idx * 1000,
        )
        print(f"Generated {args.count} images in {args.output / shape_name}")


if __name__ == "__main__":
    main()
