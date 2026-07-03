import hashlib
import json
import re
import shutil
import uuid
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image as PILImage

from data_cluster.app.config import Settings
from data_cluster.app.services.dataset_import_log import log_import

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def slug_category(name: str) -> str:
    s = name.strip().replace(" ", "_")
    s = _SAFE_NAME.sub("_", s)
    return s or "category"


# ── Fixed subdirectory names ──────────────────────────────────────────────────

def original_images_dir(settings: Settings, dataset_id: str) -> Path:
    """Filesystem directory where original uploaded images live."""
    return settings.upload_dir / dataset_id / "original_images"


def crop_images_dir(settings: Settings, dataset_id: str) -> Path:
    """Filesystem directory where all crop patches for a dataset live (flat)."""
    return settings.upload_dir / dataset_id / "crop_images"


def original_static_url(dataset_id: str, filename: str) -> str:
    return f"/static/{dataset_id}/original_images/{filename}"


def crop_static_url(dataset_id: str, filename: str) -> str:
    return f"/static/{dataset_id}/crop_images/{filename}"


def url_to_fs_path(settings: Settings, url: str | None) -> Path | None:
    """Convert a ``/static/...`` URL back to its filesystem path under upload_dir."""
    if not url or not url.startswith("/static/"):
        return None
    rel = url.removeprefix("/static/").lstrip("/")
    return settings.upload_dir / rel


def thumbnails_dir(settings: Settings) -> Path:
    """Directory holding cached JPEG thumbnails for grid previews."""
    return settings.upload_dir / "_thumbnails"


def get_or_create_thumbnail(
    settings: Settings, static_url: str, size: int = 384
) -> Path | None:
    """Return a cached JPEG thumbnail for a ``/static/...`` image, generating it lazily.

    The cache key includes the source mtime+size so edited/replaced files are
    regenerated automatically. Returns ``None`` if the source is missing or
    cannot be decoded.
    """
    src = url_to_fs_path(settings, static_url)
    if not src or not src.is_file():
        return None
    try:
        st = src.stat()
    except OSError:
        return None

    key_src = f"{static_url}|{size}|{int(st.st_mtime)}|{st.st_size}"
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    out_dir = thumbnails_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.jpg"
    if out.is_file():
        return out

    try:
        with PILImage.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), PILImage.Resampling.LANCZOS)
            tmp = out.with_suffix(".tmp.jpg")
            im.save(tmp, format="JPEG", quality=80, optimize=True)
            tmp.replace(out)
    except Exception:
        return None
    return out


def slug_model_folder(name: str) -> str:
    """Safe directory segment for a model name as shown in the UI (under checkpoints/)."""
    s = name.strip().replace(" ", "_")
    s = _SAFE_NAME.sub("_", s)
    return s or "model"


def experiment_checkpoint_run_dir(settings: Settings, model_display_name: str, experiment_id: str) -> Path:
    """``checkpoints/{model}/{experiment_id}/`` — weights and CSV artifacts live here."""
    return settings.checkpoints_dir / slug_model_folder(model_display_name) / experiment_id


def _count_common_prefix_depth(paths: list[str]) -> int:
    """Count leading directory segments shared by all paths so they can be stripped.

    Stops when stripping another level would leave any path with fewer than 2 segments
    (i.e. no category/file pair would remain).
    """
    if not paths:
        return 0
    split = [p.replace("\\", "/").strip("/").split("/") for p in paths]
    depth = 0
    while True:
        if any(len(p) <= depth + 1 for p in split):
            break
        candidate = split[0][depth]
        if any(p[depth] != candidate for p in split):
            break
        depth += 1
    return depth


def _image_ext_from_bytes(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _file_stem(filename: str) -> str:
    """Stem used to pair images with JSON sidecars (extensionless names stay as-is)."""
    p = Path(filename)
    ext = p.suffix.lower()
    if ext in _IMAGE_EXTS or ext == ".json":
        return p.stem
    return filename


def _is_zip_image(filename: str, json_stems: set[str]) -> bool:
    ext = Path(filename).suffix.lower()
    if ext in _IMAGE_EXTS:
        return True
    if ext == ".json":
        return False
    # Extensionless files are images when a matching .json sidecar exists.
    return _file_stem(filename) in json_stems


def unique_filename(original: str, *, content_ext: str | None = None) -> str:
    ext = Path(original).suffix.lower()
    if ext not in _IMAGE_EXTS:
        ext = content_ext or ".jpg"
    return f"{uuid.uuid4().hex}{ext}"


def static_url(dataset_id: str, category_slug: str, filename: str) -> str:
    return f"/static/{dataset_id}/{category_slug}/{filename}"


def fs_path(settings: Settings, dataset_id: str, category_slug: str, filename: str) -> Path:
    return settings.upload_dir / dataset_id / category_slug / filename


def ensure_upload_root(settings: Settings) -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)


def delete_dataset_files(settings: Settings, dataset_id: str) -> None:
    root = settings.upload_dir / dataset_id
    if root.is_dir():
        shutil.rmtree(root, ignore_errors=True)


def dataset_upload_dir_bytes(settings: Settings, dataset_id: str) -> int:
    """Total size in bytes of all files under the dataset upload directory."""
    root = settings.upload_dir / dataset_id
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.1f} GB"


def is_safe_zip_path(name: str) -> bool:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized or normalized.startswith("..") or "/../" in f"/{normalized}/":
        return False
    if "__MACOSX" in normalized or ".DS_Store" in normalized:
        return False
    return True


async def save_upload_file(
    settings: Settings,
    dataset_id: str,
    category_name: str,
    upload: UploadFile,
    *,
    contents: bytes | None = None,
) -> tuple[str, int]:
    """保存上传文件到磁盘。

    ``contents`` 为可选的已读字节内容；若未提供则从 upload 读取。
    磁盘写入放入默认线程池，避免阻塞事件循环。
    """
    import asyncio

    ensure_upload_root(settings)
    dest_dir = original_images_dir(settings, dataset_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = unique_filename(upload.filename or "image.jpg")
    path = dest_dir / fname

    if contents is None:
        contents = await upload.read()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, path.write_bytes, contents)
    return original_static_url(dataset_id, fname), len(contents)


def save_upload_bytes_sync(
    settings: Settings,
    dataset_id: str,
    original: str,
    contents: bytes,
) -> tuple[str, int]:
    """Synchronous variant used by background import jobs."""
    ensure_upload_root(settings)
    dest_dir = original_images_dir(settings, dataset_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = unique_filename(original or "image.jpg")
    path = dest_dir / fname
    path.write_bytes(contents)
    return original_static_url(dataset_id, fname), len(contents)


def extract_zip_to_dataset(
    settings: Settings,
    dataset_id: str,
    data: bytes,
    *,
    on_extract_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[tuple[str, str]], int, dict[str, dict]]:
    """Extract images and JSON annotations from a zip archive.

    The zip may have a single top-level wrapper folder (e.g. the zip name) that is
    automatically stripped.  Each remaining first-level folder is treated as a defect
    category name; files inside are images or JSON annotation sidecars sharing the
    same stem as the image they annotate.

    Returns:
        records:      [(category_name, public_url), ...]
        total_bytes:  total size of image data written to disk
        url_annotations: {public_url -> annotation_dict} for images that have a sidecar
    """
    ensure_upload_root(settings)
    total_bytes = 0
    records: list[tuple[str, str]] = []

    # {(category_name, original_stem) -> saved_url}
    stem_to_url: dict[tuple[str, str], str] = {}
    # {(category_name, original_stem) -> annotation_dict}
    stem_to_annotation: dict[tuple[str, str], dict] = {}

    with zipfile.ZipFile(BytesIO(data)) as zf:
        valid = [i for i in zf.infolist() if not i.is_dir() and is_safe_zip_path(i.filename)]
        strip_n = _count_common_prefix_depth([i.filename for i in valid])

        json_stems_by_category: dict[str, set[str]] = {}
        for info in valid:
            parts = info.filename.replace("\\", "/").strip("/").split("/")[strip_n:]
            if len(parts) < 2:
                continue
            filename = parts[-1]
            if Path(filename).suffix.lower() == ".json":
                category_name = parts[0]
                json_stems_by_category.setdefault(category_name, set()).add(_file_stem(filename))

        image_total = 0
        for info in valid:
            parts = info.filename.replace("\\", "/").strip("/").split("/")[strip_n:]
            if len(parts) < 2:
                continue
            category_name = parts[0]
            filename = parts[-1]
            json_stems = json_stems_by_category.get(category_name, set())
            if _is_zip_image(filename, json_stems):
                image_total += 1
        log_import(f"[{dataset_id}] 开始解压 zip（约 {image_total} 张图片）")
        image_idx = 0

        for info in valid:
            parts = info.filename.replace("\\", "/").strip("/").split("/")[strip_n:]
            if len(parts) < 2:
                continue
            category_name = parts[0]
            filename = parts[-1]
            if not filename:
                continue
            stem = _file_stem(filename)
            ext = Path(filename).suffix.lower()
            key = (category_name, stem)
            json_stems = json_stems_by_category.get(category_name, set())

            if _is_zip_image(filename, json_stems):
                image_idx += 1
                log_import(
                    f"[{dataset_id}] 解压 ({image_idx}/{image_total}) "
                    f"[{category_name}] {filename}"
                )
                raw = zf.read(info)
                total_bytes += len(raw)
                dest_dir = original_images_dir(settings, dataset_id)
                dest_dir.mkdir(parents=True, exist_ok=True)
                detected_ext = _image_ext_from_bytes(raw) if ext not in _IMAGE_EXTS else None
                fname = unique_filename(filename, content_ext=detected_ext)
                (dest_dir / fname).write_bytes(raw)
                url = original_static_url(dataset_id, fname)
                records.append((category_name, url))
                stem_to_url[key] = url
                if on_extract_progress and image_total:
                    on_extract_progress(image_idx, image_total, f"[{category_name}] {filename}")

            elif ext == ".json":
                try:
                    raw = zf.read(info)
                    stem_to_annotation[key] = json.loads(raw.decode("utf-8"))
                except Exception:
                    pass

    url_annotations: dict[str, dict] = {
        url: stem_to_annotation[key]
        for key, url in stem_to_url.items()
        if key in stem_to_annotation
    }
    log_import(
        f"[{dataset_id}] 解压完成: {len(records)} 张图片, "
        f"{len(url_annotations)} 张带标注, {human_size(total_bytes)}"
    )
    return records, total_bytes, url_annotations
