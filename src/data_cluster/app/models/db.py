import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data_cluster.app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Dataset(Base):
    __tablename__ = "Dataset"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, default="Image")
    size: Mapped[str | None] = mapped_column(String, nullable=True)
    items: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="Ready")
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    defect_classes: Mapped[list["DefectClass"]] = relationship(
        "DefectClass", back_populates="dataset", cascade="all, delete-orphan"
    )
    images: Mapped[list["Image"]] = relationship(
        "Image", back_populates="dataset", cascade="all, delete-orphan"
    )
    crop_images: Mapped[list["CropImage"]] = relationship("CropImage", back_populates="dataset")


class DefectClass(Base):
    """Dataset-level registry of defect classes.

    A defect class is assigned per-annotation-region (not per-image), so a single
    image can contain regions of several different classes.
    """

    __tablename__ = "DefectClass"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    #: UI color hint (hex string e.g. "#ef4444"); optional.
    color: Mapped[str | None] = mapped_column(String, nullable=True)
    sort_order: Mapped[int] = mapped_column("sortOrder", Integer, default=0)
    dataset_id: Mapped[str] = mapped_column(
        "datasetId", String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="defect_classes")
    regions: Mapped[list["AnnotationRegion"]] = relationship(
        "AnnotationRegion", back_populates="defect_class"
    )
    crop_images: Mapped[list["CropImage"]] = relationship("CropImage", back_populates="defect_class")


class Image(Base):
    __tablename__ = "Image"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    url: Mapped[str] = mapped_column(String, nullable=False)
    #: absolute filesystem path of the original image (optional convenience cache)
    file_path: Mapped[str | None] = mapped_column("filePath", Text, nullable=True)
    dataset_id: Mapped[str] = mapped_column(
        "datasetId", String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True
    )
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: how the image entered the platform: "zip" | "single" | "batch"
    source: Mapped[str] = mapped_column(String, default="single")
    #: "unannotated" | "annotated"
    annotation_status: Mapped[str] = mapped_column("annotationStatus", String, default="unannotated")
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="images")
    annotation: Mapped["Annotation | None"] = relationship(
        "Annotation", back_populates="image", uselist=False, cascade="all, delete-orphan"
    )
    regions: Mapped[list["AnnotationRegion"]] = relationship(
        "AnnotationRegion", back_populates="image", cascade="all, delete-orphan"
    )
    crops: Mapped[list["CropImage"]] = relationship(
        "CropImage", back_populates="source_image", cascade="all, delete-orphan"
    )


class Annotation(Base):
    """Per-image annotation metadata (1:1).

    The geometry now lives in :class:`AnnotationRegion` rows; this table keeps the
    original sidecar metadata (e.g. the raw zip ``.json`` payload) for traceability.
    """

    __tablename__ = "Annotation"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(
        "imageId", String, ForeignKey("Image.id", ondelete="CASCADE"), unique=True
    )
    image: Mapped["Image"] = relationship("Image", back_populates="annotation")
    shape_type: Mapped[str | None] = mapped_column("shapeType", String, nullable=True)
    network_type: Mapped[str | None] = mapped_column("networkType", String, nullable=True)
    #: 原始标注 JSON 全文（含 image_name、image_uuid、image_path 等所有字段）
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class AnnotationRegion(Base):
    """A single annotated region (polygon) on an image, with an assigned defect class.

    ``points`` is a list of [x, y] normalized to 0-1 in the original image space.
    ``is_subtract`` regions punch holes in overlapping non-subtract regions when
    crops/masks are generated.
    """

    __tablename__ = "AnnotationRegion"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(
        "imageId", String, ForeignKey("Image.id", ondelete="CASCADE"), index=True
    )
    class_id: Mapped[str | None] = mapped_column(
        "classId", String, ForeignKey("DefectClass.id", ondelete="SET NULL"), nullable=True, index=True
    )
    points: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    is_subtract: Mapped[bool] = mapped_column("isSubtract", Boolean, default=False)
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)

    image: Mapped["Image"] = relationship("Image", back_populates="regions")
    defect_class: Mapped["DefectClass | None"] = relationship("DefectClass", back_populates="regions")


class CropImage(Base):
    """Crop 专用数据表，一张从原图按标注区域生成的裁剪样本。"""

    __tablename__ = "CropImage"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    #: 裁剪图的 /static/... 访问 URL
    url: Mapped[str] = mapped_column(String, nullable=False)
    #: 裁剪图在磁盘上的绝对路径，训练时直接按索引读取，避免复制数据
    file_path: Mapped[str | None] = mapped_column("filePath", Text, nullable=True)
    #: 直接归属的数据集，避免查询 crop 时再绕 Image -> Dataset
    dataset_id: Mapped[str | None] = mapped_column(
        "datasetId", String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True, nullable=True
    )
    dataset: Mapped["Dataset | None"] = relationship("Dataset", back_populates="crop_images")
    #: 直接归属的缺陷类别（来源标注区域的类别），训练/统计直接按 crop 表分组
    class_id: Mapped[str | None] = mapped_column(
        "classId", String, ForeignKey("DefectClass.id", ondelete="CASCADE"), index=True, nullable=True
    )
    defect_class: Mapped["DefectClass | None"] = relationship("DefectClass", back_populates="crop_images")
    #: 生成该 crop 的标注区域
    region_id: Mapped[str | None] = mapped_column(
        "regionId", String, ForeignKey("AnnotationRegion.id", ondelete="SET NULL"), nullable=True, index=True
    )
    #: 对应的 mask 图 URL（同目录下 *_mask.png）
    mask_url: Mapped[str | None] = mapped_column("maskUrl", String, nullable=True)
    #: mask 图在磁盘上的绝对路径
    mask_path: Mapped[str | None] = mapped_column("maskPath", Text, nullable=True)
    #: 来源原图
    source_image_id: Mapped[str] = mapped_column(
        "sourceImageId", String, ForeignKey("Image.id", ondelete="CASCADE")
    )
    source_image: Mapped["Image"] = relationship("Image", back_populates="crops")
    #: 在原图上的 instance 序号（0-based）
    instance_index: Mapped[int] = mapped_column("instanceIndex", Integer, default=0)
    #: 原图上标注 bbox [x_min, y_min, x_max, y_max]（像素）
    bbox: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    #: 实际 crop 区域 [x_min, y_min, x_max, y_max]（像素）
    crop_bbox: Mapped[list[int] | None] = mapped_column("cropBbox", JSON, nullable=True)
    #: 原图尺寸 [h, w]
    original_size: Mapped[list[int] | None] = mapped_column("originalSize", JSON, nullable=True)
    #: crop 图尺寸 [h, w]
    patch_size: Mapped[list[int] | None] = mapped_column("patchSize", JSON, nullable=True)
    #: crop 图坐标系内的标注 polygon（归一化 0-1，可直接叠加到 crop 图上）
    #: 格式: [{"points": [[x,y],...], "isSubtract": bool}, ...]
    crop_annotation: Mapped[list[Any] | None] = mapped_column("cropAnnotation", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)
    experiment_samples: Mapped[list["ExperimentSample"]] = relationship(
        "ExperimentSample", back_populates="crop_image"
    )


class InferenceUploadCategory(Base):
    """User-defined class names for Inference page upload mode (not tied to Dataset)."""

    __tablename__ = "InferenceUploadCategory"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    sort_order: Mapped[int] = mapped_column("sortOrder", Integer, default=0)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)


class InferenceRun(Base):
    """A saved inference session (like an experiment): config + optional cached analysis result."""

    __tablename__ = "InferenceRun"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="Draft")
    model_id: Mapped[str | None] = mapped_column("modelId", String, ForeignKey("Experiment.id"), nullable=True)
    dataset_mode: Mapped[str] = mapped_column("datasetMode", String, default="existing")
    dataset_id: Mapped[str | None] = mapped_column("datasetId", String, ForeignKey("Dataset.id"), nullable=True)
    algorithm: Mapped[str] = mapped_column(String, default="tsne")
    view_mode: Mapped[str] = mapped_column("viewMode", String, default="distribution")
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    projections: Mapped[list["InferenceRunProjection"]] = relationship(
        "InferenceRunProjection", back_populates="run", cascade="all, delete-orphan"
    )


class InferenceRunProjection(Base):
    """Per-algorithm projection cache for an InferenceRun.

    Stores the 2-D scatter coordinates + anomaly scores for each of TSNE / UMAP / PCA
    independently, so switching algorithm tabs is instant after the first compute.
    """

    __tablename__ = "InferenceRunProjection"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        "runId", String, ForeignKey("InferenceRun.id", ondelete="CASCADE"), index=True
    )
    algorithm: Mapped[str] = mapped_column(String, nullable=False)  # "tsne" | "umap" | "pca"
    labels: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    points: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)

    run: Mapped["InferenceRun"] = relationship("InferenceRun", back_populates="projections")


class Experiment(Base):
    __tablename__ = "Experiment"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    dataset: Mapped[str] = mapped_column(String, nullable=False)
    dataset_id: Mapped[str | None] = mapped_column("datasetId", String, ForeignKey("Dataset.id"), nullable=True)
    status: Mapped[str] = mapped_column(String, default="Completed")
    duration: Mapped[str | None] = mapped_column(String, nullable=True)
    accuracy: Mapped[str | None] = mapped_column(String, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)
    samples: Mapped[list["ExperimentSample"]] = relationship(
        "ExperimentSample", back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentSample(Base):
    __tablename__ = "ExperimentSample"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(
        "experimentId", String, ForeignKey("Experiment.id", ondelete="CASCADE"), index=True
    )
    crop_image_id: Mapped[str | None] = mapped_column(
        "cropImageId", String, ForeignKey("CropImage.id", ondelete="SET NULL"), nullable=True, index=True
    )
    split: Mapped[str] = mapped_column(String, default="train")
    sort_order: Mapped[int] = mapped_column("sortOrder", Integer, default=0)
    #: the defect-class name of this sample (label used for contrastive grouping)
    category_name: Mapped[str] = mapped_column("categoryName", String, nullable=False)
    source_image_id: Mapped[str | None] = mapped_column("sourceImageId", String, nullable=True)
    crop_url: Mapped[str | None] = mapped_column("cropUrl", String, nullable=True)
    crop_path: Mapped[str] = mapped_column("cropPath", Text, nullable=False)
    mask_url: Mapped[str | None] = mapped_column("maskUrl", String, nullable=True)
    mask_path: Mapped[str | None] = mapped_column("maskPath", Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="samples")
    crop_image: Mapped["CropImage | None"] = relationship("CropImage", back_populates="experiment_samples")
