from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    dataset_id: str = Field(alias="datasetId")
    method: Literal["tsne", "umap", "pca"] = "tsne"
    experiment_id: str | None = Field(default=None, alias="experimentId")
    model_id: str | None = Field(default=None, alias="modelId")
    #: optional golden/reference crop ids (subset of the dataset) to anchor anomaly scoring
    golden_crop_ids: list[str] = Field(default_factory=list, alias="goldenCropIds")
    #: defect-class ids to include; empty means all classes with crops
    class_ids: list[str] = Field(default_factory=list, alias="classIds")

    model_config = {"populate_by_name": True}


class PlotPoint(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    x: float
    y: float
    cluster: int
    url: str
    anomaly_score: float = Field(serialization_alias="anomalyScore")
    label: str
    #: 该点是否为 golden 参考样本
    is_golden: bool = Field(default=False, serialization_alias="isGolden")
    #: 溯源与裁剪图标注（与数据集 crop 一致，可选）
    source_image_id: str | None = Field(default=None, serialization_alias="sourceImageId")
    instance_index: int | None = Field(default=None, serialization_alias="instanceIndex")
    crop_annotation: list[Any] | None = Field(default=None, serialization_alias="cropAnnotation")


class AnalyzeResponse(BaseModel):
    points: list[PlotPoint]
    labels: list[str]


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str = "Vision"


class UploadCategoryOut(BaseModel):
    model_config = {"populate_by_name": True, "serialize_by_alias": True}

    id: str
    name: str
    sort_order: int = Field(serialization_alias="sortOrder")


class UploadCategoryCreate(BaseModel):
    name: str


class UploadCategoryPatch(BaseModel):
    name: str


class InferenceRunCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    model_id: str | None = Field(default=None, alias="modelId")
    dataset_mode: Literal["existing", "upload"] = Field(default="existing", alias="datasetMode")
    dataset_id: str | None = Field(default=None, alias="datasetId")
    golden_crop_ids: list[str] = Field(default_factory=list, alias="goldenCropIds")
    class_ids: list[str] = Field(default_factory=list, alias="classIds")
    algorithm: str = "tsne"
    view_mode: Literal["distribution", "anomaly"] = Field(default="distribution", alias="viewMode")


class InferenceRunPatch(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    model_id: str | None = Field(default=None, alias="modelId")
    dataset_mode: Literal["existing", "upload"] | None = Field(default=None, alias="datasetMode")
    dataset_id: str | None = Field(default=None, alias="datasetId")
    golden_crop_ids: list[str] | None = Field(default=None, alias="goldenCropIds")
    class_ids: list[str] | None = Field(default=None, alias="classIds")
    algorithm: str | None = None
    view_mode: Literal["distribution", "anomaly"] | None = Field(default=None, alias="viewMode")


class InferenceRunSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    status: str
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    dataset_mode: str = Field(serialization_alias="datasetMode")
    dataset_id: str | None = Field(default=None, serialization_alias="datasetId")
    dataset_name: str | None = Field(default=None, serialization_alias="datasetName")
    golden_crop_ids: list[str] = Field(default_factory=list, serialization_alias="goldenCropIds")
    class_ids: list[str] = Field(default_factory=list, serialization_alias="classIds")
    algorithm: str
    view_mode: str = Field(serialization_alias="viewMode")
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")


class InferenceRunDetail(InferenceRunSummary):
    result_json: dict[str, Any] | None = Field(default=None, serialization_alias="resultJson")
    # Algorithms for which a cached projection already exists
    cached_algorithms: list[str] = Field(default_factory=list, serialization_alias="cachedAlgorithms")


class InferenceRunAnalyzeUpload(BaseModel):
    """Upload-mode: client sends mock analysis output to persist."""

    model_config = ConfigDict(populate_by_name=True)

    labels: list[str]
    points: list[dict[str, Any]]


class ProjectionOut(BaseModel):
    """Response for a single cached projection."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    algorithm: str
    labels: list[str]
    points: list[dict[str, Any]]
