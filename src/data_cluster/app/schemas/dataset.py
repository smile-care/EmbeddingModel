from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DefectClassOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    color: str | None = None
    sort_order: int = Field(default=0, serialization_alias="sortOrder")


class AnnotationRegionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    class_id: str | None = Field(default=None, serialization_alias="classId")
    #: normalized 0-1 polygon points in original-image space: [[x, y], ...]
    points: list[Any] = []
    is_subtract: bool = Field(default=False, serialization_alias="isSubtract")
    order: int = 0


class CropImageOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    url: str
    class_id: str | None = Field(default=None, serialization_alias="classId")
    mask_url: str | None = Field(default=None, serialization_alias="maskUrl")
    source_image_id: str = Field(serialization_alias="sourceImageId")
    region_id: str | None = Field(default=None, serialization_alias="regionId")
    instance_index: int = Field(default=0, serialization_alias="instanceIndex")
    bbox: list[int] | None = None
    crop_bbox: list[int] | None = Field(default=None, serialization_alias="cropBbox")
    original_size: list[int] | None = Field(default=None, serialization_alias="originalSize")
    patch_size: list[int] | None = Field(default=None, serialization_alias="patchSize")
    #: polygon points remapped to crop-image space (normalized 0-1)
    #: format: [{"points": [[x,y],...], "isSubtract": bool}, ...]
    crop_annotation: list[Any] | None = Field(default=None, serialization_alias="cropAnnotation")


class ImageOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    url: str
    width: int | None = None
    height: int | None = None
    source: str = "single"
    annotation_status: str = Field(default="unannotated", serialization_alias="annotationStatus")
    regions: list[AnnotationRegionOut] = []
    crops: list[CropImageOut] = []


class DatasetSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    type: str = "Image"
    size: str | None = None
    items: int = 0
    status: str = "Ready"
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")
    defect_classes: list[DefectClassOut] = Field(default_factory=list, serialization_alias="defectClasses")


class DatasetDetail(DatasetSummary):
    images: list[ImageOut] = []


class DefectClassCreate(BaseModel):
    name: str
    color: str | None = None


class DefectClassUpdate(BaseModel):
    name: str | None = None
    color: str | None = None


class RegionIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_id: str | None = Field(default=None, alias="classId")
    points: list[Any] = []
    is_subtract: bool = Field(default=False, alias="isSubtract")


class AnnotationSaveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    regions: list[RegionIn] = []
    #: when true, (re)generate crops right after saving the annotation
    generate_crops: bool = Field(default=True, alias="generateCrops")


class DatasetCreateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    items: int


class ImageUploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    image: ImageOut
