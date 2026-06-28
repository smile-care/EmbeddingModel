from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentCreate(BaseModel):
    name: str
    model: str
    dataset: str
    dataset_id: str | None = Field(default=None, alias="datasetId")
    config: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    model: str
    dataset: str
    status: str
    duration: str | None = None
    accuracy: str | None = None
    progress: float = 0.0
    created_at: datetime = Field(serialization_alias="createdAt")


class ExperimentDetail(ExperimentSummary):
    dataset_id: str | None = Field(default=None, serialization_alias="datasetId")
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    checkpoint_path: str | None = Field(default=None, serialization_alias="checkpointPath")
    sample_count: int = Field(default=0, serialization_alias="sampleCount")
    train_sample_count: int = Field(default=0, serialization_alias="trainSampleCount")
    val_sample_count: int = Field(default=0, serialization_alias="valSampleCount")
