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
    #: 最近一次训练尝试的运行状态（None/Running/Completed/Stopped/Failed）
    run_status: str | None = Field(default=None, serialization_alias="runStatus")
    #: 最近一次训练尝试的实时进度（0-100）
    run_progress: float = Field(default=0.0, serialization_alias="runProgress")
    created_at: datetime = Field(serialization_alias="createdAt")


class ExperimentDetail(ExperimentSummary):
    dataset_id: str | None = Field(default=None, serialization_alias="datasetId")
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    #: 本次运行的实时指标（stage/step/liveSeries/error 等），与 metrics（结果）分离
    run_metrics: dict[str, Any] | None = Field(default=None, serialization_alias="runMetrics")
    checkpoint_path: str | None = Field(default=None, serialization_alias="checkpointPath")
    sample_count: int = Field(default=0, serialization_alias="sampleCount")
    train_sample_count: int = Field(default=0, serialization_alias="trainSampleCount")
    val_sample_count: int = Field(default=0, serialization_alias="valSampleCount")
