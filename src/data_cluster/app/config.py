from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_CLUSTER_", extra="ignore")

    database_url: str = "sqlite:///./data/data_cluster/app.db"
    upload_dir: Path = Path("data/data_cluster/datasets")
    project_root: Path = _repo_root()
    static_mount_path: str = "/static"
    checkpoints_dir: Path = Path("data/data_cluster/checkpoints")
    #: Regenerable per-run full-dim embedding cache (npz) for relation analysis.
    inference_cache_dir: Path = Path("data/data_cluster/inference_cache")

    def model_post_init(self, __context) -> None:
        root = self.project_root.resolve()
        self.project_root = root

        if not self.upload_dir.is_absolute():
            self.upload_dir = (root / self.upload_dir).resolve()
        if not self.checkpoints_dir.is_absolute():
            self.checkpoints_dir = (root / self.checkpoints_dir).resolve()
        if not self.inference_cache_dir.is_absolute():
            self.inference_cache_dir = (root / self.inference_cache_dir).resolve()

        url = self.database_url
        sqlite_prefix = "sqlite:///"
        if url.startswith(sqlite_prefix):
            raw_path = url[len(sqlite_prefix):]
            db_path = Path(raw_path)
            if not db_path.is_absolute():
                db_path = (root / db_path).resolve()
            self.database_url = f"{sqlite_prefix}{db_path}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
