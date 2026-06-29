from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from data_cluster.app.config import get_settings


class Base(DeclarativeBase):
    pass


def _engine_url(url: str) -> str:
    return url


def make_engine():
    settings = get_settings()
    url = _engine_url(settings.database_url)
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={
                "check_same_thread": False,
                # 每次连接开启 WAL 日志模式：写不阻塞读，大幅减少训练 on_progress
                # db.commit() 与前端轮询 SELECT 之间的锁争用
                "timeout": 30,
            },
            # 连接池：保留少量空闲连接，避免每次请求都重新打开文件
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            # WAL 模式：并发读写性能最优
            cursor.execute("PRAGMA journal_mode=WAL")
            # 同步策略 NORMAL：WAL 下足够安全，比 FULL 快很多
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 64 MB page cache（默认 2 MB）
            cursor.execute("PRAGMA cache_size=-65536")
            # 临时表放内存
            cursor.execute("PRAGMA temp_store=MEMORY")
            # mmap 512 MB：大 JSON 列（metrics/liveSeries）读取加速
            cursor.execute("PRAGMA mmap_size=536870912")
            cursor.close()

        return engine

    return create_engine(
        url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


engine = make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


#: Columns added after the initial schema; ``create_all`` only creates missing
#: *tables*, so for SQLite we additively ``ALTER TABLE`` any missing columns to
#: avoid wiping existing dev data on schema bumps.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "InferenceRun": [("goldenCropIds", "JSON"), ("selectedClassIds", "JSON")],
    "Experiment": [
        ("runStatus", "VARCHAR"),
        ("runProgress", "FLOAT DEFAULT 0"),
        ("runMetrics", "JSON"),
    ],
}


def _ensure_added_columns() -> None:
    if engine.dialect.name != "sqlite":
        return
    from sqlalchemy import text

    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row[1] for row in conn.execute(text(f'PRAGMA table_info("{table}")'))}
            if not existing:
                continue  # table not created yet — create_all handles it
            for name, decl in columns:
                if name not in existing:
                    conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {decl}'))


def init_db() -> None:
    import data_cluster.app.models.db  # noqa: F401 — register models

    Base.metadata.create_all(bind=engine)
    _ensure_added_columns()
