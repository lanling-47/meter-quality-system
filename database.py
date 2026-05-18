"""数据库初始化与会话管理"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI 依赖注入：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库，创建所有表"""
    from app.models import db_models  # noqa: F401 触发模型注册
    Base.metadata.create_all(bind=engine)
    _ensure_schema_evolution()


def _ensure_schema_evolution():
    """Apply additive schema updates for existing local SQLite databases."""
    inspector = inspect(engine)
    if "quality_feature_snapshot" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("quality_feature_snapshot")}
    alter_sql = []
    if "order_no" not in columns:
        alter_sql.append("ALTER TABLE quality_feature_snapshot ADD COLUMN order_no VARCHAR(50)")
    if "batch_key" not in columns:
        alter_sql.append("ALTER TABLE quality_feature_snapshot ADD COLUMN batch_key VARCHAR(100)")

    if not alter_sql:
        return
    with engine.begin() as conn:
        for sql in alter_sql:
            conn.execute(text(sql))
