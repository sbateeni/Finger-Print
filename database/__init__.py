from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import BASE_DIR

_db_file = (Path(BASE_DIR) / "fingerprint.db").resolve()
# Forward slashes — required for SQLAlchemy sqlite URLs on Windows
DATABASE_URL = f"sqlite:///{_db_file.as_posix()}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_tables():
    """Create all tables if they don't exist."""
    import database.models  # noqa: E402, F811
    Base.metadata.create_all(bind=engine)
