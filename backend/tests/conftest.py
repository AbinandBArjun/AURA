import pytest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.db.base import Base
from app.db.session import get_db
from app.main import app

# Import all models so SQLAlchemy registers their tables.
from app.models import (
    User,
    Workspace,
    Document,
    DocumentPage,
    DocumentChunk,
    ResearchRun,
)


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)

    yield engine

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(test_engine):
    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    db = TestingSessionLocal()

    yield db

    db.close()


@pytest.fixture
def client(test_engine):
    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_workspace(db_session):
    user = User(
        email="integration-test@example.com",
        display_name="Integration Test User",
    )

    db_session.add(user)
    db_session.flush()

    workspace = Workspace(
        owner_id=user.id,
        name="Integration Test Workspace",
    )

    db_session.add(workspace)
    db_session.commit()
    db_session.refresh(workspace)

    return workspace

