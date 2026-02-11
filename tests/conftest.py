# tests/conftest.py
import joblib
import numpy as np
import pytest
from app.main import app as fastapi_app


@pytest.fixture(autouse=True)
def mock_joblib_load(monkeypatch):
    """
    Replace joblib.load with a lightweight fake that returns a dict
    matching your artifact shape: {'pipeline': <object>, 'threshold': float}
    This avoids unpickling the large sklearn pipeline and removes
    version / category issues during tests.
    """

    class MockPipeline:
        def predict_proba(self, X):
            # return deterministic probability for tests
            # shape: (n_samples, 2)
            return np.array([[0.25, 0.75] for _ in range(len(X))])

    def fake_load(path):
        return {"pipeline": MockPipeline(), "threshold": 0.45}

    # Apply monkeypatch to joblib.load globally for tests
    monkeypatch.setattr(joblib, "load", fake_load)
    yield
    # monkeypatch auto-reverts after each test


@pytest.fixture
def client():
    """TestClient that triggers FastAPI startup (which will call the monkeypatched joblib.load)."""
    from fastapi.testclient import TestClient

    with TestClient(fastapi_app) as client:
        yield client
