# tests/conftest.py
"""
Global pytest configuration and fixtures for the Avatar Renderer Pod tests.
Ensures the `app/` package is on PYTHONPATH and provides common fixtures.
"""

import sys
import os
from pathlib import Path

import pytest

# Add project root so `import app` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def ensure_models_dir(tmp_path, monkeypatch):
    """
    Create an empty 'models/' directory for any code that expects it at runtime.
    Many components mount checkpoints here; tests do not use real models.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    return models_dir
