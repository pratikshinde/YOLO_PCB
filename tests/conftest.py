from pathlib import Path

import pytest

from pcb_inspection.config import load_config, load_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_config():
    return load_config(ROOT / "configs" / "project.yaml")


@pytest.fixture(scope="session")
def active_profile(project_config):
    profiles = project_config["profiles"]
    return load_profile(ROOT / profiles["directory"], profiles["active"])


@pytest.fixture(autouse=True, scope="session")
def _camera_lock_dir(tmp_path_factory):
    """Camera locks taken by the tests never collide with a camera program running on this PC."""
    import os
    os.environ["PCB_CAMERA_LOCK_DIR"] = str(tmp_path_factory.mktemp("camera_locks"))
    yield
