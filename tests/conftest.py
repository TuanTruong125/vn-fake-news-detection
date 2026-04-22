from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml


# Build a minimal temporary repo layout used by pipeline unit tests.
@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    for rel in [
        "configs",
        "data/raw/internal",
        "data/staging",
        "data/processed",
        "logs",
        "reports",
        "src/pipeline",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    return tmp_path


# Write YAML content with UTF-8 encoding and stable key order.
@pytest.fixture
def write_yaml() -> Any:
    # Persist one YAML payload to disk for test setup.
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    return _write


# Write rows to CSV using UTF-8 BOM for Windows/Excel compatibility.
@pytest.fixture
def write_csv() -> Any:
    # Persist one CSV payload to disk for test setup.
    def _write(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    return _write


# Patch module __file__ so module.main() resolves repo root to tmp_repo.
@pytest.fixture
def patch_module_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_repo: Path) -> Any:
    # Redirect module __file__ so Path(__file__).parents[...] resolves to tmp repo.
    def _patch(module: Any, script_rel_path: str) -> None:
        fake_file = tmp_repo / script_rel_path
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_text("# test stub\n", encoding="utf-8")
        monkeypatch.setattr(module, "__file__", str(fake_file))

    return _patch
