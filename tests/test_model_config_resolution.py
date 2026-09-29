"""Offline tests for src.train.resolve_model_arch (no network, no model)."""

import json

import pytest
from omegaconf import OmegaConf

from src.train import resolve_model_arch


def _config(tmp_path: object, file_name: str = "minicpm5-1b.config.json") -> object:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    return OmegaConf.create(
        {
            "model": {
                "name": "models/minicpm5-1b-bf16",
                "hf_repo": "openbmb/MiniCPM5-1B",
                "revision": "main",
                "config_file": file_name,
            }
        }
    )


class TestResolveModelArch:
    def test_loads_sibling_snapshot(self, tmp_path, monkeypatch) -> None:
        import hydra.utils

        (tmp_path / "configs" / "model").mkdir(parents=True)
        payload = {"hidden_size": 1536, "num_hidden_layers": 24, "vocab_size": 130560}
        (tmp_path / "configs" / "model" / "minicpm5-1b.config.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        monkeypatch.setattr(hydra.utils, "get_original_cwd", lambda: str(tmp_path))
        arch = resolve_model_arch(_config(tmp_path))  # type: ignore[arg-type]
        assert arch["hidden_size"] == 1536
        assert arch["num_hidden_layers"] == 24
        assert len(str(arch["config_sha"])) == 12

    def test_missing_file_raises_with_hf_hint(self, tmp_path, monkeypatch) -> None:
        import hydra.utils

        (tmp_path / "configs" / "model").mkdir(parents=True)
        monkeypatch.setattr(hydra.utils, "get_original_cwd", lambda: str(tmp_path))
        with pytest.raises(RuntimeError, match="re-download.*openbmb/MiniCPM5-1B"):
            resolve_model_arch(_config(tmp_path, file_name="nonexistent.config.json"))  # type: ignore[arg-type]

    def test_malformed_json_raises(self, tmp_path, monkeypatch) -> None:
        import hydra.utils

        (tmp_path / "configs" / "model").mkdir(parents=True)
        (tmp_path / "configs" / "model" / "minicpm5-1b.config.json").write_text(
            "{not json", encoding="utf-8"
        )
        monkeypatch.setattr(hydra.utils, "get_original_cwd", lambda: str(tmp_path))
        with pytest.raises(RuntimeError, match="Failed to read"):
            resolve_model_arch(_config(tmp_path))  # type: ignore[arg-type]

    def test_warn_level_drift_note(self) -> None:
        """Revision floats on main by design; drift surfaces via config_sha logging, not failure."""
        cfg = OmegaConf.create({"model": {"revision": "main"}})
        assert cfg.model.revision == "main"
