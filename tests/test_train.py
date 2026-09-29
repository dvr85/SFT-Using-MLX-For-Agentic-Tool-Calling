import random

from omegaconf import OmegaConf

from src.train import (
    _ITER_RE,
    _TEST_RE,
    _VAL_RE,
    _num,
    build_lora_config,
    compute_iters,
    count_jsonl,
    get_registry_model_name,
    run_lora,
    set_seed,
    write_lora_config,
)


def _config():
    return OmegaConf.create(
        {
            "model": {"name": "dummy-model", "revision": "main", "trust_remote_code": True},
            "lora": {
                "fine_tune_type": "dora",
                "r": 8,
                "dropout": 0.0,
                "scale": 16.0,
                "num_layers": 4,
                "keys": ["self_attn.q_proj"],
            },
            "training": {
                "seed": 42,
                "warmup_ratio": 0.1,
                "optimizer": "adamw",
                "batch_size": 2,
                "val_batches": 1,
                "learning_rate": 1e-5,
                "lr_schedule": "cosine_decay",
                "steps_per_report": 1,
                "steps_per_eval": 2,
                "save_every": 2,
                "grad_checkpoint": False,
                "grad_accumulation_steps": 1,
            },
            "tokenizer": {"max_seq_length": 512, "mask_prompt": False},
        }
    )


class TestBuildLoraConfig:
    def test_uses_mlx_lm_keys(self, tmp_path):
        cfg = build_lora_config(_config(), tmp_path, tmp_path / "adapters", iters=100)

        assert cfg["train"] is True
        assert cfg["iters"] == 100
        assert cfg["data"] == str(tmp_path)
        assert cfg["adapter_path"] == str(tmp_path / "adapters")
        assert cfg["lora_parameters"] == {"rank": 8, "dropout": 0.0, "scale": 16.0, "keys": ["self_attn.q_proj"]}
        assert cfg["lr_schedule"]["name"] == "cosine_decay"
        assert cfg["lr_schedule"]["warmup"] == 10
        assert cfg["optimizer_config"] == {"adamw": {}}
        assert cfg["mask_prompt"] is False
        assert cfg["seed"] == 42
        assert "report_to" not in cfg and "project_name" not in cfg

    def test_test_eval_off_without_heldout(self, tmp_path):
        cfg = build_lora_config(_config(), tmp_path, tmp_path / "adapters", iters=100)
        assert cfg["test"] is False
        assert cfg["test_batches"] == -1  # getattr fallback; fixture has no test_batches key

    def test_test_eval_on_with_heldout(self, tmp_path):
        cfg = build_lora_config(_config(), tmp_path, tmp_path / "adapters", iters=100, has_heldout=True)
        assert cfg["test"] is True
        assert cfg["test_batches"] == -1

    def test_rejects_bad_hparams(self, tmp_path):
        bad = _config()
        bad.training.learning_rate = "not-a-number"
        try:
            build_lora_config(bad, tmp_path, tmp_path, iters=10)
        except RuntimeError as e:
            assert "Invalid training/LoRA config" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


class TestComputeIters:
    def test_epochs_times_rows_over_batch(self):
        assert compute_iters(2, 100, 4) == 50

    def test_minimum_one(self):
        assert compute_iters(1, 1, 64) == 1


class TestWriteLoraConfig:
    def test_yaml_roundtrip(self, tmp_path):
        path = write_lora_config(build_lora_config(_config(), tmp_path, tmp_path, iters=10), tmp_path / "lora.yaml")
        assert path.exists()
        assert "iters: 10" in path.read_text(encoding="utf-8")


class TestCountJsonl:
    def test_counts_lines(self, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        assert count_jsonl(path) == 2

    def test_missing_file_raises(self, tmp_path):
        try:
            count_jsonl(tmp_path / "nope.jsonl")
        except RuntimeError as e:
            assert "Failed to read" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


class TestSetSeed:
    def test_deterministic_python_random(self):
        set_seed(123)
        first = [random.random() for _ in range(3)]
        set_seed(123)
        assert [random.random() for _ in range(3)] == first


class TestParseHelpers:
    def test_num_strips_sentence_period(self):
        assert _num("1.288.") == 1.288
        assert _num("0.253") == 0.253
        assert _num("1.0e-05") == 1.0e-05

    def test_iter_line(self):
        m = _ITER_RE.search("Iter 10: Train loss 0.838, Learning Rate 2.010e-07, It/sec 5.1")
        assert m is not None and int(m.group("step")) == 10
        assert _num(m.group("train")) == 0.838

    def test_val_on_own_line(self):
        # trainer.py prints Val loss separately from the Train loss line
        assert _ITER_RE.search("Iter 200: Val loss 0.512, Val took 1.234s") is None
        v = _VAL_RE.search("Iter 200: Val loss 0.512, Val took 1.234s")
        assert v is not None and int(v.group("step")) == 200
        assert _num(v.group("val")) == 0.512

    def test_test_line_with_trailing_period(self):
        t = _TEST_RE.search("Test loss 0.253, Test ppl 1.288.")
        assert t is not None
        assert _num(t.group("test")) == 0.253
        assert _num(t.group("ppl")) == 1.288


class TestRunLora:
    def _popen(self, monkeypatch, lines):
        import subprocess
        from unittest.mock import MagicMock

        proc = MagicMock()
        proc.stdout = iter(lines)
        proc.wait.return_value = 0
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
        return proc

    def test_logs_train_val_and_test(self, monkeypatch, tmp_path):
        from unittest.mock import patch

        self._popen(
            monkeypatch,
            [
                "Iter 10: Train loss 0.838, Learning Rate 2.010e-07, It/sec 5.1\n",
                "Iter 200: Val loss 0.512, Val took 1.234s\n",
                "Test loss 0.253, Test ppl 1.288.\n",
            ],
        )
        logged = []
        with patch("src.train.mlflow") as mock_mlflow:
            mock_mlflow.log_metric.side_effect = lambda k, v, step=None: logged.append((k, v, step))
            run_lora(tmp_path / "lora.yaml")
        assert ("train_loss", 0.838, 10) in logged
        assert ("eval_loss", 0.512, 200) in logged
        assert ("test_loss", 0.253, None) in logged
        assert ("test_ppl", 1.288, None) in logged

    def test_skips_metrics_without_active_run(self, monkeypatch, tmp_path):
        from unittest.mock import patch

        self._popen(monkeypatch, ["Iter 10: Train loss 0.838, Learning Rate 2.010e-07\n"])
        with patch("src.train.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            run_lora(tmp_path / "lora.yaml")
            mock_mlflow.log_metric.assert_not_called()


class TestRegistryModelName:
    def test_returns_mlflow_model_name(self):
        assert get_registry_model_name(OmegaConf.create({"mlflow": {"model_name": "m"}})) == "m"

    def test_none_when_missing_or_empty(self):
        assert get_registry_model_name(OmegaConf.create({})) is None
        assert get_registry_model_name(OmegaConf.create({"mlflow": {"model_name": ""}})) is None
