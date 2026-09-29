"""Parity with mlx-lm tests/test_finetune.py: LoRA param math + LR schedule.

Uses a tiny Llama model (MiniCPM5 GQA shape scaled down) so tests stay
offline and CPU-only.
"""

from mlx.utils import tree_flatten
from mlx_lm.tuner import utils as tuner_utils
from mlx_lm.tuner.utils import build_schedule
from omegaconf import OmegaConf

from src.train import build_lora_config


def _tiny_llama_args():
    from mlx_lm.models import llama

    return llama.ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=1024,
        tie_word_embeddings=False,
    )


class TestLoraParity:
    def test_trainable_params_scale_with_rank(self):
        from mlx_lm.models import llama

        args = _tiny_llama_args()
        counts = {}
        for rank in (1, 8):
            model = llama.Model(args)
            model.freeze()
            tuner_utils.linear_to_lora_layers(
                model, 2, {"rank": rank, "dropout": 0.0, "scale": 16.0}
            )
            counts[rank] = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
            assert counts[rank] > 0
        assert counts[8] == 8 * counts[1]

    def test_dora_has_more_trainable_params_than_lora(self):
        from mlx_lm.models import llama

        lora_counts = {}
        for use_dora in (False, True):
            model = llama.Model(_tiny_llama_args())
            model.freeze()
            tuner_utils.linear_to_lora_layers(
                model,
                2,
                {"rank": 8, "dropout": 0.0, "scale": 16.0},
                use_dora=use_dora,
            )
            lora_counts[use_dora] = sum(
                v.size for _, v in tree_flatten(model.trainable_parameters())
            )
        assert lora_counts[True] > lora_counts[False]

    def test_attention_keys_reduce_trainable_params(self):
        from mlx_lm.models import llama

        attn_keys = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        counts = {}
        for keys in (None, attn_keys):
            model = llama.Model(_tiny_llama_args())
            model.freeze()
            cfg: dict = {"rank": 8, "dropout": 0.0, "scale": 16.0}
            if keys is not None:
                cfg["keys"] = keys
            tuner_utils.linear_to_lora_layers(model, 2, cfg)
            counts[bool(keys)] = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        assert counts[True] > 0
        assert counts[True] < counts[False]

    def test_build_lora_config_schedule_matches_upstream(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "model": {"name": "dummy"},
                "lora": {"fine_tune_type": "lora", "r": 8, "dropout": 0.0, "scale": 16.0, "num_layers": 2},
                "training": {
                    "optimizer": "adamw",
                    "batch_size": 2,
                    "val_batches": 1,
                    "learning_rate": 1e-5,
                    "lr_schedule": "cosine_decay",
                    "warmup_ratio": 0.1,
                    "steps_per_report": 1,
                    "steps_per_eval": 2,
                    "save_every": 2,
                    "grad_checkpoint": False,
                    "grad_accumulation_steps": 1,
                    "seed": 42,
                },
                "tokenizer": {"max_seq_length": 512, "mask_prompt": True},
            }
        )
        out = build_lora_config(cfg, tmp_path, tmp_path / "adapters", iters=100)
        sched = build_schedule(out["lr_schedule"])
        assert sched is not None
        assert sched(0) == 0.0
