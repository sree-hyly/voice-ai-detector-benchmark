"""Run fairseq-trained wav2vec2 checkpoints on Hugging Face's wav2vec2.

SONAR and W2V2-AASIST were trained with Meta's `fairseq` library, which no longer installs on
current Python. The network is the same as Hugging Face's Wav2Vec2Model; only parameter names
differ. This module renames them and offers a drop-in for those projects' `SSLModel`.
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

XLSR_300M_CONFIG = Path(__file__).resolve().parents[1] / "models" / "prosdd_asv19" / "xlsr300m_config.json"

_DROP = re.compile(r"^(quantizer\.|project_q\.|final_proj\.|label_embs)")
_RULES = [
    (r"^mask_emb$", "masked_spec_embed"),
    (r"^feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)$", r"feature_extractor.conv_layers.\1.conv.\2"),
    (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)$", r"feature_extractor.conv_layers.\1.layer_norm.\2"),
    (r"^post_extract_proj\.(weight|bias)$", r"feature_projection.projection.\1"),
    (r"^layer_norm\.(weight|bias)$", r"feature_projection.layer_norm.\1"),
    (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
    (r"^encoder\.pos_conv\.0\.weight_g$", "encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
    (r"^encoder\.pos_conv\.0\.weight_v$", "encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
    (r"^encoder\.layer_norm\.(weight|bias)$", r"encoder.layer_norm.\1"),
    (r"^encoder\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|q_proj|out_proj)\.(weight|bias)$", r"encoder.layers.\1.attention.\2.\3"),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$", r"encoder.layers.\1.layer_norm.\2"),
    (r"^encoder\.layers\.(\d+)\.fc1\.(weight|bias)$", r"encoder.layers.\1.feed_forward.intermediate_dense.\2"),
    (r"^encoder\.layers\.(\d+)\.fc2\.(weight|bias)$", r"encoder.layers.\1.feed_forward.output_dense.\2"),
    (r"^encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$", r"encoder.layers.\1.final_layer_norm.\2"),
]


def fairseq_to_hf_key(k: str) -> str | None:
    """Hugging Face name for a fairseq wav2vec2 parameter; None for pretraining-only parts."""
    if _DROP.match(k):
        return None
    for pat, rep in _RULES:
        if re.match(pat, k):
            return re.sub(pat, rep, k)
    raise KeyError(f"no mapping for fairseq key {k!r}")


def remap_state_dict(sd: dict, prefix: str) -> dict:
    """Rename every key under `prefix` from fairseq to HF naming; leave other keys alone."""
    out = {}
    for k, v in sd.items():
        if prefix and not k.startswith(prefix):
            out[k] = v
            continue
        new = fairseq_to_hf_key(k[len(prefix):])
        if new is not None:
            out[prefix + new] = v
    return out


class HFSSL(nn.Module):
    """Drop-in for the `SSLModel` class in SONAR / SSL_Anti-spoofing, backed by HF wav2vec2."""

    def __init__(self, config: Wav2Vec2Config | None = None, *_, **__):
        super().__init__()
        if config is None or not isinstance(config, Wav2Vec2Config):
            config = Wav2Vec2Config.from_json_file(str(XLSR_300M_CONFIG))
        config.apply_spec_augment = False
        config.layerdrop = 0.0
        self.model = Wav2Vec2Model(config)
        self.out_dim = config.hidden_size

    def load_fairseq_state(self, sd: dict) -> None:
        self.model.load_state_dict(remap_state_dict(sd, prefix=""), strict=True)

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, 0, :]
        return self.model(x).last_hidden_state


def install_fairseq_stub() -> None:
    """Let `import fairseq` succeed so upstream model files can be imported; we never call into it."""
    if "fairseq" in sys.modules:
        return
    m = types.ModuleType("fairseq")
    m.checkpoint_utils = types.SimpleNamespace()
    sys.modules["fairseq"] = m
