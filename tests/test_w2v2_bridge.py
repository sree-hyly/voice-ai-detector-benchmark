"""The bridge lets checkpoints trained with the old `fairseq` library load into Hugging Face's wav2vec2."""
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from vad_bench.w2v2_bridge import HFSSL, fairseq_to_hf_key, remap_state_dict


@pytest.mark.parametrize(
    "fs, hf",
    [
        ("feature_extractor.conv_layers.3.0.weight", "feature_extractor.conv_layers.3.conv.weight"),
        ("feature_extractor.conv_layers.0.2.1.bias", "feature_extractor.conv_layers.0.layer_norm.bias"),
        ("post_extract_proj.weight", "feature_projection.projection.weight"),
        ("layer_norm.bias", "feature_projection.layer_norm.bias"),
        ("encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.bias"),
        ("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
        ("encoder.pos_conv.0.weight_v", "encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
        ("encoder.layer_norm.weight", "encoder.layer_norm.weight"),
        ("encoder.layers.17.self_attn.q_proj.weight", "encoder.layers.17.attention.q_proj.weight"),
        ("encoder.layers.2.self_attn.out_proj.bias", "encoder.layers.2.attention.out_proj.bias"),
        ("encoder.layers.2.self_attn_layer_norm.weight", "encoder.layers.2.layer_norm.weight"),
        ("encoder.layers.5.fc1.weight", "encoder.layers.5.feed_forward.intermediate_dense.weight"),
        ("encoder.layers.5.fc2.bias", "encoder.layers.5.feed_forward.output_dense.bias"),
        ("encoder.layers.5.final_layer_norm.weight", "encoder.layers.5.final_layer_norm.weight"),
        ("mask_emb", "masked_spec_embed"),
    ],
)
def test_key_mapping(fs, hf):
    assert fairseq_to_hf_key(fs) == hf


@pytest.mark.parametrize("k", ["quantizer.vars", "quantizer.weight_proj.weight", "project_q.bias", "final_proj.weight"])
def test_pretraining_only_keys_are_dropped(k):
    assert fairseq_to_hf_key(k) is None


def test_unknown_key_is_an_error_not_silently_dropped():
    with pytest.raises(KeyError):
        fairseq_to_hf_key("encoder.something_new.weight")


def tiny_config():
    return Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), conv_bias=True,
        feat_extract_norm="layer", do_stable_layer_norm=True, num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
    )


def fairseq_names_for(cfg):
    names = ["mask_emb", "post_extract_proj.weight", "post_extract_proj.bias", "layer_norm.weight",
             "layer_norm.bias", "encoder.pos_conv.0.bias", "encoder.pos_conv.0.weight_g",
             "encoder.pos_conv.0.weight_v", "encoder.layer_norm.weight", "encoder.layer_norm.bias",
             "quantizer.vars", "project_q.weight"]
    for i in range(len(cfg.conv_dim)):
        names += [f"feature_extractor.conv_layers.{i}.0.{p}" for p in ("weight", "bias")]
        names += [f"feature_extractor.conv_layers.{i}.2.1.{p}" for p in ("weight", "bias")]
    for i in range(cfg.num_hidden_layers):
        for m in ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj",
                  "self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"):
            names += [f"encoder.layers.{i}.{m}.{p}" for p in ("weight", "bias")]
    return names


def test_remapped_fairseq_checkpoint_loads_strictly_and_reproduces_outputs():
    """Round trip: HF weights -> fake fairseq names -> remap -> fresh HF model gives identical output."""
    cfg = tiny_config()
    torch.manual_seed(0)
    src = Wav2Vec2Model(cfg).eval()
    hf_sd = src.state_dict()
    fs_sd = {}
    for name in fairseq_names_for(cfg):
        hf = fairseq_to_hf_key(name)
        fs_sd[name] = hf_sd[hf] if hf is not None else torch.zeros(1)
    assert set(remap_state_dict(fs_sd, prefix="").keys()) == set(hf_sd.keys())

    dst = HFSSL(cfg)
    dst.load_fairseq_state(fs_sd)
    dst.eval()
    x = torch.randn(2, 4000)
    expect = src(x).last_hidden_state
    got = dst.extract_feat(x)
    assert got.shape == (2, expect.shape[1], 16)
    torch.testing.assert_close(got, expect)


def test_extract_feat_accepts_channel_dimension():
    ssl = HFSSL(tiny_config()).eval()
    x = torch.randn(1, 4000)
    torch.testing.assert_close(ssl.extract_feat(x.unsqueeze(1)), ssl.extract_feat(x))


def test_remap_with_prefix_only_touches_prefixed_keys():
    sd = {"ssl_model.model.mask_emb": torch.zeros(1), "LL.weight": torch.ones(1)}
    out = remap_state_dict(sd, prefix="ssl_model.model.")
    assert set(out) == {"ssl_model.model.masked_spec_embed", "LL.weight"}
