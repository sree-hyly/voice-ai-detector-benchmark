"""One runner per detector. Each takes 16 kHz mono audio and returns an AI score.

AI score = log-odds the clip is AI (see vad_bench.prep): > 0 means the model calls it AI.
Each runner follows the input handling of the model's own evaluation code (cited per class).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vad_bench.prep import ai_log_odds, first_n_tiled, first_n_zero_padded, standardize
from vad_bench.w2v2_bridge import HFSSL, install_fairseq_stub, remap_state_dict

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "models"


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def import_file(name: str, path: Path):
    """Import a model's own source file under a unique module name (several are called _net.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


class Runner:
    name = "?"
    device = "cpu"

    def load(self) -> None: ...

    def ai_score(self, wav16k: np.ndarray) -> float: ...

    def unload(self) -> None:
        """Drop loaded weights/modules; keep plain settings like name and folder."""
        for k, v in list(vars(self).items()):
            if not isinstance(v, (str, int, float, bool, type(None))):
                setattr(self, k, None)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# --------------------------------------------------------------------------- AASIST
class AASIST(Runner):
    """clovaai/aasist, official AASIST.pth. Eval: first 64600 samples, short clips tiled; class 1 = bona fide."""
    name = "AASIST"

    def load(self):
        self.device = pick_device()
        mod = import_file("aasist_model", M / "upstream/aasist/models/AASIST.py")
        cfg = json.loads((M / "upstream/aasist/config/AASIST.conf").read_text())["model_config"]
        self.net = mod.Model(cfg)
        self.net.load_state_dict(torch.load(M / "aasist/AASIST.pth", map_location="cpu", weights_only=True))
        self.net.to(self.device).eval()

    @torch.no_grad()
    def ai_score(self, wav):
        x = torch.from_numpy(first_n_tiled(wav, 64600))[None].to(self.device)
        _, out = self.net(x)
        return float(ai_log_odds(out.cpu(), fake_index=0)[0])


# --------------------------------------------------------------------------- W2V2-AASIST
class W2V2AASIST(Runner):
    """TakHemlata/SSL_Anti-spoofing LA_model.pth (XLS-R 300M + AASIST). Eval pad: first 64600, tiled."""
    name = "W2V2-AASIST"

    def load(self):
        self.device = pick_device()
        install_fairseq_stub()
        mod = import_file("ssl_as_model", M / "upstream/SSL_Anti-spoofing/model.py")
        mod.SSLModel = lambda device, *a, **k: HFSSL()
        net = mod.Model(None, self.device)
        sd = torch.load(M / "w2v2_aasist/LA_model.pth", map_location="cpu", weights_only=True)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        net.load_state_dict(remap_state_dict(sd, prefix="ssl_model.model."), strict=True)
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def ai_score(self, wav):
        x = torch.from_numpy(first_n_tiled(wav, 64600))[None].to(self.device)
        return float(ai_log_odds(self.net(x).cpu(), fake_index=0)[0])


# --------------------------------------------------------------------------- SONAR
class SONAR(Runner):
    """SONAR-Full (dual XLS-R + AASIST). Eval: first 64600 samples, zero-padded; class 1 = real."""
    name = "SONAR"

    def load(self):
        self.device = pick_device()
        install_fairseq_stub()
        mod = import_file("sonar_net", M / "sonar/_net.py")
        mod.SSLModel = lambda device, *a, **k: HFSSL()
        net = mod.GuidedModel(None, self.device)
        sd = torch.load(M / "sonar/sonar_full_xlsr_aasist_eer6.pth", map_location="cpu", weights_only=True)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        sd = remap_state_dict(sd, prefix="ssl_model.model.")
        sd = remap_state_dict(sd, prefix="noise_ssl_model.model.")
        net.load_state_dict(sd, strict=True)
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def ai_score(self, wav):
        x = torch.from_numpy(first_n_zero_padded(wav, 64600))[None, None].to(self.device)
        return float(ai_log_odds(self.net(x)[0].reshape(-1, 2).cpu(), fake_index=0)[0])


# --------------------------------------------------------------------------- ProSDD
class ProSDD(Runner):
    """ProSDD (XLS-R 300M, prosody pre-training). Eval: 4 s centre crop; class 1 = bona fide."""

    def __init__(self, variant: str):
        self.variant = variant  # "asv19" or "asv5"
        self.name = {"asv19": "ProSDD (ASVspoof2019)", "asv5": "ProSDD (ASVspoof5)"}[variant]

    def load(self):
        self.device = pick_device()
        d = M / f"prosdd_{self.variant}"
        self.net_mod = import_file(f"prosdd_net_{self.variant}", d / "_net.py")
        ckpt = d / ("ProSDD_Trained2019" if self.variant == "asv19" else "ProSDD_Trained2024")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        unused = {"mask_embed", "pros_ln.weight", "pros_ln.bias", "final_proj.weight", "final_proj.bias"}
        sd = {k: v for k, v in sd.items() if k not in unused}
        net = self.net_mod.ProSDDNet()
        net.load_state_dict(sd, strict=True)
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def ai_score(self, wav):
        x = torch.from_numpy(self.net_mod.pad_np(wav))[None].to(self.device)
        return float(ai_log_odds(self.net(x).cpu(), fake_index=0)[0])

    def unload(self):
        self.net = self.net_mod = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# --------------------------------------------------------------------------- Pellav2
class Pellav2(Runner):
    """Pellav2 (w2v-BERT 2.0 + layer mix). pellav2_infer.py: loudness-normalised 4 s centre crop, sigmoid(logit)=P(fake)."""
    name = "Pellav2"

    def load(self):
        from safetensors.torch import load_file

        self.device = pick_device()
        self.mod = import_file("pellav2_net", M / "pellav2/_net.py")
        net = self.mod.Detector()
        net.load_state_dict(load_file(M / "pellav2/pellav2_detector.safetensors"), strict=True)
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def ai_score(self, wav):
        x = torch.from_numpy(self.mod.preprocess(wav))[None].to(self.device)
        return float(self.net(x).float().cpu()[0])  # already logit(fake)

    def unload(self):
        self.net = self.mod = None


# --------------------------------------------------------------------------- wav2vec2 XLS-R (Gustking)
class Wav2Vec2XLSR(Runner):
    """Gustking/wav2vec2-large-xlsr-deepfake-audio-classification. Labels: 0 real, 1 fake. Whole clip."""
    name = "wav2vec2-XLSR (Gustking)"

    def load(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = pick_device()
        p = M / "wav2vec2_xlsr"
        self.fe = AutoFeatureExtractor.from_pretrained(p)
        self.net = AutoModelForAudioClassification.from_pretrained(p).to(self.device).eval()
        assert self.net.config.id2label[1] == "fake"

    @torch.no_grad()
    def ai_score(self, wav):
        inp = self.fe(wav, sampling_rate=16000, return_tensors="pt")
        logits = self.net(**{k: v.to(self.device) for k, v in inp.items()}).logits
        return float(ai_log_odds(logits.cpu(), fake_index=1)[0])


# --------------------------------------------------------------------------- NII AntiDeepfake
class NIIAntiDeepfake(Runner):
    """nii-yamagishilab AntiDeepfake (SSL + mean pool + linear). README: whole clip, waveform
    standardised; softmax index 0 = fake, 1 = real."""

    def __init__(self, repo_dir: str, name: str):
        self.repo_dir, self.name = repo_dir, name

    def load(self):
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Config

        self.device = pick_device()
        d = M / self.repo_dir
        cfg = Wav2Vec2Config.from_json_file(str(d / "config.json"))
        self.ssl = HFSSL(cfg)
        sd = load_file(d / "model.safetensors")
        ssl_sd = {k.removeprefix("m_ssl.model."): v for k, v in sd.items() if k.startswith("m_ssl.model.")}
        self.ssl.load_fairseq_state(ssl_sd)
        self.fc = torch.nn.Linear(cfg.hidden_size, 2)
        self.fc.load_state_dict({"weight": sd["proj_fc.weight"], "bias": sd["proj_fc.bias"]})
        self.ssl.to(self.device).eval()
        self.fc.to(self.device).eval()

    @torch.no_grad()
    def ai_score(self, wav):
        x = torch.from_numpy(standardize(wav))[None].to(self.device)
        emb = self.ssl.extract_feat(x).mean(dim=1)
        return float(ai_log_odds(self.fc(emb).cpu(), fake_index=0)[0])


# --------------------------------------------------------------------------- VeriSpoof-Plus
class VeriSpoofPlus(Runner):
    """AustinXiao/VeriSpoof-Plus audio LLM. detect(): score = log p(spoof answer) - log p(bona fide answer).

    The package assumes an NVIDIA GPU (bf16 autocast). On a Mac we load the LLM in float32 and cast the
    audio encoder's bf16 output to float32 before the tokenizer; the computation is otherwise unchanged.
    """
    name = "VeriSpoof-Plus"

    def load(self):
        sys.path.insert(0, str(M / "verispoof_plus/code"))
        from verispoof.inference import VeriSpoof
        from verispoof.model import Network

        self.device = pick_device()
        path = str(M / "verispoof_plus")
        cfg = json.loads((M / "verispoof_plus/config.json").read_text())
        cfg["llm_dtype"] = "float32"
        net = Network(path, cfg, device=self.device)
        enc = net.encoder
        orig_call = enc.__class__.__call__
        net.encoder = type("FloatEncoder", (), {"__call__": lambda s, w: orig_call(enc, w).float()})()
        self.vs = VeriSpoof(net.to(self.device).eval(), device=self.device)

    def ai_score(self, wav):
        return float(self.vs.detect(wav, sr=16000)["score"])


def all_runners() -> list[Runner]:
    return [
        AASIST(),
        W2V2AASIST(),
        ProSDD("asv19"),
        ProSDD("asv5"),
        Pellav2(),
        Wav2Vec2XLSR(),
        SONAR(),
        NIIAntiDeepfake("nii_mms_1b", "NII MMS-1B"),
        NIIAntiDeepfake("nii_xlsr_2b", "NII XLS-R-2B"),
        VeriSpoofPlus(),
    ]
