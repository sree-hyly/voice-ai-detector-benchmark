from vad_bench.runners import NIIAntiDeepfake, ProSDD, Runner, all_runners


def test_unload_frees_weights_but_keeps_identity():
    r = NIIAntiDeepfake("nii_mms_1b", "NII MMS-1B")
    r.ssl, r.fc = object(), object()
    r.unload()
    assert r.name == "NII MMS-1B" and r.repo_dir == "nii_mms_1b"
    assert r.ssl is None and r.fc is None


def test_unload_on_prosdd_keeps_variant():
    r = ProSDD("asv5")
    r.net = object()
    r.unload()
    assert r.variant == "asv5" and r.name == "ProSDD (ASVspoof5)" and r.net is None


def test_unload_before_load_is_harmless():
    Runner().unload()


def test_every_requested_model_is_registered_once():
    names = [r.name for r in all_runners()]
    assert len(names) == len(set(names)) == 10
    for want in ["AASIST", "W2V2-AASIST", "Pellav2", "SONAR", "NII MMS-1B", "NII XLS-R-2B", "VeriSpoof-Plus",
                 "wav2vec2-XLSR (Gustking)", "ProSDD (ASVspoof2019)", "ProSDD (ASVspoof5)"]:
        assert want in names
