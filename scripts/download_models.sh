#!/bin/bash
# Downloads every model's weights + runner code into models/<name>/ (skips per-dataset score dumps).
set -u
cd "$(dirname "$0")/.."
HF=.venv/bin/hf; [ -x $HF ] || HF=.venv/bin/huggingface-cli
dl() { echo ">>> $1 -> $2"; $HF download "$1" --local-dir "models/$2" --exclude ".eval_results/*" --exclude "*.onnx" --exclude "onnx__*" --exclude "net.*" --exclude "assets/*" --exclude "*.mp4" --exclude "*.gif" --exclude "trt_*" ; }
dl SpeechAntiSpoofingBenchmarks/AASIST aasist
dl SpeechAntiSpoofingBenchmarks/W2V2-AASIST w2v2_aasist
dl SpeechAntiSpoofingBenchmarks/ProSDD-ASV19 prosdd_asv19
dl SpeechAntiSpoofingBenchmarks/ProSDD-ASV5 prosdd_asv5
dl SpeechAntiSpoofingBenchmarks/Pellav2 pellav2
dl Gustking/wav2vec2-large-xlsr-deepfake-audio-classification wav2vec2_xlsr
dl SpeechAntiSpoofingBenchmarks/SONAR-Full sonar
dl nii-yamagishilab/mms-1b-anti-deepfake nii_mms_1b
dl nii-yamagishilab/xls-r-2b-anti-deepfake nii_xlsr_2b
dl AustinXiao/VeriSpoof-Plus verispoof_plus
echo HF_DONE

# ProSDD weights are only on the authors' Google Drive (no licence granted, so nobody re-hosts them).
GD=.venv/bin/gdown
[ -f models/prosdd_asv19/ProSDD_Trained2019 ] || $GD 1RHY-VELyi4XH-L1egeRcgWjHdkdWV2bO -O models/prosdd_asv19/ProSDD_Trained2019
[ -f models/prosdd_asv5/ProSDD_Trained2024 ] || $GD 1AyD4f0RVVyDncqYbhi8QaP6-GFZ_ZGZl -O models/prosdd_asv5/ProSDD_Trained2024
# expected sha256: 2019 d2cb857c...c1c8a, 2024 c044a402...6c437 (see models/prosdd_*/meta.yaml)

# Original network code for AASIST and W2V2-AASIST (their weights repos don't include it).
mkdir -p models/upstream
[ -d models/upstream/aasist ] || git clone -q --depth 1 https://github.com/clovaai/aasist.git models/upstream/aasist
[ -d models/upstream/SSL_Anti-spoofing ] || git clone -q --depth 1 https://github.com/TakHemlata/SSL_Anti-spoofing.git models/upstream/SSL_Anti-spoofing
echo EXTRAS_DONE
