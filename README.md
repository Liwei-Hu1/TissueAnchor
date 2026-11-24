# TissueAnchor

TissueAnchor reconstructs deforming tissue from endoscopic video with a reusable dynamic radiance field. A frozen OpenD4RT encoder and point-query decoder condition a per-recording cache on 48 context frames. A shared field with 472,173 trainable parameters combines 1,024 dynamic anchors with motion-aligned appearance and renders RGB and expected depth. A new recording can use the source field directly or adapt a separate copy.

## Repository layout

```text
configs/                  Source-training hyperparameters
docs/                     Data, evaluation and comparison methods
scripts/                  StereoMIS preparation and OpenD4RT checkpoint retrieval
src/tissueanchor/
  datasets/               StereoMIS, DRENDS, EndoNeRF and iMED loaders
  models/                 Shared field and volume rendering
  engines/                Cache, source training, target inference and scoring
  metrics/                Masked image quality metrics
tests/                    CPU smoke tests
```

## Installation

The original implementation used PyTorch 2.8 with CUDA 12.8. Use Python 3.9–3.12 on Linux and a driver compatible with the selected PyTorch wheel. Use a CUDA GPU for anchor preparation and source training. The OpenD4RT checkpoint is about 13 GiB on disk.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 1. Prepare data and the frozen conditioner

See [dataset formats and split rules](docs/data.md). For the full source workflow, prepare the seven StereoMIS tracking clips with the registered calibration and RAFT-Stereo Middlebury priors. Already-prepared inputs can be supplied directly using `--data-root`.

```bash
git clone https://github.com/Lijiaxin0111/Open-d4rt.git external/OpenD4RT
git -C external/OpenD4RT checkout 403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3
bash scripts/fetch_opend4rt.sh
base=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG

tissueanchor-cache --data-root data/stereomis --output runs/stereomis_cache \
  --opend4rt external/OpenD4RT --checkpoint "$base/opend4rt.ckpt" \
  --model-config "$base/model.yaml"
```

The checkpoint is from the **unofficial OpenD4RT implementation**. Cache creation checks its SHA256 before loading. Existing compatible caches can be supplied to subsequent commands without loading the frozen base model.

## 2. Train the shared source field

```bash
tissueanchor-train --data-root data/stereomis --cache-root runs/stereomis_cache \
  --config configs/stereomis_source.json --output runs/source_seed42
```

The source clips are `P1_1`, `P2_0`, `P2_1`, and `P2_2`. Checkpoint selection uses the fixed four-frame probe on `P2_6`. The default configuration uses seed 42, 3,000 updates, 512 rays per update, and 48 samples per ray. `P3_1` and `P3_2` are reserved for target evaluation. Use `runs/source_seed42/best.pt` after the run. If you already have the original seed-42 checkpoint, its SHA256 is `9671b2e1b0c9b071401f9b8a1d3a8b813500012e8e9c95ec8268b3f5e13ec93c`.

## 3. Render a target

```bash
tissueanchor-infer --dataset stereomis --scene P3_1 --data-root data/stereomis \
  --cache-root runs/stereomis_cache --checkpoint runs/source_seed42/best.pt \
  --steps 0 --output runs/p3_1_zero
```

Use `--steps 100` or `--steps 1000` for optional single-recording adaptation. Saved RGB, expected depth, and a frame manifest go into the output directory. For another dataset, use the corresponding dataset flag and create a cache from that recording with `tissueanchor-cache --dataset ... --scenes ...`. Conditioning and inference use context and camera inputs. Target updates additionally use the disjoint adaptation pool.

## 4. Score saved images

```bash
tissueanchor-evaluate --predictions runs/p3_1_zero/renders \
  --references evaluation_inputs/p3_1/rgb --masks evaluation_inputs/p3_1/tissue \
  --output runs/p3_1_zero/metrics.json
```

The evaluator accepts three sets of PNGs with identical frame IDs. White mask pixels retain tissue. It computes PSNR on retained pixels and SSIM and AlexNet LPIPS on black-masked RGB. Supply the canonical evaluation masks for comparison with the manuscript. The first LPIPS invocation may fetch AlexNet weights. See [evaluation scope](docs/reproduction.md) and [baseline revisions](docs/baselines.md).

## Data and release permissions

Obtain datasets and external checkpoints from their own publishers. This repository ships source code and configuration. Confirm the TissueAnchor source-code licence with the authors before public release. [Release checklist](docs/release.md)
