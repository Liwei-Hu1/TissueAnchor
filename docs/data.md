# Data preparation

Keep local datasets under `data/` or pass any local path as `--data-root`. Each loader takes the directory containing the recording folders. The repository does not distribute patient or animal images, any dataset mirror, or checkpoints.

| Dataset | Source and format |
|---|---|
| StereoMIS tracking | [Tracking release](https://zenodo.org/records/10867949) and [parent release](https://zenodo.org/records/8154924). Each processed clip contains `rect_left/`, `tissue_masks/`, `stereo_depth/`, `times.npy`, `c2w_cm.npy`, and `input_manifest.json`. Stereo depth is float TIFF in mm. Time is normalized and camera-to-world translations are in cm relative to the first camera. |
| DRENDS | [Released data and calibration](https://zenodo.org/records/17598453). Use the processed `ExVivo/` root. It contains `calibration.json`. Each recording contains `rect_left/`, `tool_masks_left/`, `stereo_depth/` and `timestamps.csv`. The loader follows the 960-pixel-wide benchmark input, with a fixed rectified-left camera and stereo priors in mm. Tissue is tool-mask values below 128. Independent ToF is outside the RGB and stereo-prior inputs. |
| EndoNeRF | [Official example access](https://github.com/med-air/EndoNeRF). Read each clip's `images/`, `masks/`, `depth/`, and `poses_bounds.npy` after obtaining the necessary access. The released depth is a stereo estimate in the dataset's prior units, with no conversion to physical units. |
| iMED v2 | [Dataset release](https://doi.org/10.5522/04/32764638.v2). Supply the prepared `author_stereo_v2` inputs at 640×512 with `manifest.json`, `split_manifest.json`, `times.npy`, and `endoscope1/` containing RGB, calibration, metre-unit c2w, left tissue/ROI masks and non-test pseudo-depth/validity. The iMED loader validates this profile. The raw-to-profile iMED preparation pipeline is outside this release. |

## StereoMIS preparation from a local release

```bash
python scripts/prepare_stereomis.py --raw-root data/stereomis_tracking_release \
  --camera-yaml-dir data/stereomis_tracking_calibration --output data/stereomis \
  --p2-2-calibration-pair data/stereomis_parent/P2_0/StereoCalibration.ini \
                         data/stereomis_parent/P2_2/StereoCalibration.ini

git clone https://github.com/L0za007/DRENDS_Eval.git external/DRENDS-Eval
git -C external/DRENDS-Eval checkout e3cd0bde8f3cc4f52e979b73a66889ae381ceeb0
# Follow the DRENDS-Eval setup for its RAFT-Stereo adapter and Middlebury weights first.
python scripts/prepare_stereo_depth.py --data-root data/stereomis \
  --drends-eval external/DRENDS-Eval --scenes P1_1 P2_0 P2_1 P2_2 P2_6 P3_1 P3_2
```

Use the rectified calibration YAMLs from [online_endo_track](https://github.com/mhayoz/online_endo_track) containing `cam.{fx,fy,cx,cy,stereo_baseline}`. Some tracking configurations only supply the P2_0 YAML. The script requires byte-identical official P2_0 and P2_2 calibration INI files before sharing that YAML. The script links original images and masks in place. It makes no synthetic interpolated frames.

## Frame assignment

The held-out sequence is zero-based positions 1, 9, 17, and so on. Each recording uses 48 evenly spaced non-test context positions. Adaptation uses the remaining non-test positions. For DRENDS it uses an evenly spaced 100-frame subset of that pool. For iMED the fixed 200-frame partition comes from the prepared split manifest. Canonical scoring masks are separate files, with a shared mask across methods for each evaluated frame.
