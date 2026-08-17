# CURE-OOD survival prediction and OOD detection

This repository provides the code and benchmark splits needed for:

- task-specific ViT-MTLR training for OS, LFFS, RFFS, and DFFS;
- survival prediction evaluation on a clinical CSV;
- mixed ID/OOD evaluation with post-hoc OOD detectors.

Sanitized benchmark split CSV templates are included. They retain the schema and patient IDs but contain no clinical values. Checkpoints, NIfTI images, experiment logs, figures, papers, and one-off analysis scripts are intentionally excluded.

## Environment

The tested environment uses Python 3.9, PyTorch 2.1.2, torchvision 0.16.2, CUDA 11.8, PyTorch Lightning 1.8.6, and MONAI 1.4.0.

From the repository root:

```bash
conda env create -f environment.yml
conda activate cure-ood
```

Alternatively, install a CUDA-compatible PyTorch build first, then run:

```bash
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
PYTHONPATH="$PWD/torchmtlr:$PWD" \
python -c "import torch, monai, pytorch_lightning, torchmtlr; print(torchmtlr.__file__); print(torch.__version__, torch.cuda.is_available())"
```

This release bundles a project-specific `torchmtlr` variant whose `MTLR.forward()` returns both cumulative MTLR logits and separate interval logits. The wrapper scripts set `PYTHONPATH` so this repository-local implementation takes precedence over any separately installed `torchmtlr` package.

The training and evaluation configurations expect one visible CUDA GPU. Use the final positional argument of each script to select a physical GPU.

## Data preparation

Download the RADCURE dataset from its official [TCIA collection page](https://www.cancerimagingarchive.net/collection/radcure/) and preprocess each CT/GTV pair to the representation expected by the original project. The default model uses an `80 x 80 x 48` crop with two channels: CT and GTV mask.

The source RADCURE clinical data are distributed by TCIA under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). Use of the included derived benchmark splits requires attribution to the RADCURE dataset and compliance with the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).

The original image preprocessing workflow is included in
[`data_preprocessing/ImagePreProcess.ipynb`](data_preprocessing/ImagePreProcess.ipynb). Before running it, set these four variables in the notebook to paths on your system:

- `oriPath`: downloaded RADCURE DICOM and RTSTRUCT data;
- `savePath`: intermediate NIfTI output directory;
- `resample_path`: directory for images resampled to `2 x 2 x 2` mm spacing;
- `crop_path`: final directory for the model-ready crops.

The notebook performs the following steps:

1. converts DICOM CT and RTSTRUCT data to NIfTI with `dcmrtstruct2nii`;
2. excludes cases without a CT image or GTVp mask;
3. resamples CT images with B-spline interpolation and masks with nearest-neighbor interpolation;
4. crops an `80 x 80 x 48` region around the GTV centroid and clamps CT intensities to `[-500, 500]` HU.

Notebook outputs are cleared in this release. Running it creates the preprocessed files locally; the image data themselves are not distributed in this repository.

The NIfTI directory must contain:

```text
RADCURE-xxxx_image.nii.gz
RADCURE-xxxx_mask_GTV.nii.gz
```

The benchmark CSV templates are directly under `data/`:

```text
ExposureTime_{Train,ID,OOD}.csv
PixelSpacing_{Train,ID,OOD}.csv
SliceThickness_{Train,ID,OOD}.csv
XRayTubeCurrent_{Train,ID,OOD}.csv
```

Each acquisition split contains 1,462 training IDs, 100 ID test IDs, and 100 OOD test IDs. Before running the scripts, obtain the RADCURE clinical data from TCIA and populate the blank columns by matching on `ID`. Clinical CSV requirements and accepted column names are documented in `data/README.md`.

## Training

```bash
./scripts/train.sh \
  os \
  data/ExposureTime_Train.csv \
  data/ExposureTime_ID.csv \
  /absolute/path/to/nifti_cache \
  0
```

Replace `os` with `lffs`, `rffs`, or `dffs`. Useful environment overrides are:

```bash
MAX_EPOCHS=400 BATCH_SIZE=32 NUM_WORKERS=8 LR=1e-3 FOLD=5 \
  ./scripts/train.sh os \
    data/ExposureTime_Train.csv \
    data/ExposureTime_ID.csv \
    nifti_cache 0
```

The best and last checkpoints are written under `outputs/train_<task>/checkpoints/`. Generated checkpoints remain ignored by Git.

## Survival prediction evaluation

```bash
./scripts/survival_eval.sh \
  os \
  /absolute/path/to/model.ckpt \
  data/ExposureTime_Train.csv \
  data/ExposureTime_ID.csv \
  /absolute/path/to/nifti_cache \
  0
```

The training CSV is required during evaluation to reconstruct the MTLR time bins and clinical feature order.

## OOD detection

```bash
./scripts/ood_eval.sh \
  os \
  /absolute/path/to/model.ckpt \
  data/ExposureTime_Train.csv \
  data/ExposureTime_ID.csv \
  data/ExposureTime_OOD.csv \
  /absolute/path/to/nifti_cache \
  msp \
  0
```

Bundled OOD methods are `msp`, `mls`, `ebo`, `gen`, `odin`, `dropout`, `ash`, `scale`, `vim`, `knn`, `dice`, `residual`, `nnguide`, `mds`, and `rmds`. Method-specific settings are in `configs/postprocessors/`.

OOD metrics and per-sample scores are written under `outputs/ood_<task>_<method>/`.

## Direct Hydra usage

The wrapper scripts are recommended because they provide all required portable paths and select the bundled `torchmtlr`. For direct Hydra commands, use the same import-path prefix:

```bash
PYTHONPATH="$PWD/torchmtlr:$PWD" python train.py [Hydra overrides...]
PYTHONPATH="$PWD/torchmtlr:$PWD" python test.py [Hydra overrides...]
```

Advanced users can inspect the wrapper commands and override any Hydra value directly through `train.py` or `test.py`.

## Release note

This repository distributes sanitized benchmark split CSV templates containing patient IDs and column headers, but no clinical values, corresponding RADCURE images, or pretrained model weights. The bundled `torchmtlr` component retains its own license in `torchmtlr/LICENSE`; additional provenance and licensing information is documented in `THIRD_PARTY_NOTICES.md` and `DATA_LICENSE.md`.

## License

Except where otherwise noted, the original code in this repository is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their original licenses; see `THIRD_PARTY_NOTICES.md`. The benchmark CSV files are subject to the RADCURE data license and attribution requirements described in `DATA_LICENSE.md`.

## Acknowledgements

The benchmark data are derived from the RADCURE collection hosted by The Cancer Imaging Archive (TCIA): Welch, M. L., et al. (2023), *Computed Tomography Images from Large Head and Neck Cohort (RADCURE)*, Version 4, https://doi.org/10.7937/J47W-NM11.

This repository builds on the [Multi-Label Survival Prediction](https://github.com/LabAIRT/Multi-Label-Survival-Prediction) project. Please refer to its GitHub repository for the original implementation.

If you use this code, please also cite:

```bibtex
@article{chen2024vision,
  title={Vision transformer-based multilabel survival prediction for oropharynx cancer after radiation therapy},
  author={Chen, Meixu and Wang, Kai and Wang, Jing},
  journal={International Journal of Radiation Oncology* Biology* Physics},
  volume={118},
  number={4},
  pages={1123--1134},
  year={2024},
  publisher={Elsevier}
}
```
