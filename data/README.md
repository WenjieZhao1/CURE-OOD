# Data layout

Sanitized benchmark split CSV templates are distributed directly in this `data/` directory. They retain the original column headers and patient IDs, while every clinical and acquisition value is blank. CT images, masks, clinical values, and checkpoints are not included.

Each acquisition parameter has three files:

| Parameter | Train | ID test | OOD test |
| --- | --- | --- | --- |
| Exposure time | `ExposureTime_Train.csv` | `ExposureTime_ID.csv` | `ExposureTime_OOD.csv` |
| Pixel spacing | `PixelSpacing_Train.csv` | `PixelSpacing_ID.csv` | `PixelSpacing_OOD.csv` |
| Slice thickness | `SliceThickness_Train.csv` | `SliceThickness_ID.csv` | `SliceThickness_OOD.csv` |
| X-ray tube current | `XRayTubeCurrent_Train.csv` | `XRayTubeCurrent_ID.csv` | `XRayTubeCurrent_OOD.csv` |

Every acquisition split contains 1,462 unique training IDs, 100 unique ID test IDs, and 100 unique OOD test IDs. The three cohorts do not overlap.

Before running training or evaluation, download the RADCURE clinical data from TCIA and populate every blank field by matching on `ID`. Do not change the supplied cohort membership or row order.

Each completed clinical CSV must contain `ID` plus the four survival outcomes. The loader accepts either the original RADCURE names
`Death`, `RT2Follow`, `LF`, `RT2LF`, `RF`, `RT2RF`, `DF`, `RT2DF`, or their normalized equivalents
`event1`, `time1`, ..., `event4`, `time4`. Additional columns are treated as clinical covariates.

The preprocessed NIfTI directory must contain one image and one GTV mask per patient:

```text
<ID>_image.nii.gz
<ID>_mask_GTV.nii.gz
```

After populating the CSV templates, pass their paths and the separately downloaded NIfTI directory to the scripts in `scripts/`.
