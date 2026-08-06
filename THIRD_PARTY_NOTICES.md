# Third-party notices

This repository includes or adapts code from third-party projects. Their attribution and license notices must be retained.

## torchmtlr

The bundled `torchmtlr` component is based on Michal Kazmierski's `torchmtlr` project and is distributed under the MIT License. The license is included at `torchmtlr/LICENSE`, and the upstream project is https://github.com/mkazmier/torchmtlr.

This repository modifies `MTLR.forward()` to return both cumulative MTLR logits and separate interval logits for OOD scoring.

## MONAI

The ViT and patch-embedding implementations in `src/models/components/net_vit.py`, `net_vit_img.py`, and `net_cnvit.py` include code from the MONAI Consortium. The original copyright and Apache License 2.0 headers are retained in those files. A copy of the Apache License 2.0 is included at `licenses/Apache-2.0.txt`.

Upstream project: https://github.com/Project-MONAI/MONAI

## Swin Transformer and TransMorph-derived components

The files `src/models/components/net_swin.py` and `net_swin_img.py` retain their original source and author attribution, including references to the Swin Transformer semantic-segmentation implementation and VoxelMorph-derived utilities.

Referenced upstream projects:

- https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
- https://github.com/voxelmorph/voxelmorph

## ODIN

`src/postprocessors/odin_postprocessor.py` is adapted from the ODIN implementation referenced in its source header:

- https://github.com/facebookresearch/odin

Python packages installed through `requirements.txt` or `environment.yml` are not redistributed by this repository and remain subject to their respective licenses.
