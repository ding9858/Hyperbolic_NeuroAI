# Experiment I — Diffusion MRI microstructure inference

This directory contains the analysis code for **Experiment I** of *Multiscale analysis of hyperbolic geometry across neuroscience and artificial intelligence*. It examines NODDI–Watson diffusion MRI microstructure inference using simulation-based inference (SBI), posterior diagnostics, and conventional Dmipy fitting.

## Contents

| File | Purpose |
| --- | --- |
| `SBI_4_dMRI.ipynb` | Main analysis notebook: data normalization, NODDI–Watson forward model, five SBI estimator comparisons (SNPE-A, SNPE-C, SNLE, SNRE, BNRE), posterior diagnostics, and manuscript-related figures. |
| `SBI_Pipeline_SNPE_C.py` | Standalone SNPE-C workflow for voxelwise posterior-mean maps and inference-failure diagnostics. |
| `Conventional_Pipeline.py` | Conventional voxelwise NODDI–Watson fitting using Dmipy, including fitting runtime and microstructural output maps. |

*If the uploaded filenames contain a suffix such as `(1)` or `(4)`, either retain those names in the links above or rename them consistently before use.*

## Input data and path check

The notebook reads input files **relative to its working directory**. Run Jupyter from this directory, or change its paths accordingly:

```text
01_microstructure_inference/
├── SBI_4_dMRI.ipynb
├── SBI_Pipeline_SNPE_C.py
├── Conventional_Pipeline.py
├── data/                       # local, not committed
│   ├── data.nii.gz             # diffusion-weighted MRI
│   ├── nodif_brain_mask.nii.gz # brain mask
│   ├── bvals                   # diffusion-weighting values
│   └── bvecs                   # diffusion-gradient directions
└── output/                     # generated locally
```

Check that the four expected files exist before running the notebook:

```python
from pathlib import Path
required = ["data.nii.gz", "nodif_brain_mask.nii.gz", "bvals", "bvecs"]
missing = [name for name in required if not (Path("data") / name).is_file()]
assert not missing, f"Missing input files: {missing}"
print("Input paths verified.")
```

The first notebook cells load the inputs, construct the acquisition scheme, normalize signals using the mean `b=0` signal, and display diagnostic output. **Refer to the notebook itself for the complete processing steps, model settings, and figures.** Do not upload restricted participant imaging data to GitHub.

## Environment

- **Python:** 3.10.21 is recorded in the notebook metadata (kernel: `sbi_env`).
- **Core dependencies:** `numpy`, `scipy`, `matplotlib`, `torch`, `sbi`, `dmipy`, `fslpy`, `tqdm`, and `jupyter`.
- **Version reproducibility:** Exact versions of these third-party packages are not recorded in the three supplied source files. The older SBI APIs used here (including `SNPE_A` and `sbi.utils.user_input_checks`) may require a compatible `sbi` release. Verify imports and record the versions from a working environment rather than installing unverified latest releases.

```bash
python -m pip list --format=freeze > environment_versions.txt
```

The two Python scripts use command-line inputs for DWI image, mask, b-values, b-vectors, and output prefix. Review their `argparse` option declarations before running them directly; the supplied versions currently combine short and long option names into single strings.

## Outputs

The notebook writes plots beneath `./output/figures/` and NIfTI parameter maps beneath `./output/brainslice/`. The SNPE-C script generates voxelwise ODI, ICVF, ISOVF, noise, and quality-control maps; the conventional script produces matching microstructure parameter maps and a fitting-time record. The original MRI data are not included here.

## Project origin

The original SBI–diffusion MRI research was carried out in **2024 during an academic research visit at the University of Oxford**, and has subsequently been incorporated into the present cross-scale NeuroAI manuscript.

**Original project repository:** [SBI_MRI_diffusion](https://github.com/ding9858/SBI_MRI_diffusion)

For the integrated manuscript and its other two experiments, see the [parent repository](https://github.com/ding9858/Hyperbolic_NeuroAI).
