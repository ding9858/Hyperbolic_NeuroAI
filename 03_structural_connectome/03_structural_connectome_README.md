
# III. Hyperbolic Geometry of the Structural Brain Connectome

## Project overview

This project investigates the geometric organization of the structural brain connectome derived from diffusion MRI (dMRI), examining whether hyperbolic embeddings provide a better representation of structural connectivity than Euclidean embeddings.

The analysis combines probabilistic tractography, voxelwise graph construction, shortest-path analysis, and geometric embedding comparisons.

## Prerequisite: FSL Tractography

The diffusion MRI processing pipeline uses **FSL BEDPOSTX** for estimating voxelwise fiber orientation distributions and **PROBTRACKX** (`probtrackx2`) for probabilistic tractography.

The initial tractography workflow builds upon our previous project:

**[FSL Brain Connectivity — Previous Implementation](https://github.com/ding9858/FSL-brain-conenctivity-)**

Please install and configure [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/) before performing the diffusion MRI processing.

## Source files

- **`Preprocessing.ipynb`** — Main preprocessing notebook covering diffusion MRI outputs, voxelwise orientation-based graph construction, shortest-path analysis, Euclidean and hyperbolic embeddings, and geometric controls.
- **`HyperbolicTest.ipynb`** — Main validation notebook evaluating the geometric properties of the structural connectome through Euclidean–hyperbolic embedding comparisons and multiscale robustness analyses.


## Running the experiments

1. Process the dMRI data using FSL BEDPOSTX and PROBTRACKX.
**[FSL Brain Connectivity — Previous Implementation](https://github.com/ding9858/FSL-brain-conenctivity-)**
2. Run `Preprocessing.ipynb` to construct the voxelwise graphs and perform the initial geometric analyses.
3. Run `HyperbolicTest.ipynb` to evaluate hyperbolic embedding performance and robustness.

All required paths, software configurations, experimental parameters, and version information are documented in the corresponding notebooks.

## Data availability

Due to institutional data-sharing policies, the underlying MRI datasets and derived participant-level data are not publicly distributed. This repository provides the computational code and analysis pipeline only.

Data may be made available upon reasonable request for scientific research, subject to applicable institutional policies and approvals.
