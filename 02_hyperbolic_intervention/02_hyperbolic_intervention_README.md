# II. Hyperbolic Interventions in MRI Latent Representations

## Project overview

This project investigates curvature-controlled hyperbolic interventions in the latent space of a pretrained physics-informed latent diffusion model (phy-LDM) for multimodal brain MRI generation.

The experiments compare Euclidean and hyperbolic latent drifting, including endpoint-matched geometric controls and multiseed curvature-response analyses. The pretrained model remains frozen throughout.

## Prerequisite: Pretrained MRI model

**This project builds directly on our previously pretrained phy-LDM model.**

Before running the experiments, please clone and set up the original model repository:

**[phy-ldm-mri — Original pretrained model](https://github.com/sven-luepke/phy-ldm-mri)**

```bash
git clone https://github.com/sven-luepke/phy-ldm-mri.git
```

Follow the original repository's installation instructions and obtain the required pretrained VAE and diffusion model checkpoints.

The hyperbolic experiments use these pretrained weights without retraining the model.

## Source files

- **`manuscript.ipynb`** — Main experimental notebook containing the complete workflow, experimental configurations, analyses, results, and figure-generation procedures used in the manuscript.
- **`geometry.py`** — Implementation of the hyperbolic geometry operations, including Poincaré mappings, Möbius addition, parallel transport, and curvature-controlled latent drifting.
- **`generate.py`, `models.py`, `train_ldm.py`, `train_vae.py`, `utils.py`** — Supporting model implementation, generation, training, and utility scripts associated with the original phy-LDM framework.

## Running the experiments

1. Install and configure the original phy-LDM framework.
2. Set the pretrained checkpoint paths and local data/output directories in `manuscript.ipynb`.
3. Execute the notebook sequentially to reproduce the geometric interventions and associated analyses.

For detailed experimental methods, parameters, results, and figure generation, refer to **`manuscript.ipynb`**. For the mathematical implementation of the hyperbolic interventions, refer to **`geometry.py`**.

## Project origin

Developed as part of the Munich computational neuroscience research project investigating geometric interventions in learned brain MRI representations, building upon the pretrained phy-LDM framework by Sven Lüpke and collaborators.
