## HighSpeedFLFM: Ultra-High Speed Fourier Light Field Mesoscope Processing Pipeline

Processing and analysis pipeline for ultra-high-speed Fourier light field mesoscope (FLFM) imaging of peg-based jumping gel disks.

This repository contains calibration, reconstruction, tracking, and displacement analysis workflows adapted from the original ant exoskeleton deformation pipeline.
The original repo can be found here: https://github.com/clarebcook/HighSpeedFLFM

## Environment and Installation 

We recommend running this code using a conda environment. The repository and environment can be set up using the following steps. 

1. Clone repository

```
git clone https://github.com/avaduri/HighSpeedFLFM_JumpingDisks.git
cd HighSpeedFLFM_JumpingDisks
```

2. Set up the environment
```
conda env create -f environment.yml
conda activate hsflfm_cpu
conda develop .
```

Note that using `conda develop` requires `conda-build` to be installed. This can be done with: 
```
conda install conda-build
