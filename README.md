## HighSpeedFLFM: Ultra-High Speed Fourier Light Field Mesoscope Processing Pipeline

Processing and analysis pipeline for ultra-high-speed Fourier light field mesoscope (FLFM) imaging of PEG-based jumping gel disks, provided by the Crosby Lab from UMASS.

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
```

## Data Organization
Specimen videos and images should be recorded in a metadata sheet and placed within a folder within the user's directory. `home_directory` in `hsflfm/config.py` should then be set to this folder. 

## Quick Test
The current pipline for visualizing results from single trials of disk deformation is as follows:

1. run_calibration.ipynb (including remove_identified_vertices.py & select_alignment_points.py)
2. match_points_gui.py
3. process_without_alignment.ipynb

## Analysis
New bulk analysis scripts have been added(see disk_result_bulk_display.ipynb) as a starting point for analyzing result files in bulk
