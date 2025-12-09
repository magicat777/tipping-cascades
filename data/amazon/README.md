# Amazon Moisture Recycling Data

This directory contains data from Wunderling et al. (2022) for studying Amazon rainforest tipping cascades.

## Citation

> Wunderling, N., Staal, A., Sakschewski, B., Hirota, M., Tuinenburg, O. A., Donges, J. F., Barbosa, H. M. J., & Winkelmann, R. (2022). Recurrent droughts increase risk of cascading tipping events by outpacing adaptive capacities in the Amazon rainforest. *Proceedings of the National Academy of Sciences*, 119(32), e2120777119. https://doi.org/10.1073/pnas.2120777119

## Data Source

- **Repository**: https://figshare.com/articles/software/Amazon_Adaptation_Model/20089331
- **Provider**: Dr. Arie Staal, Utrecht University

## Contents

- `amazon_adaptation_model/` - Extracted data from Figshare
  - `ERA5_Amazon_monthly_1deg/` - Monthly NetCDF files (2003-2014)
  - `_classes/` - PyCascades modules for Amazon analysis
  - `ensemble_prob_m{1-100}/` - Probabilistic ensemble members

## Data Structure

Each monthly NetCDF file contains:
- `network` (567 × 567): Moisture recycling matrix
- `lat`, `lon` (567): Grid cell coordinates
- `rain`, `evap` (567): Monthly rainfall and evapotranspiration

## Note

The raw data files (*.nc, *.zip) are excluded from git via `.gitignore`.
Clone the repository and download from Figshare to reproduce analyses.
