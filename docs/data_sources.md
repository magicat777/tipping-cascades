# Data Sources

## Amazon Moisture Recycling Network Data

**Source**: Wunderling et al. (2022), PNAS

**Full Citation**:
> Wunderling, N., Staal, A., Sakschewski, B., Hirota, M., Tuinenburg, O. A., Donges, J. F., Barbosa, H. M. J., & Winkelmann, R. (2022). Recurrent droughts increase risk of cascading tipping events by outpacing adaptive capacities in the Amazon rainforest. *Proceedings of the National Academy of Sciences*, 119(32), e2120777119. https://doi.org/10.1073/pnas.2120777119

**Data Repository**: [Figshare - Amazon Adaptation Model](https://figshare.com/articles/software/Amazon_Adaptation_Model/20089331)

**Data Provider**: Dr. Arie Staal, Utrecht University (personal communication, December 2025)

### Dataset Description

The dataset contains spatially-explicit moisture recycling data for the Amazon basin:

- **Temporal Coverage**: 2003-2014 (monthly data)
- **Spatial Resolution**: 1° × 1° grid
- **Grid Cells**: 567 cells covering the Amazon basin
- **Data Source**: ERA5 reanalysis

### Variables

| Variable | Dimensions | Description |
|----------|------------|-------------|
| `network` | 567 × 567 | Moisture recycling matrix (mm/month) |
| `lat` | 567 | Latitude of each grid cell |
| `lon` | 567 | Longitude of each grid cell |
| `rain` | 567 | Total rainfall (mm/month) |
| `evap` | 567 | Total evapotranspiration (mm/month) |

### File Structure

```
data/amazon/amazon_adaptation_model/
├── ERA5_Amazon_monthly_1deg/          # 140 monthly NetCDF files
│   ├── Amazon_monthly_1deg_2003_01.nc
│   ├── Amazon_monthly_1deg_2003_02.nc
│   └── ...
├── _classes/                          # PyCascades modules for Amazon
├── ensemble_prob_m{1-100}/            # 100 probabilistic ensemble members
└── *.py                               # Analysis scripts from paper
```

### Usage Notes

1. The 567×567 network matrix represents moisture flow between grid cells
2. Non-zero entries indicate moisture recycling pathways
3. Network is derived from WAM-2layers atmospheric moisture tracking model
4. Ensemble members represent uncertainty in forest dieback thresholds

### Acknowledgments

We thank Dr. Arie Staal (Utrecht University) for providing access to this dataset and for helpful discussions on Amazon moisture recycling dynamics.

---

## Other Data Sources

*(Additional data sources will be documented here as they are added to the project)*
