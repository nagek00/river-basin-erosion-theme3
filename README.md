# River Basin Erosion – Theme 3: Longitudinal profile analysis (Python)

Python port of the course MATLAB/TopoToolbox script `singleBasin.m` (Theme 3, ETH Zurich),
plus a channel-head / Amin sensitivity analysis. No MATLAB or TopoToolbox needed.

| Script | What it does |
|---|---|
| `singleBasin.py` | River network extraction, knickpoints, concavity m/n (χ analysis), k<sub>sn</sub>, basin slope, valley confinement. Writes a stream shapefile, knickpoint CSV and figures. |
| `amin_analysis.py` | Objectives 1 & 2: slope–area analysis (channel-head area) and network / χ–z / m/n for a range of A<sub>min</sub>. |

## Setup

Python ≥ 3.9:

```
python -m pip install -r requirements.txt
```

## Data

The data are **not** in this repository. Use the course folder `Project data-20261008`
(DEM extracted from `DEM.zip`, basin outlines from `Basin outline.zip`). The scripts expect

```
Project data-20261008/
├── DEM/DEM/DEM/<basin>/<basin>_<year>.tif
└── Basin outline/Basin outline/<basin>/<basin>_basin.shp
```

Tell the scripts where that folder is, either by setting an environment variable

```
# PowerShell
$env:RBE_DATA_DIR = "C:\path\to\Project data-20261008"
# macOS / Linux
export RBE_DATA_DIR="/path/to/Project data-20261008"
```

or by editing `DATA_DIR` at the top of `singleBasin.py`.

## Run

Set `basin` and `year` at the top of `singleBasin.py` (default: Plessur 2023), then

```
python amin_analysis.py   # choose Amin first
python singleBasin.py     # main analysis
```

Results go to `output/<basin>_<year>/` (graphs in `figures/`). The DTM is clipped to the
basin outline and resampled to 5 m (`resample_bool`, `res`); 2 m needs several GB of RAM.

## Differences to TopoToolbox

The TopoToolbox functions are re-implemented, not wrapped, so results are close but not identical:

- **Flow routing**: priority-flood with least-cost routing through pits (≈ `'carve'`).
- **knickpointfinder**: each flow path is fitted with the lower convex hull of the carved profile; the largest
  deviation becomes a knickpoint and the fit is repeated until all deviations are < `tol`.
- **crs**: replaced by pit filling + minimum gradient + moving-average smoothing that does not cross knickpoints.
- **mnoptimvar**: m/n that minimises the robust spread of elevations between flow paths at equal χ.
- χ uses drainage area in m² consistently (the MATLAB script mixed cell counts and m²).
