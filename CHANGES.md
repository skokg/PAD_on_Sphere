## Changes by Gregor Skok (c09c701)
- Rewrote the C++ kd-tree and attribution code
- Cutoffs are great-circle distances (were chord distances)
- New `random_seed` argument; default picks a random seed and prints it
- Errors raise Python exceptions instead of printing and returning `None`
- C++ errors passed to Python via `PAD_last_error()`; library interface version check (ABI v3)
- Stricter input validation (lat range, same-grid coordinates, alignment, positive areas)
- xarray wrapper: replaced `gridded_output` with `residual_as_df`; different grids returned `(df, [ndarray, ndarray])` in m³
- `distance_m` kept as float
- `weighted_average` no longer modifies its input
- Example plot: PDF y-axis in km⁻¹
- Notebook rebuilt the gridded dataset by hand

## Changes by Llorenç Lledó (on top)
- Removed compiled `.so` from the repo; added `.gitignore`
- `Makefile` to build the `.so` (`make`); replaces `HOW_TO_COMPILE.txt`; README build instructions
- **Undoes Gregor's change:** restored `gridded_output=True` (default) instead of `residual_as_df`
- **Undoes Gregor's change:** restored the 3-tuple return for different grids, residuals in mm
- **Undoes Gregor's change:** `distance_m` stored as int64 again (saves output storage)
- **Undoes Gregor's change:** notebook calls `gridded_output=True` again instead of building the gridded dataset by hand
- **Relaxes Gregor's check:** near-zero areas (float32 cos(90°) at the poles) treated as zero instead of rejected
- New `aggregate_transportplan_at_gridpoints_unequal_grids`: gridded volume/distance for unequal grids (obs distances negative)
- NumPy-style docstrings for all public functions; unit-conversion comments restored
