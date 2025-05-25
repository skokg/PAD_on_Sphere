# The PAD on Sphere Python package

#### Description:

The package can be used for calculation of the Precipitation Attribution Distance (PAD) verification metric in a global domain. A detailed description of the methodology is available in the paper listed in the References section. 

The underlying code is written in C++, and a Python ctypes-based wrapper is provided for easy use in the Python environment. A precompiled shared library file is available for easy use with Python on Linux systems (the C++ source code is available in the `source_for_Cxx_shared_library` folder - it can be used to compile the shared library for other types of systems). 

#### Usage:

To see how the package can be used in practice, please refer to the jupyter notebook examples.

#### Author:

Gregor Skok, Faculty of Mathematics and Physics, University of Ljubljana, Slovenia
Llorenç Lledó, ECMWF (for the xarray wrapper and the jupyter notebooks only)

Email: Gregor.Skok@fmf.uni-lj.si

#### References:
Gregor Skok & Llorenç Lledó, 2025, Spatial verification of global precipitation forecasts. Preprint. https://arxiv.org/abs/2407.20624
