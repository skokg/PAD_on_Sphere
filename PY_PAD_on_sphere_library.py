import numpy as np
import os
from ctypes import *

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

Earth_radius = 6371.0 * 1000.0

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

# search for the PAD C++ shared library file (PAD_on_sphere_Cxx_shared_library.so) in the same folder
libc = CDLL(
    os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
    + os.path.sep
    + "PAD_on_sphere_Cxx_shared_library.so"
)

# define data types of library functions
ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
libc.free_mem_double_array.argtypes = [POINTER(c_double)]
libc.free_mem_double_array.restype = None
libc.calculate_PAD_results_assume_same_grid_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    POINTER(c_size_t),
    c_double,
]
libc.calculate_PAD_results_assume_same_grid_ctypes.restype = POINTER(c_double)
libc.calculate_PAD_results_assume_different_grid_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    POINTER(c_size_t),
    c_double,
]
libc.calculate_PAD_results_assume_different_grid_ctypes.restype = POINTER(c_double)

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


def check_input_array(f, name):

    # test if fields are numpy arrays
    if type(f) is not np.ndarray:
        print(
            "ERROR: the "
            + name
            + ' input array is not a numpy array of type numpy.ndarray, which is not permitted! Perhaps it is a masked arrays, which is also not permitted. Returning "None" as result!'
        )
        return False

    # check dimensions of fields
    if f.ndim != 1:
        print(
            "ERROR: the "
            + name
            + ' input array is not one-dimensional, which is not permitted!. Returning "None" as result!'
        )
        return False

    # check if the array has some elements
    if f.size == 0:
        print(
            "ERROR: the "
            + name
            + ' array does not contain any elements. Returning "None" as result!'
        )
        return False

    # detect non-numeric values
    result = np.where(np.isfinite(f) == False)
    if len(result[0]) > 0:
        print(
            "ERROR: the "
            + name
            + ' arrays contains some non-numeric values, which is not permitted!. Returning "None" as result!'
        )
        return False

    # detect masked array
    if isinstance(f, np.ma.MaskedArray):
        print(
            "ERROR: the "
            + name
            + ' array is a masked array which is not permitted. Returning "None" as result!'
        )
        return False

    return True


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


def calculate_attributions_from_numpy(
    values1,
    values2,
    lat1,
    lon1,
    lat2=None,
    lon2=None,
    same_grid="True",
    distance_cutoff=100 * 1000 * 1000,
):

    if same_grid:
        if (lat2 is not None) or (lon2 is not None):
            print(
                'ERROR: lat2, lon2 were provided but same_grid=True !. Returning "None" as result!'
            )
            return None
    else:
        if (lat2 is None) or (lon2 is None):
            print(
                'ERROR: lat2, lon2 arrays not provided but same_grid=False !. Returning "None" as result!'
            )
            return None

    # check input arrays
    if check_input_array(lat1, "lat1") == False:
        return None
    if check_input_array(lon1, "lon1") == False:
        return None
    if check_input_array(values1, "values1") == False:
        return None
    if check_input_array(values2, "values2") == False:
        return None
    if not same_grid:
        if check_input_array(lat2, "lat2") == False:
            return None
        if check_input_array(lon2, "lon2") == False:
            return None

    # check dimensions of fields
    if lat1.shape != lon1.shape or lat1.shape != values1.shape:
        print(
            'ERROR: the lat1, lon1, values1 arrays do not have the same shape !. Returning "None" as result!'
        )
        return None
    if same_grid:
        if lat1.shape != values2.shape:
            print(
                'ERROR: the lat1, lon1, values2 arrays do not have the same shape !. Returning "None" as result!'
            )
            return None
    else:
        if lat2.shape != lon2.shape or lat2.shape != values2.shape:
            print(
                'ERROR: the lat2, lon2 and values2 arrays do not have the same shape !. Returning "None" as result!'
            )
            return None

    # detect negative values
    if values1[values1 < 0].size > 0 or values2[values2 < 0].size > 0:
        print(
            'ERROR: the values1 or values2 array contains some negative values, which is not allowed . Returning "None" as result!'
        )
        return None

    # check if all the values are zero
    if np.sum(values1) == 0 or np.sum(values2) == 0:
        print(
            'ERROR: the values1 or values2 array contains only zeroes, which is not allowed . Returning "None" as result!'
        )
        return None

    # cast distance to float64 and ensure it is positive
    distance_cutoff = np.float64(distance_cutoff)
    if distance_cutoff <= 0:
        distance_cutoff = 1e99

    # if needed make a copy and cast to float64 - just in case the input fields are integers
    if lat1.dtype != np.float64:
        lat1 = lat1.astype(np.float64, copy=True)
    if lon1.dtype != np.float64:
        lon1 = lon1.astype(np.float64, copy=True)
    if values1.dtype != np.float64:
        values1 = values1.astype(np.float64, copy=True)
    if values2.dtype != np.float64:
        values2 = values2.astype(np.float64, copy=True)
    if not same_grid:
        if lat2.dtype != np.float64:
            lat2 = lat2.astype(np.float64, copy=True)
        if lon2.dtype != np.float64:
            lon2 = lon2.astype(np.float64, copy=True)

    # convert to C_CONTIGUOUS arrays if needed (these are required by C++)
    if lat1.flags["C_CONTIGUOUS"] != True:
        lat1 = np.ascontiguousarray(lat1)
    if lon1.flags["C_CONTIGUOUS"] != True:
        lon1 = np.ascontiguousarray(lon1)
    if values1.flags["C_CONTIGUOUS"] != True:
        values1 = np.ascontiguousarray(values1)
    if values2.flags["C_CONTIGUOUS"] != True:
        values2 = np.ascontiguousarray(values2)
    if not same_grid:
        if lat2.flags["C_CONTIGUOUS"] != True:
            lat2 = np.ascontiguousarray(lat2)
        if lon2.flags["C_CONTIGUOUS"] != True:
            lon2 = np.ascontiguousarray(lon2)

    c_number_of_attributions = c_size_t()

    if same_grid:
        results = libc.calculate_PAD_results_assume_same_grid_ctypes(
            lat1,
            lon1,
            values1,
            values2,
            lat1.shape[0],
            byref(c_number_of_attributions),
            distance_cutoff,
        )
    else:
        results = libc.calculate_PAD_results_assume_different_grid_ctypes(
            lat1,
            lon1,
            values1,
            lat1.shape[0],
            lat2,
            lon2,
            values2,
            lat2.shape[0],
            byref(c_number_of_attributions),
            distance_cutoff,
        )

    number_of_attributions = c_number_of_attributions.value

    # Deserialize
    nlat1 = lat1.shape[0]
    if same_grid:
        nlat2 = nlat1
    else:
        nlat2 = lat2.shape[0]
    attributions = np.asarray(results[0 : number_of_attributions * 4]).reshape(
        number_of_attributions, -1
    )
    non_attributed_values1 = np.asarray(
        results[(number_of_attributions * 4) : (number_of_attributions * 4 + nlat1)]
    )
    non_attributed_values2 = np.asarray(
        results[
            (number_of_attributions * 4 + nlat1) : (
                number_of_attributions * 4 + nlat1 + nlat2
            )
        ]
    )

    libc.free_mem_double_array(results)

    return [attributions, non_attributed_values1, non_attributed_values2]


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


def calculate_PAD_on_sphere_from_attributions(PAD_attributions):
    return np.sum(PAD_attributions[:, 0] * PAD_attributions[:, 1]) / np.sum(
        PAD_attributions[:, 1]
    )
