// Compile command for a linux system
// g++ -fopenmp -O2 -Wall -Wno-unused-result -Wno-unknown-pragmas -shared -o PAD_on_sphere_Cxx_shared_library.so -fPIC CC_PAD_on_sphere_python_lib.cc

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <sys/resource.h>
#include <string.h>
#include <random>
#include <chrono>
#include <cstring>

using namespace std;

#define BAD_DATA_FLOAT -9999

// da prav dela error(..) - da prav displaya line number
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)
#define FU __PRETTY_FUNCTION__
#define TT "\t"
#define tabt "\t"
#define ERRORIF(x) if (x) error(AT,FU, #x)

const rlim_t kStackSize = 1000 * 1024 * 1024;   // min stack size = 16 MB

long random_number_seed=-1;

//#include "CU_utils.cc"
#include "CU_utils_subset.cc"
#include "CU_PAD_on_sphere_code.cc"

vector <vector <double> > lat_lon_value_arrays_to_XYZ_points(const double *lat, const double *lon, const double *values, size_t size)
	{
	vector <vector <double> > points;
	for (size_t il = 0; il < size; il++)
		{
		double x,y,z;
		spherical_to_cartesian_coordinates(deg2rad(lat[il]),deg2rad(lon[il]), Earth_radius, x, y, z);
		points.push_back({x,y,z,values[il]});
		}
	return(points);
	}

extern "C" void free_mem_double_array(double* a)
	{
	delete[] a;
	}

extern "C"  double * calculate_PAD_results_assume_same_grid_ctypes(const double *lat, const double *lon, const double *values1, const double *values2, size_t size, size_t *number_of_attributions, double attribution_distance_cutoff)
	{
    vector <vector <double> > points1 =  lat_lon_value_arrays_to_XYZ_points(lat,lon,values1, size);
    vector <vector <double> > points2 =  lat_lon_value_arrays_to_XYZ_points(lat,lon,values2, size);

	vector <double> non_attributed_values1;
	vector <double> non_attributed_values2;

	vector <vector <double>> results = calculate_PAD_results_assume_same_grid_and_remove_overlap(points1, points2, attribution_distance_cutoff, non_attributed_values1, non_attributed_values2);

	// SERIALIZE the output into a double vector
	vector <double> out;
	for (unsigned long il=0; il < results.size(); il++)
		{
		out.push_back(results[il][0]);
		out.push_back(results[il][1]);
		out.push_back(results[il][2]);
		out.push_back(results[il][3]);
		}
	out.insert(out.end(), non_attributed_values1.begin(), non_attributed_values1.end());
	out.insert(out.end(), non_attributed_values2.begin(), non_attributed_values2.end());

	double* out_arr = new double[out.size()];
	std::copy(out.begin(), out.end(), out_arr);

	*number_of_attributions = results.size();

	return(out_arr);
	}

extern "C"  double * calculate_PAD_results_assume_different_grid_ctypes(const double *lat1, const double *lon1, const double *values1, size_t size1, const double *lat2, const double *lon2, const double *values2, size_t size2, size_t *number_of_attributions, double attribution_distance_cutoff)
	{
    vector <vector <double> > points1 =  lat_lon_value_arrays_to_XYZ_points(lat1,lon1,values1, size1);
    vector <vector <double> > points2 =  lat_lon_value_arrays_to_XYZ_points(lat2,lon2,values2, size2);

	vector <double> non_attributed_values1;
	vector <double> non_attributed_values2;

	vector <vector <double>> results = calculate_PAD_results_assume_different_grid(points1, points2, attribution_distance_cutoff, non_attributed_values1, non_attributed_values2);

	// SERIALIZE the output into a double vector
	vector <double> out;
	for (unsigned long il=0; il < results.size(); il++)
		{
		out.push_back(results[il][0]);
		out.push_back(results[il][1]);
		out.push_back(results[il][2]);
		out.push_back(results[il][3]);
		}
	out.insert(out.end(), non_attributed_values1.begin(), non_attributed_values1.end());
	out.insert(out.end(), non_attributed_values2.begin(), non_attributed_values2.end());

	double* out_arr = new double[out.size()];
	std::copy(out.begin(), out.end(), out_arr);

	*number_of_attributions = results.size();

	return(out_arr);
	}

