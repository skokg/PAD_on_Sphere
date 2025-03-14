#include "CU_kdtree_with_index.cc"

double Earth_radius= 6371*1000;

double great_circle_distance_to_euclidian_distance(double great_circle_distance)
	{
	return(2*Earth_radius*sin(great_circle_distance/(2*Earth_radius)));
	}

double euclidian_distance_to_great_circle_distance(double euclidian_distance)
	{
	return(2*Earth_radius*asin(euclidian_distance/(2*Earth_radius)));
	}

vector <vector <double> > convert_latlon_points_to_3D_points(const vector <vector <double> > &points_lat_lon, vector <size_t> &indexmap)
	{
	vector <vector <double> > points;
	for ( unsigned long il=0; il < points_lat_lon.size(); il++)
		//if (points_lat_lon[il][2] > 0)
			{
			double x,y,z;
			spherical_to_cartesian_coordinates(deg2rad(points_lat_lon[il][0]), deg2rad(points_lat_lon[il][1]), Earth_radius, x, y, z);
			points.push_back({x,y,z,points_lat_lon[il][2]});
			indexmap.push_back(il);
			}
	return(points);
	}


kdtree::Point_str kdtree_Point_str_from_point(const vector <double> &point, size_t ind)
	{
	kdtree::Point_str p = { {point.begin(), point.end() -1} , ind };
	return(p);
	}

vector <double> perform_one_PAD_iteration(const vector <vector <double>> &points1, const  vector <vector <double>> &points2,  vector <double> &values1, vector <double> &values2,  vector <size_t> &index_list1, vector <size_t> &index_list2, kdtree::KdTree &kdtree1, kdtree::KdTree &kdtree2, long &idum, bool f1_is_fa)
	{
	// choose the last point from list1
	auto ind1=index_list1.back();
	kdtree::Point_str p1 = kdtree_Point_str_from_point(points1[ind1], ind1);


	// find the closes non-zero point in the other field
	//auto begin = std::chrono::high_resolution_clock::now();
	auto node = kdtree2.findNearestNode(p1);
	//temp.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9);
	ERRORIF(node == nullptr);
	kdtree::Point_str p2 = node->val;
	auto ind2=p2.index;

	double min_squared_distance=squared_euclidian_distance_multidimensional(p1.coords, p2.coords);

	//cout << ind1 << " " << values1[ind1] << endl;
	//cout << ind2 << " " << values2[ind2] << endl;

	// value reduction
	double value_reduction= min(values1[ind1],values2[ind2]);
	values1[ind1]-=value_reduction;
	values2[ind2]-=value_reduction;

	//cout << list_nzp_f1.size() << " " << list_nzp_f2.size() << endl;

	// if necessary remove the point from kdtree1
	if (values1[ind1] == 0)
		kdtree1.deleteNode(p1);
	// else swap with random point so this is not the last point anymore - so it is not automatically chosen the next time
	else
		swap(index_list1.back(),index_list1[ floor(ran2(&idum) * (double)index_list1.size()) ]);

	// if necessary remove the point from kdtree2
	if (values2[ind2] == 0)
		kdtree2.deleteNode(p2);

	//cout << list_nzp_f1.size() << " " << list_nzp_f2.size() << endl;


	vector <double> result;

	//result.push_back(sqrt(min_squared_distance));
	result.push_back(euclidian_distance_to_great_circle_distance(sqrt(min_squared_distance)));
	result.push_back(value_reduction);
	if (f1_is_fa)
		{
		result.push_back(ind1);
		result.push_back(ind2);
		}
	else
		{
		result.push_back(ind2);
		result.push_back(ind1);
		}

	//cout << index_list1.size() << " " << index_list2.size() << endl;

	// remove all zero points at the end of list
	while (index_list1.size() > 0 && values1[index_list1.back()] == 0)
		index_list1.pop_back();

	// remove all zero points at the end of vector
	while (index_list2.size() > 0 && values2[index_list2.back()] == 0)
		index_list2.pop_back();

	//cout << index_list1.size() << " " << index_list2.size() << endl;

	return(result);
	}


vector <double> perform_one_PAD_iteration_with_attribution_distance_cutoff(const vector <vector <double>> &points1, const  vector <vector <double>> &points2,  vector <double> &values1, vector <double> &values2,  vector <size_t> &index_list1, vector <size_t> &index_list2, kdtree::KdTree &kdtree1, kdtree::KdTree &kdtree2, long &idum, bool f1_is_fa, double squared_attribution_distance_cutoff)
	{
	// choose the last point from list1
	auto ind1=index_list1.back();
	kdtree::Point_str p1 = kdtree_Point_str_from_point(points1[ind1], ind1);

	// find the closes non-zero point in the other field
	//auto begin = std::chrono::high_resolution_clock::now();
	auto node = kdtree2.findNearestNode(p1);
	//temp.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9);
	ERRORIF(node == nullptr);
	kdtree::Point_str p2 = node->val;
	auto ind2=p2.index;

	double min_squared_distance=squared_euclidian_distance_multidimensional(p1.coords, p2.coords);

	//cout << ind1 << " " << values1[ind1] << endl;
	//cout << ind2 << " " << values2[ind2] << endl;

	vector <double> result;

	// distance larger than the attribution distance cutoff
	if (min_squared_distance >  squared_attribution_distance_cutoff)
		{
		// remove the point from the kdtree and index
		kdtree1.deleteNode(p1);
		index_list1.pop_back();

		//cout << index_list1.size() << " " << index_list2.size() << " " << sqrt(min_squared_distance) << endl;

		}

	else
		{
			// value reduction
		double value_reduction= min(values1[ind1],values2[ind2]);
		values1[ind1]-=value_reduction;
		values2[ind2]-=value_reduction;

		//cout << list_nzp_f1.size() << " " << list_nzp_f2.size() << endl;

		// if necessary remove the point from kdtree1
		if (values1[ind1] == 0)
			kdtree1.deleteNode(p1);
		// else swap with random point so this is not the last point anymore - so it is not automatically chosen the next time
		else
			swap(index_list1.back(),index_list1[ floor(ran2(&idum) * (double)index_list1.size()) ]);

		// if necessary remove the point from kdtree2
		if (values2[ind2] == 0)
			kdtree2.deleteNode(p2);

		//cout << list_nzp_f1.size() << " " << list_nzp_f2.size() << endl;



		//result.push_back(sqrt(min_squared_distance));
		result.push_back(euclidian_distance_to_great_circle_distance(sqrt(min_squared_distance)));
		result.push_back(value_reduction);
		if (f1_is_fa)
			{
			result.push_back(ind1);
			result.push_back(ind2);
			}
		else
			{
			result.push_back(ind2);
			result.push_back(ind1);
			}

		//cout << index_list1.size() << " " << index_list2.size() << endl;
		}

	// remove all zero points at the end of list
	while (index_list1.size() > 0 && values1[index_list1.back()] == 0)
		index_list1.pop_back();

	// remove all zero points at the end of vector
	while (index_list2.size() > 0 && values2[index_list2.back()] == 0)
		index_list2.pop_back();

		//cout << index_list1.size() << " " << index_list2.size() << endl;

	return(result);

	}



// check points for negative values, subsample if necessary, and normalize
vector <double> check_points_subsample_and_normalize(const vector <vector <double> > &points, long max_number_of_nonzero_points)
	{
	// check and count non-zero points
	size_t count=0;
	vector <double> values;
	double sum=0;
	for (unsigned long il=0; il < points.size(); il++)
		{
		double val=points[il].back();
		ERRORIF( val < 0 );
		values.push_back(val);
		if (val > 0)
			{
			count++;
			sum+=val;
			}
		}

	// check if there are no non-zero points
	ERRORIF(count == 0);

	// subset points by setting some of their values to zero
	if ((long)count > max_number_of_nonzero_points && max_number_of_nonzero_points > 0)
		{
		vector <double> v;
		long idum=1;
		generate_random_binary_vector_with_fixed_number_of_1_values(max_number_of_nonzero_points, count, v, idum);

		size_t counter=0;
		sum=0;
		for (unsigned long il=0; il < values.size(); il++)
			if (values[il] > 0)
				{
				if (v[counter] == 0)
					values[il]=0;
				else
					sum+=values[il];
				counter++;
				}

		cout << "Notice: Subsampling " << max_number_of_nonzero_points << " nonzero points out of " << count << "." << endl;
		}

	// normalize
	for (unsigned long il=0; il < values.size(); il++)
		values[il]*=1.0/sum;

	return(values);
	}

// check points for negative values
vector <double> check_points(const vector <vector <double> > &points)
	{
	// check and count non-zero points
	size_t count=0;
	vector <double> values;
	for (unsigned long il=0; il < points.size(); il++)
		{
		double val=points[il].back();
		ERRORIF( val < 0 );
		values.push_back(val);
		if (val > 0)
			{
			count++;
			}
		}

	// check if there are no non-zero points
	ERRORIF(count == 0);

	return(values);
	}


// from non-zero points construct kdtree and shuffled index list
kdtree::KdTree construct_kdtree_with_shuffled_index_list(const vector <vector <double>> &points,  const vector <double> &values, vector <size_t> &index_list)
	{
	index_list.clear();
	vector<kdtree::Point_str> pointsX;
	for (size_t ip=0; ip < points.size(); ip++)
		if (values[ip] > 0)
			{
			pointsX.push_back(kdtree_Point_str_from_point(points[ip], ip));
			index_list.push_back(ip);
			}

	// shuffle list
	std::shuffle(index_list.begin(), index_list.end(), rng_state_global);

	// construct kdtree
	kdtree::KdTree kdtree;
	kdtree.buildKdTree(pointsX);

	return(kdtree);
	}

vector <vector <double>> calculate_PAD_results(const vector <vector <double> > &points1, const vector <vector <double> > &points2, long max_number_of_nonzero_points)
	{

	// check points for negative values, subsample and normalize
	auto begin = std::chrono::high_resolution_clock::now();
	vector <double> values1 = check_points_subsample_and_normalize(points1, max_number_of_nonzero_points);
	vector <double> values2 = check_points_subsample_and_normalize(points2, max_number_of_nonzero_points);
	cout << "----- preprocessing: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;
	//cout << sum_vector(values1) << endl;
	//cout << sum_vector(values2) << endl;

	//cout << output_vector_as_string(values1, " ") << endl;
	//cout << output_vector_as_string(values2, " ") << endl;

	// construct kdtree with shuffled non-zero points index list
	begin = std::chrono::high_resolution_clock::now();
	vector <size_t> index_list1;
	kdtree::KdTree kdtree1 = construct_kdtree_with_shuffled_index_list(points1, values1, index_list1);
	vector <size_t> index_list2;
	kdtree::KdTree kdtree2 = construct_kdtree_with_shuffled_index_list(points2, values2, index_list2);
	cout << "----- kdtree construction: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" <<  endl;

	//kdtree1.printKdTree();
	cout << index_list1.size() << endl;
	cout << index_list2.size() << endl;
	//cout << output_vector_as_string(index_list1, " ") << endl;
	//cout << values1[index_list1.back()] << endl;
	//cout << values2[index_list2.back()] << endl;

	long idum=1;
	vector <vector <double>> out;
	bool fa_turn=true;

	// iteratevely remove non-zero points
	begin = std::chrono::high_resolution_clock::now();
	while (index_list1.size() > 0 && index_list2.size() > 0)
		{
		vector <double> result;
		if (fa_turn)
			{
			result = perform_one_PAD_iteration(points1, points2, values1, values2, index_list1, index_list2, kdtree1, kdtree2, idum, fa_turn);
			fa_turn=false;
			}
		else
			{
			result = perform_one_PAD_iteration(points2, points1, values2, values1, index_list2, index_list1, kdtree2, kdtree1, idum, fa_turn);
			fa_turn=true;
			}
		//cout << output_vector_as_string(result, " ") << endl;
		out.push_back(result);
		}
	cout << "----- attribution: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;
	return(out);
	}

vector <vector <double>> calculate_PAD_results_assume_same_grid_and_remove_overlap(const vector <vector <double> > &points1, const vector <vector <double> > &points2, double attribution_distance_cutoff, vector <double> &values1, vector <double> &values2)
	{

	double	squared_attribution_distance_cutoff = attribution_distance_cutoff * attribution_distance_cutoff;

	// check if the number of points is the same
	ERRORIF(points1.size() != points2.size());

	auto begin = std::chrono::high_resolution_clock::now();
	// check points for negative values
	values1 = check_points(points1);
	values2 = check_points(points2);
	cout << "----- preprocessing: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	//cout << sum_vector(values1) << endl;
	//cout << sum_vector(values2) << endl;

	//cout << "---" << endl;
	//cout << points1.size() << endl;
	//cout << points2.size() << endl;

	//cout << "---" << endl;
	//long temp_counter1=0;
	//long temp_counter2=0;
	//for (unsigned long il=0; il < values1.size(); il++)
	//	{
	//	if (values1[il] > 0) temp_counter1++;
	//	if (values2[il] > 0) temp_counter2++;
	//	}
	//cout << temp_counter1 << endl;
	//cout << temp_counter2 << endl;


	vector <vector <double>> out;
	bool any_non_zero_points1= false;
	bool any_non_zero_points2= false;
	// remove overlap
	for (unsigned long il=0; il < points1.size(); il++)
		{
		double value_reduction= min(values1[il],values2[il]);
		if (value_reduction > 0)
			{
			values1[il]-=value_reduction;
			values2[il]-=value_reduction;
			out.push_back({0,value_reduction,(double)il,(double)il});
			}

		if (any_non_zero_points1 == false)
			if (values1[il] > 0) any_non_zero_points1=true;
		if (any_non_zero_points2 == false)
			if (values2[il] > 0) any_non_zero_points2=true;
		}


	//cout << output_vector_as_string(values1, " ") << endl;
	//cout << output_vector_as_string(values2, " ") << endl;

	// only do nearest-neighbour-atrribution if any larger-than-zero points remain in both fields - otherwise the k-d tree construction will fail
	if (any_non_zero_points1 && any_non_zero_points2)
		{
		// construct kdtree with shuffled non-zero points index list
		begin = std::chrono::high_resolution_clock::now();
		vector <size_t> index_list1;
		kdtree::KdTree kdtree1 = construct_kdtree_with_shuffled_index_list(points1, values1, index_list1);
		vector <size_t> index_list2;
		kdtree::KdTree kdtree2 = construct_kdtree_with_shuffled_index_list(points2, values2, index_list2);
		cout << "----- kdtree construction: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

		//kdtree1.printKdTree();
		//cout << "---" << endl;
		//cout << index_list1.size() << endl;
		//cout << index_list2.size() << endl;
		//cout << output_vector_as_string(index_list1, " ") << endl;
		//cout << values1[index_list1.back()] << endl;
		//cout << values2[index_list2.back()] << endl;

		long idum=1;
		bool fa_turn=true;

		// iteratevely remove non-zero points
		begin = std::chrono::high_resolution_clock::now();
		while (index_list1.size() > 0 && index_list2.size() > 0)
			{
			vector <double> result;
			if (fa_turn)
				{
				result = perform_one_PAD_iteration_with_attribution_distance_cutoff(points1, points2, values1, values2, index_list1, index_list2, kdtree1, kdtree2, idum, fa_turn, squared_attribution_distance_cutoff);
				fa_turn=false;
				}
			else
				{
				result = perform_one_PAD_iteration_with_attribution_distance_cutoff(points2, points1, values2, values1, index_list2, index_list1, kdtree2, kdtree1, idum, fa_turn, squared_attribution_distance_cutoff);
				fa_turn=true;
				}
			//cout << output_vector_as_string(result, " ") << endl;
			//cout << index_list1.size() << " " << index_list2.size() << " " << output_vector_as_string(result, " ") << endl;

			if (result.size() > 0)
				out.push_back(result);
			}
		cout << "----- attribution: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;
		}

	return(out);
	}


vector <vector <double>> calculate_PAD_results_assume_different_grid(const vector <vector <double> > &points1, const vector <vector <double> > &points2, double attribution_distance_cutoff, vector <double> &values1, vector <double> &values2)
	{
	double	squared_attribution_distance_cutoff = attribution_distance_cutoff * attribution_distance_cutoff;

	auto begin = std::chrono::high_resolution_clock::now();
	// check points for negative values
	values1 = check_points(points1);
	values2 = check_points(points2);
	cout << "----- preprocessing: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	//cout << sum_vector(values1) << endl;
	//cout << sum_vector(values2) << endl;

	//cout << "---" << endl;
	//cout << points1.size() << endl;
	//cout << points2.size() << endl;

	//cout << "---" << endl;

	//long temp_counter1=0;
	//long temp_counter2=0;
	//for (unsigned long il=0; il < values1.size(); il++)
	//	if (values1[il] > 0) temp_counter1++;
	//for (unsigned long il=0; il < values1.size(); il++)
	//	if (values2[il] > 0) temp_counter2++;
	//ERRORIF(temp_counter1 == 0 || temp_counter2 == 0);
	//cout << temp_counter1 << endl;
	//cout << temp_counter2 << endl;

	begin = std::chrono::high_resolution_clock::now();
	vector <size_t> index_list1;
	kdtree::KdTree kdtree1 = construct_kdtree_with_shuffled_index_list(points1, values1, index_list1);
	vector <size_t> index_list2;
	kdtree::KdTree kdtree2 = construct_kdtree_with_shuffled_index_list(points2, values2, index_list2);
	cout << "----- kdtree construction: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	//kdtree1.printKdTree();
	//cout << "---" << endl;
	//cout << index_list1.size() << endl;
	//cout << index_list2.size() << endl;
	//cout << output_vector_as_string(index_list1, " ") << endl;
	//cout << values1[index_list1.back()] << endl;
	//cout << values2[index_list2.back()] << endl;

	vector <vector <double>> out;
	long idum=1;
	bool fa_turn=true;

	// iteratevely remove non-zero points
	begin = std::chrono::high_resolution_clock::now();
	while (index_list1.size() > 0 && index_list2.size() > 0)
		{
		vector <double> result;
		if (fa_turn)
			{
			result = perform_one_PAD_iteration_with_attribution_distance_cutoff(points1, points2, values1, values2, index_list1, index_list2, kdtree1, kdtree2, idum, fa_turn, squared_attribution_distance_cutoff);
			fa_turn=false;
			}
		else
			{
			result = perform_one_PAD_iteration_with_attribution_distance_cutoff(points2, points1, values2, values1, index_list2, index_list1, kdtree2, kdtree1, idum, fa_turn, squared_attribution_distance_cutoff);
			fa_turn=true;
			}
		//cout << output_vector_as_string(result, " ") << endl;
		//cout << index_list1.size() << " " << index_list2.size() << " " << output_vector_as_string(result, " ") << endl;

		if (result.size() > 0)
			out.push_back(result);
		}
	cout << "----- attribution: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	return(out);
	}



double calculate_PAD_from_PAD_results(const vector <vector <double>> &results)
	{
	double sum_weights=0;
	double sum=0;
	for (unsigned long il=0; il < results.size(); il++)
		{
		sum+=results[il][0]*results[il][1];
		sum_weights+=results[il][1];
		}

	return(sum/sum_weights);
	}


/*void convert_results_euclidian_distance_to_great_circle_distance(vector <vector <double>> &results)
	{
	for (unsigned long il=0; il < results.size(); il++)
		results[il][0] = euclidian_distance_to_great_circle_distance(results[il][0]);
	}
*/
/*
vector <vector <double>> convert_results_on_sphere(const vector <vector <double>> &results, const vector <vector <double> > &points1, const vector <vector <double> > &points2)
	{
	vector <vector <double>> results_on_sphere;
	for (unsigned long il=0; il < results.size(); il++)
		{
		double euclidian_distance = results[il][0];
		double value = results[il][1];
		size_t ind1 = results[il][2];
		size_t ind2 = results[il][3];

		double great_circle_distance = euclidian_distance_to_great_circle_distance(euclidian_distance);
		double lat1 = points1[ind1][0];
		double lon1 = points1[ind1][1];
		double lat2 = points2[ind2][0];
		double lon2 = points2[ind2][1];

		results_on_sphere.push_back({great_circle_distance, value, lat1, lon1, lat2, lon2 });
		}

	return(results_on_sphere);
	}
*/

double calculate_PAD(const vector <vector <double> > &points1, const vector <vector <double> > &points2, long max_number_of_nonzero_points)
	{
	auto results = calculate_PAD_results(points1,points2, max_number_of_nonzero_points);
	double PAD = calculate_PAD_from_PAD_results(results);
	return(PAD);
	}

