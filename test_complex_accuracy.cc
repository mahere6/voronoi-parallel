/**
 * @file test_complex_accuracy.cc
 * @brief This file contains a function that tests the accuracy of Voronoi cell data
 * against other cell data considered accurate.
 * 
 * The function can be compiled with the command
 * 		g++ -Wall -Wextra -std=c++20 -Wsign-conversion -o test_accuracy test_complex_accuracy.cc -lm -lstdc++
 * @author E. Maher
 * @version 1
 * @date 8-9-2024
 */


#include <stdio.h>  
#include <unistd.h>  
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <sstream>

void printUsage(void);

using namespace std;

int main(int argc, char *argv[]){
	int num_points = -1;
	string correct_filename;
	string filename_to_test;

	/* parse arguments */
	int opt; 
	int exit = 0;
	int user_specified_correct_filename = 0;
	int user_specified_filename_to_test = 0;
	while ((opt = getopt(argc, argv, "n:c:t:h")) != -1){
		switch(opt){
			case 'n':
				num_points = atoi(optarg);
				cout<<"Number of points is "<<num_points<<"\n";
				break;
			case 'c':
				correct_filename = optarg;
				cout<<"Reading correct cells from " << correct_filename << "\n";
				user_specified_correct_filename = 1;
				break;
			case 't':
				filename_to_test = optarg;
				cout << "Testing accuracy of cells from " << filename_to_test << "\n";
				user_specified_filename_to_test = 1;
				break;
			case 'h':
				exit = 1;
				break;
			default:
				cout<<"Unknown option passed\n";
				exit = 1;;
		}
	}

	if (num_points == -1){
		cout<<"User failed to specify number of points.\n";
		exit = 1;
	}

	if (exit){
		printUsage();
		return 1;
	}

	if(!user_specified_correct_filename){	// use default correct filename
		correct_filename = "results_simple_parallel_" + to_string(num_points) + ".txt";
		cout<<"Reading correct cells from " << correct_filename << "\n";
	}

	if(!user_specified_filename_to_test){	// use default filename_to_test
		filename_to_test = "results_complex_parallel_" + to_string(num_points) + ".txt";
		cout << "Testing accuracy of cells from " << filename_to_test << "\n";
	}


	// read in points from correct_filename
	vector<double> x_values;			// contains x values of points
	vector<double> y_values;			// contains y values of points
	vector<double> cells_vertices;		// contains cell vertices
	vector<unsigned int> locations;		// stores the start of vertex data for each cell

	x_values.reserve((size_t)num_points);			// reserve space for data
	y_values.reserve((size_t)num_points);
	cells_vertices.reserve((size_t)num_points*14);
	locations.reserve((size_t)num_points+1);
	locations.emplace_back(0);		// the first vertex for the first cell starts here

	unsigned int number_of_vertices_read{0};

	fstream fp;
	fp.open(correct_filename, ios::in);	// open file
	if (!fp.is_open()){
		cout<<"Error opening " << correct_filename << "\n";
		return 1;
	}

	for (int i = 0; i<num_points; i++){	// loop through number of points
		string one_line;
		getline(fp, one_line);	// read one line from correct_filename

		stringstream stream(one_line);
		string one_word;

		stream >> one_word;	// read in x_value of point
		x_values.emplace_back(stod(one_word));	// store xvalue of point

		stream >> one_word;	// read in y_value of point
		y_values.emplace_back(stod(one_word));	// store yvalue of point

		// read in vertices
		while(stream >> one_word){
			cells_vertices.emplace_back(stod(one_word));
			number_of_vertices_read++;
			//cout<<cells_vertices.back()<<" ";
		}
		//cout<<"\n";

		// done reading vertices, so store location data
		locations.emplace_back(number_of_vertices_read);
	}
	fp.close();

	//cout<<"locations : ";
	for(auto it = locations.cbegin(); it != locations.cend(); ++it){
		//cout<<(*it)<<" ";
	}
	//cout<<"\n";


	// now we have stored all of the correct data
	// we slowly parse through filename_to_test data and see if it is accurate.

	fp.open(filename_to_test, ios::in);
	if (!fp.is_open()){
		cout<<"Error opening " << filename_to_test << "\n";
		return 1;
	}

	double x,y;
	vector<double> current_cell_vertices;
	current_cell_vertices.reserve(20);
	unsigned int number_of_vertices_in_this_cell{0};
	unsigned int total_number_of_cell_mismatches{0};
	unsigned int total_number_of_points_without_matches{0};

	for (int i = 0; i<num_points; i++){	// loop through number of points

		//cout<<"READ CELL ----------------------------------\n";
		string one_line;
		getline(fp, one_line);	// read one line from filename_to_test

		stringstream stream(one_line);
		string one_word;

		stream >> one_word;	// read in x_value of point
		try{
			x = stod(one_word);	// store xvalue of point
		} catch (const invalid_argument& e) {
			cout << "Invalid input: cannot convert to a number. Skipping cell\n" << endl;
			continue;
		} catch (const out_of_range& e) {
			cout << "Input is out of range for a double. Skipping cell\n" << endl;
			continue;
		}

		stream >> one_word;	// read in y_value of point
		try{
			y = stod(one_word);	// store yvalue of point
		} catch (const invalid_argument& e) {
			cout << "Invalid input: cannot convert to a number. Skipping cell\n" << endl;
			continue;
		} catch (const out_of_range& e) {
			cout << "Input is out of range for a double. Skipping cell\n" << endl;
			continue;
		}
		//cout<<"POINT " << x << " " << y << "\nVERTICES ";

		// read in vertices
		while(stream >> one_word){
			try{
				current_cell_vertices.emplace_back(stod(one_word));
			} catch (const invalid_argument& e) {
				cout << "Invalid input: cannot convert to a number. Skipping cell\n" << endl;
				continue;
			} catch (const out_of_range& e) {
				cout << "Input is out of range for a double. Skipping cell\n" << endl;
				continue;
			}
			number_of_vertices_in_this_cell++;		// count number of vertices in this cell
			//cout<<current_cell_vertices.back() << " ";
		}
		//cout<<"\n";

		// done reading vertices; now we must check the accuracy of this cell.
		// we loop through x_value until we find a match
		bool did_we_ever_find_a_point_match = false;
		for (size_t j = 0; j < (size_t)num_points; j++){
			if (x_values[j] == x){	// we have found a matching x value
				if (y_values[j] == y){	// we have found the matching point
					did_we_ever_find_a_point_match = true;
					if ((locations[j+1] - locations[j]) == number_of_vertices_in_this_cell){	// if the number of vertices in the correct cell is equal to the number of vertices in the cell to test
						// we must loop through vertices and check that they match
						bool mismatch = false;	// have we found a vertex mismatch?
						for (size_t k = 0; k < number_of_vertices_in_this_cell; k++){
							if (current_cell_vertices[k] != cells_vertices[locations[j]+k]){	// vertex mismatch
								cout<<"Vertex mismatch ! compared " << current_cell_vertices[k] << " with " << cells_vertices[locations[j+k]] << "\n";
								mismatch = true;		// we have found a vertex mismatch
								break;
							}
						}
						if (mismatch){
							total_number_of_cell_mismatches++;	// increment the total number of mismatches found
						}

					} else {	// mismatch number of vertices.... yikes
						total_number_of_cell_mismatches++;	// increment the total number of mismatches found
						cout<<"Number of vertices mismatch !\n";
					}
					break;	// we are done testing this cell
				}
			}
		}

		if(!did_we_ever_find_a_point_match){	// if we never found a matching point
			total_number_of_points_without_matches++;
		}
		


		// we are done with this cell so we move onto the next one
		current_cell_vertices.clear();
		number_of_vertices_in_this_cell = 0;
	}
	fp.close();

	

	// print results:
	cout<<"\n\nProgram complete\n";
	cout<<"Total number of points from " << filename_to_test << " that we couldn't find in " << correct_filename << "\n\t"<< total_number_of_points_without_matches << "\n\n";
	cout<<"Total number of cell mismatches: " << total_number_of_cell_mismatches;
	if (total_number_of_cell_mismatches > 20) cout<<"\t Be careful to ensure that boundaries used for both simple and complex programs were the same!";
	cout<<"\n";



	return 0;

}


void printUsage(void){

	cout<<"This program tests the accuracy of results generated from v_parallel_complex against the results generated from v_parallel_simple.\n";
	cout<<"Usage: ./test_complex [options]\n";

	cout<<"    -n value      : specifies the number of points used\n";
	cout<<"    -c filename   : specifies the name of the file with accurate cell data\n";
	cout<<"    -t filename   : specifies the name of the file with cell data to be tested\n";
	cout<<"    -h            : will print this usage statement\n\n";

	cout<<"The number of points must be specified.\n";
	cout<<"If the flag -c is not passed, then the default filename\n";
	cout<<"    results_simple_parallel_[value].txt\n";
	cout<<"will be used.\n";
	cout<<"If the flag -t is not passed, then the default filename\n";
	cout<<"    results_complex_parallel_[value].txt\n";
	cout<<"will be used.\n";

}
