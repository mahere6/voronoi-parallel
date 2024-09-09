/**
 * @file complex_parallel_voronoi.cc
 * @brief An implementation of Daniel Reem's Projector Algorithm. See
 * Daniel Reem. The projector algorithm: A simple parallel algorithm for computing voronoi diagrams and delaunay graphs.
 * Theoretical Computer Science, 970:114054, 2023. 
 * ISSN = 0304-3975
 * DOI = {https://doi.org/10.1016/j.tcs.2023.114054}
 * URL = {https://www.sciencedirect.com/science/article/pii/S0304397523003675}
 * 
 * This version can be used for timing purposes.
 * 
 * This parallel implementation simply divides the input region ("the world") into a user-specified
 * grid or rectangles, assigning one process to each section of the world. The processed work
 * together to approximate the Voronoi diagram of input points specified in the world.
 * 
 * This file can be compiled with
 *   mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o v_parallel_complex complex_parallel_voronoi.cc -lm -lstdc++
 * along with the header file voronoi.h
 * @author E. Maher
 * @version 2.0
 * @date 28-07-2024
 */

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include "voronoi.h"
#include <sstream>


/* Function Declarations */
int		parseArguments			(int argc, char * argv[], int rank, int size, char ** points_filename, double * spacing, int * xdim, int * ydim);
void	printUsage				(void);
void	printFileInfo			(void);

int		in_region				(point pt, std::vector<line>& boundary_lines);

unsigned int	send1			(double ** backup_buffers, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid);
void	send2					(std::vector<double *>& backup_buffers, std::vector<unsigned int>& backup_index, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid);
void	send3					(std::vector<double *>& backup_buffers, std::vector<unsigned int>& backup_index, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid, bool * we_send_to_original_copy);
void	send4					(std::vector<double *>& backup_buffers, std::vector<unsigned int>& backup_index, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid);

int		print_cells_to_file								(std::string filename, int rank, int size, std::vector<cell_info>& cells, MPI_Comm comm);

int				find_Endpoint_and_Line					(std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines, bool * send_info, unsigned int num_sides_to_send_to);
unsigned int	find_Intersection_with_Boundary			(const point p, ordered_vector& direction, std::vector<line>& boundary_lines);

void	expand_boundary_lines						(std::vector<line>& boundary_lines, double * min, double * max, double spacing);
void	compute_remaining_cells						(std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve, std::vector<cell_info>& mycells, std::vector<unsigned int>& giveup_indices, double * giveups_time, double * recompute_time);
void	find_Endpoint_and_Line_advanced				(point p, double * p_address, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve);
void	find_Intersection_with_Boundary_advanced	(const point p, ordered_vector& direction, std::vector<line>& boundary_lines);
void	recompute_some_of_mypoints					(std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve, std::vector<cell_info>& mycells, std::vector<int>& mypoints_to_recalculate);
void	find_Endpoint_and_Line_recomputation		(point p, std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve);

int		test_complex_accuracy						(int number_of_points, std::string correct_filename, std::string filename_to_test);


int print_rank = -1;
bool test_accuracy = false;
char * test_accuracy_filename{nullptr};
int number_of_points{0};


int main(int argc, char *argv[]){
	int rank, size;
	MPI_Comm cartesianCOMM;
	std::vector<int> my_neighbours(8);		// left neighbour, right neighbour, neighbour below, neighbour above, bottom+left corner neighbour, bottom+right corner neighbour, top+left neighbour, top+right neighbour
		// neighbour order:
		//
		//		  6 ______3_____  7
		//			|			|
		//		  0 |	 me 	| 1
		//			|			|
		//			|___________|
		//		  4		  2		  5

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// timing variables
	double st_total1, et_total1, st_total2, et_total2, st_readingpoints1, et_readingpoints1, st_readingpoints2, et_readingpoints2, st_cells1, et_cells1;
	double st_wait, et_wait, st_send, et_send, giveups_time, recompute_time;


	st_total1 = MPI_Wtime();

	// read user input from command line
	char * points_filename{nullptr};		// filename to read input points from
	double spacing{2.0};					// spacing size for default boundary lines
	std::vector<line> boundary_lines;		// vector to populate with boundary halfplanes
	int dims[] = {0, 0};					// we have a cartesian grid of num_procs_x by num_procs_y processors


	// parse arguments
	if (parseArguments(argc, argv, rank, size, &points_filename, &spacing, &dims[0], &dims[1]) == -1){
		return 0;
	}

	// form cartesian grid layout
	int periodic[] = {0, 0};					// grid will not be periodic in either dimension
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &cartesianCOMM);
	int myid;
	MPI_Comm_rank(cartesianCOMM, &myid);

	// find neighbours
	MPI_Cart_shift(cartesianCOMM, 0, 1, &my_neighbours[0], &my_neighbours[1]);		// horizontal neighbours
	MPI_Cart_shift(cartesianCOMM, 1, 1, &my_neighbours[2], &my_neighbours[3]);		// vertical neighbours

	// assume that we don't have corner neighbours
	my_neighbours[4] = my_neighbours[5] = my_neighbours[6] = my_neighbours[7] = -2;
	int my_coordinates[2];		// find my coordinates for use later in determining corner neighour processes
	MPI_Cart_coords(cartesianCOMM, myid, 2, my_coordinates);

	// we determine the range of input points
	std::ifstream fp;	// open file to read input points
	fp.open(points_filename, std::ifstream::in);
	if (!fp.is_open()){
		std::cerr<<"Rank "<<myid<<" failed to open input points file\n";
		MPI_Abort(cartesianCOMM, myid);
		return -1;
	}
	size_t num_points;
	fp>>num_points;		// read number of input points
	number_of_points = (int) num_points;
	int start, end, nlocal, deficit;	// this process will read points in range [start, end) and determine the local min and max x and y values
	nlocal = (int) num_points / size;
	deficit = (int) num_points % size;
	start = rank * nlocal + ((rank < deficit) ? rank : deficit);
	if (rank < deficit) nlocal++;
	end = start + nlocal;


	double value;
	double max[2];		// store max x and y values of input points
	double min[2];		// store min x and y values of input points

	st_readingpoints1 = MPI_Wtime();
	// first discard intial values
	for (int i = 0; i < start; i++){
		fp >> value;	// discard x value
		fp >> value;	// discard y value
	}
	// now read initial point
	fp >> max[0];
	fp >> max[1];
	min[0] = max[0];
	min[1] = max[1];
	// now compare with rest: determine local extreme x and y values
	for (int i = start + 1; i < end; i++){
		fp >> value;
		if (value < min[0]) min[0] = value;
		else if (value > max[0]) max[0] = value;
		fp >> value;
		if (value < min[1]) min[1] = value;
		else if (value > max[1]) max[1] = value;
	}
	fp.close();
	et_readingpoints1 = MPI_Wtime();

	MPI_Allreduce(MPI_IN_PLACE, &max, 2, MPI_DOUBLE, MPI_MAX, cartesianCOMM);	// reduce maximum x and y values
	MPI_Allreduce(MPI_IN_PLACE, &min, 2, MPI_DOUBLE, MPI_MIN, cartesianCOMM);	// reduce minimum x and y values

	/*		At this point, all processes know the range of the input points.		*/
	if(myid == print_rank) std::cout << "Range of all points: " << min[0] << " <= x <= " << max[0] << ", " << min[1] << " <= y <= " << max[1] << std::endl;

	// grid decomposition
	double local_x_range = (max[0] - min[0]) / dims[0];				// local x range size for this process
	double local_y_range = (max[1] - min[1]) / dims[1];				// local y range size for this process
	double x_start = local_x_range * (myid / dims[1]) + min[0];		// min x value for this process
	double x_end = x_start + local_x_range;							// max x value for this process
	double y_start = local_y_range * (myid % dims[1]) + min[1];		// min y value for this process
	double y_end = y_start + local_y_range;							// max y value for this process

	if(myid == print_rank) std::cout << "Rank " << myid << ": local point range is " << x_start << " <= x <= " << x_end << ", " << y_start << " <= y <= " << y_end << std::endl;


	// ---------------------------------------------------------------------------------
	// -------------------------	ADDING BOUNDARY LINES	----------------------------
	// ---------------------------------------------------------------------------------
	boundary_lines.reserve(4);	// boundary lines will be: left; right; bottom; top
	bool * we_send_to = (bool *) malloc(8*sizeof(bool));	// do we send to left? right? bottom? top? bottom+left corner? bottom+right corner? top+left corner? top+right corner?
	if (we_send_to == nullptr){		// error checking
		std::cerr<<"Error allocating memory for bool we_send_to[4]\n";
		MPI_Abort(cartesianCOMM, myid);
	}
	we_send_to[0] = we_send_to[1] = we_send_to[2] = we_send_to[3] = we_send_to[4] = we_send_to[5] = we_send_to[6] = we_send_to[7] = true;	// assume we send to all processes
	unsigned int num_sides_to_send_to = 4;
	
	// if we lie on leftmost boundary:
	if (myid / dims[1] == 0){
		// og left boundary
		we_send_to[0] = false;		// we don't send to these processes
		we_send_to[4] = false;		// ^^^
		we_send_to[6] = false;		// ^^^
		num_sides_to_send_to--;
		boundary_lines.emplace_back(-1,0, spacing-min[0]);		// add boundary line: x >= (min x value) - (requested boundary space)
	} else {	// not on left-most boundary
		boundary_lines.emplace_back(-1, 0, (0.005 * local_x_range) - x_start);		// add boundary line: x >= (x_start - 0.05 * local_x_range)
	}

	// if we lie on rightmost boundary:
	if (myid / dims[1] == dims[0] - 1){
		// og right boundary
		we_send_to[1] = false;		// we don't send to these processes
		we_send_to[5] = false;		// ^^^
		we_send_to[7] = false;		// ^^^
		num_sides_to_send_to--;
		boundary_lines.emplace_back(1,0, max[0]+spacing);		// add boundary line: x <= (max x value) + (requested boundary space)

		x_end = x_end + 1;		// needed later: make sure we actually read in max x value
	} else {	// not on rightmost boundary
		boundary_lines.emplace_back(1, 0, (x_end + 0.005 * local_x_range));		// add boundary line: x <= (x_end + 0.05 * local_x_range)
	}

	// if we lie on bottommost boundary
	if (myid % dims[1] == 0){
		// og bottom boundary
		we_send_to[2] = false;		// we don't send to these processes
		we_send_to[4] = false;		// ^^^
		we_send_to[5] = false;		// ^^^
		num_sides_to_send_to--;
		boundary_lines.emplace_back(0, -1, spacing - min[1]);		// add boundary line: y >= (min y value) - (requested boundary space)
	} else {	// not on bottommost boundary
		boundary_lines.emplace_back(0, -1, (0.005 * local_y_range)-y_start);		// add boundary line: y >= (y_start - 0.05 * local_y_range)

	}

	// if we lie on topmost boundary
	if (myid % dims[1] == dims[1] - 1){
		// og top boundary
		we_send_to[3] = false;		// we don't send to these processes
		we_send_to[6] = false;		// ^^^
		we_send_to[7] = false;		// ^^^
		num_sides_to_send_to--;
		boundary_lines.emplace_back(0, 1, max[1]+spacing);		// add boundary line: y <= (max y value) + (requested boundary space)

		y_end = y_end+1;		// needed later: make sure we actually read in max y value
	} else {	// not on topmost boundary
		boundary_lines.emplace_back(0, 1, (y_end + 0.005 * local_y_range));		// y <= (y_end + 0.05 * local_y_range)
	}

	// make a copy of we_send_to: this is because we_send_to will be modified later!
	bool * we_send_to_original_copy = (bool *) malloc(8*sizeof(bool));
	if (we_send_to_original_copy == nullptr){
		std::cerr<<"Error allocating memory for bool we_send_to_original_copy[4]\n";
		MPI_Abort(cartesianCOMM, myid);
	}
	std::memcpy(we_send_to_original_copy, we_send_to, 8*sizeof(bool));


	// We will need to know how many points to recieve from other processes later.
	// So we begin with learning how many giveups and backups to recieve. 
	//		"Giveups" are points that other processes give us with the responsibility to determine its cell.
	//		"Backups" are points that other processes give us that are needed to determine the cell for a "giveup", but
	//			we are not responsible for determining the cell of a "backup".
	std::vector<int> number_of_giveups_to_recieve(8, 0);	// number of giveups to receive from each process in the standard order
	std::vector<int> number_of_backups_to_recieve(8, 0);	// number of backups to receive from each process in the standard order
	std::vector<MPI_Request> did_I_recv_num_of_giveups(4);	// for the first four neighbours
	std::vector<MPI_Request> did_I_recv_num_of_backups(8);	// for all neighbours
	// we initiate recieves from these processes to learn how many giveups and backups to recieve later:
	if (we_send_to[0]) {
		MPI_Irecv(&number_of_giveups_to_recieve[0], 1, MPI_INT, my_neighbours[0], 0, cartesianCOMM, &did_I_recv_num_of_giveups[0]);		// left neighbour
		MPI_Irecv(&number_of_backups_to_recieve[0], 1, MPI_INT, my_neighbours[0], 2, cartesianCOMM, &did_I_recv_num_of_backups[0]);
	}
	if (we_send_to[1]) {
		MPI_Irecv(&number_of_giveups_to_recieve[1], 1, MPI_INT, my_neighbours[1], 0, cartesianCOMM, &did_I_recv_num_of_giveups[1]);		// right neighbour
		MPI_Irecv(&number_of_backups_to_recieve[1], 1, MPI_INT, my_neighbours[1], 2, cartesianCOMM, &did_I_recv_num_of_backups[1]);
	}
	if (we_send_to[2]) {
		MPI_Irecv(&number_of_giveups_to_recieve[2], 1, MPI_INT, my_neighbours[2], 0, cartesianCOMM, &did_I_recv_num_of_giveups[2]);		// neighbour below
		MPI_Irecv(&number_of_backups_to_recieve[2], 1, MPI_INT, my_neighbours[2], 2, cartesianCOMM, &did_I_recv_num_of_backups[2]);
	}
	if (we_send_to[3]) {
		MPI_Irecv(&number_of_giveups_to_recieve[3], 1, MPI_INT, my_neighbours[3], 0, cartesianCOMM, &did_I_recv_num_of_giveups[3]);		// neighbour above
		MPI_Irecv(&number_of_backups_to_recieve[3], 1, MPI_INT, my_neighbours[3], 2, cartesianCOMM, &did_I_recv_num_of_backups[3]);
	}

	// we know that we will recieve only one corner giveup from corner neighbours, so initiate those recieves:
	double recieved_corner_giveups[] = {0, 0, 0, 0, 0, 0, 0, 0};		// corner giveups that we recieve
	double corner_giveups[] = {0, 0, 0, 0, 0, 0, 0, 0};					// our corner giveups that we will send
	std::vector<int> corner_processes_to_send_to;						// record the value in [0,7] corresponding to the corner processes we communicate with
	std::vector<MPI_Request> did_I_recv_giveups(8);
	std::vector<MPI_Request> did_they_recv_corner_giveups;
	unsigned int number_of_corner_giveups{0};

	if (we_send_to[4]) {	// if we send to bottom+left corner
		int diag_neighbour_coordinates[] = {my_coordinates[0]-1, my_coordinates[1]-1};		// determine which process this neighbour is
		MPI_Cart_rank(cartesianCOMM, diag_neighbour_coordinates, &my_neighbours[4]);		// ^^^

		MPI_Irecv(&recieved_corner_giveups[0], 2, MPI_DOUBLE, my_neighbours[4], 1, cartesianCOMM, &did_I_recv_giveups[4]);
		MPI_Irecv(&number_of_backups_to_recieve[4], 1, MPI_INT, my_neighbours[4], 2, cartesianCOMM, &did_I_recv_num_of_backups[4]);
		corner_processes_to_send_to.emplace_back(my_neighbours[4]);
		number_of_corner_giveups++;
	}
	if (we_send_to[5]) {	// if we send to bottom+right corner
		int diag_neighbour_coordinates[] = {my_coordinates[0]+1, my_coordinates[1]-1};		// determine which process this neighbour is
		MPI_Cart_rank(cartesianCOMM, diag_neighbour_coordinates, &my_neighbours[5]);		// ^^^

		MPI_Irecv(&recieved_corner_giveups[2], 2, MPI_DOUBLE, my_neighbours[5], 1, cartesianCOMM, &did_I_recv_giveups[5]);
		MPI_Irecv(&number_of_backups_to_recieve[5], 1, MPI_INT, my_neighbours[5], 2, cartesianCOMM, &did_I_recv_num_of_backups[5]);
		corner_processes_to_send_to.emplace_back(my_neighbours[5]);
		number_of_corner_giveups++;
	}
	if (we_send_to[6]) {	// if we send to upper+left corner
		int diag_neighbour_coordinates[] = {my_coordinates[0]-1, my_coordinates[1]+1};		// determine which process this neighbour is
		MPI_Cart_rank(cartesianCOMM, diag_neighbour_coordinates, &my_neighbours[6]);		// ^^^

		MPI_Irecv(&recieved_corner_giveups[4], 2, MPI_DOUBLE, my_neighbours[6], 1, cartesianCOMM, &did_I_recv_giveups[6]);
		MPI_Irecv(&number_of_backups_to_recieve[6], 1, MPI_INT, my_neighbours[6], 2, cartesianCOMM, &did_I_recv_num_of_backups[6]);
		corner_processes_to_send_to.emplace_back(my_neighbours[6]);
		number_of_corner_giveups++;
	}
	if (we_send_to[7]) {	// if we send to upper+right corner
		int diag_neighbour_coordinates[] = {my_coordinates[0]+1, my_coordinates[1]+1};		// determine which process this neighbour is
		MPI_Cart_rank(cartesianCOMM, diag_neighbour_coordinates, &my_neighbours[7]);		// ^^^

		MPI_Irecv(&recieved_corner_giveups[6], 2, MPI_DOUBLE, my_neighbours[7], 1, cartesianCOMM, &did_I_recv_giveups[7]);
		MPI_Irecv(&number_of_backups_to_recieve[7], 1, MPI_INT, my_neighbours[7], 2, cartesianCOMM, &did_I_recv_num_of_backups[7]);
		corner_processes_to_send_to.emplace_back(my_neighbours[7]);
		number_of_corner_giveups++;
	}
	did_they_recv_corner_giveups.resize(number_of_corner_giveups);


	// -----------------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------	READING IN POINTS	------------------------------------------------
	// -----------------------------------------------------------------------------------------------------------------------------

	// each process reads in points (x, y) which satisfy 
	//			x_start =< x < x_end
	//			y_start <= y < y_end


	std::vector<point> mypoints;		// vector of our points that we will be responsible for
	mypoints.reserve((size_t)((num_points / (size_t)size)*1.1));	// estimate how many points we read in

	st_readingpoints2 = MPI_Wtime();

	fp.open(points_filename, std::ifstream::in);
	if (!fp.is_open()){
		std::cerr<<"Rank "<<myid<<" failed to open input points file\n";
		MPI_Abort(cartesianCOMM, myid);
		return -1;
	}

	fp>>value;	// discard the first value ( the number of points, we already know )
	mypoints.emplace_back(0,0);

	// now read in the points that are in our domain
	for (size_t i = 0; i < num_points - 1; i++){
		fp >> mypoints.back().x; // value contains x value of point
		fp >> mypoints.back().y; // value contains y value of point
		if (mypoints.back().x < x_start || mypoints.back().x >= x_end){	// not in x range
			continue;	// next iteration
		} else if (mypoints.back().y < y_start || mypoints.back().y >= y_end){	// not in y range
			continue;	// next iteration
		} else {	// we are in range
			mypoints.emplace_back(0,0);
		}
	}
	// read in the last point: 
	fp >> mypoints.back().x; // value contains x value of point
	fp >> mypoints.back().y; // value contains y value of point
	if (mypoints.back().x < x_start || mypoints.back().x >= x_end){	// not in x range
		mypoints.pop_back();
	} else if (mypoints.back().y < y_start || mypoints.back().y >= y_end){	// not in y range
		mypoints.pop_back();
	} // else we are in range and we don't need to do anything

	fp.close();

	et_readingpoints2 = MPI_Wtime();

	// now mypoints is populated with our region of points

	et_total1 = MPI_Wtime();
	// we check that each process has sufficiently many points in its domain to ensure code functions
	MPI_Barrier(MPI_COMM_WORLD);
	{
		// we will ask user if they wish to continue if a process has less than 100 points
		int should_we_ask_user_to_proceed{0};
		int does_any_process_have_insufficient_points{0};
		if (mypoints.size() < 100 ){
			should_we_ask_user_to_proceed = 1;
		}
		MPI_Reduce(&should_we_ask_user_to_proceed, &does_any_process_have_insufficient_points, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0){
			while (does_any_process_have_insufficient_points){
				std::cout << "\n\nAt least one process has less than 100 points in its local grid section. Errors may result.\nDo you wish to proceed? (y/n)  " << std::endl;
				char c;
				std::cin >> c;
				if (c == 'n') {
					MPI_Abort(cartesianCOMM, rank);
				} else if (c != 'y'){
					std::cout << "\nUnknown input. Please type 'n' or 'y' for yes/no.\n";
				} else break;

			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	st_total2 = MPI_Wtime();
	

	// we estimate the number of giveups to send each way
	int estimated_number_of_giveups{50};	// Estimate the number of giveups we will be sending in each direction
	std::vector<double *> giveups;		// allocate memory to store giveups that we will send
	std::vector<size_t> allocated_space_for_giveups;	// keep track of the amount of space we have allocated for giveups
	size_t reallocation_difference_for_giveups = 50;	// the number of giveups to increase by if capacity is reached
	for (int i = 0; i<4; i++){
		if (we_send_to[i]){
			giveups.emplace_back((double *) malloc((size_t) estimated_number_of_giveups * 2 * sizeof(double)));
			allocated_space_for_giveups.emplace_back((size_t)estimated_number_of_giveups);
			if (giveups.back() == nullptr){
				std::cerr << "Error allocating memory for giveups\n";
				MPI_Abort(cartesianCOMM, myid);
			}
		}
	}

	std::vector<cell_info> mycells;		//	vector to hold cell data
	mycells.reserve(mypoints.size());

	// -----------------------------------------------------------------------------------------------------------------------
	// ---------------------------------	BEGIN INITIAL LOCAL VORONOI DIAGRAM COMPUTATION		------------------------------
	// -----------------------------------------------------------------------------------------------------------------------

	// we re-arrange background lines, so that the lines which are not final boundary lines come first
	{
		unsigned int i=0;
		while(i<3){
			if (we_send_to[i] == false){	// if we don't send through this boundary
				unsigned int j = i+1;
				while (j<4){
					if (we_send_to[j] == true){		// we swap with the next one that we do
						std::swap(boundary_lines[i], boundary_lines[j]);
						we_send_to[j] = false;
						we_send_to[i] = true;
						break;
					}
					j++;
				}
			}
			i++;
		}
	}


	// we will loop through input points, but first we need
	unsigned int pt_id = 0;	// index of the current point
	std::vector<unsigned int> giveup_indices;		// record giveup indices for use later (when re-evaluating the mypoints cells)
	giveup_indices.reserve((size_t)estimated_number_of_giveups); 	// estimate how many giveups we will have total
	unsigned int add_id = 0;
	std::vector<size_t> counters(4, 0);	// number of giveups counted (left, right, bottom, up)

	unsigned int sinfo_size = num_sides_to_send_to*2;
	bool * send_info = (bool*) malloc(sinfo_size * mypoints.size() * sizeof(bool));		// records information for each of mypoints: whether the point is a particular giveup or backup
	if (send_info == nullptr){		// error checking
		std::cerr << "Error allocating memory for send_info (bool matrix)\n";
		MPI_Abort(cartesianCOMM, myid);
	}
	// initialize sendinfo[] with zeros
	for (unsigned int i = 0; i < sinfo_size * mypoints.size(); i++){
		send_info[i] = false;
	}
	
	st_cells1 = MPI_Wtime();
	//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~		LOOPING THROUGH MYPOINTS	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for (auto point_it = mypoints.begin(); point_it != mypoints.end(); ++point_it){
		std::vector<int> neighbours;		// will store the indices of the cell's neighbouring points
		neighbours.reserve(8);				// estimate that there are 8
		mycells.emplace_back(*point_it);	// new cell
		point p{*point_it};


		std::vector<ordered_vector> rays(4);	// this will hold the rays that we "shoot" from p
		rays.reserve(20);
		rays[0] = ordered_vector{0,1, 1};
		rays[1] = ordered_vector{1,0, 2};
		rays[2] = ordered_vector{0,-1, 3};
		rays[3] = ordered_vector{-1,0, 4};

		// Create a queue of edges to cycle through
		std::vector<subwedge> WedgeQueue(4);
		WedgeQueue.reserve(20);
		WedgeQueue[0] = subwedge{&rays[3], &rays[0]};	// order is counterclockwise starting at noon
		WedgeQueue[1] = subwedge{&rays[2], &rays[3]};
		WedgeQueue[2] = subwedge{&rays[1], &rays[2]};
		WedgeQueue[3] = subwedge{&rays[0], &rays[1]};


		size_t edge_spot = 0;		// loop through remaining edges; edge_spot records where we are in WedgeQueue
		while(WedgeQueue.begin() + (long)edge_spot != WedgeQueue.end()){
			// find endpoints and bisecting lines
			find_Endpoint_and_Line(point_it, *(WedgeQueue[edge_spot]).vector1, mypoints, boundary_lines, &send_info[sinfo_size * pt_id], num_sides_to_send_to);	// take neighbour from this guy
			find_Endpoint_and_Line(point_it, *(WedgeQueue[edge_spot]).vector2, mypoints, boundary_lines, &send_info[sinfo_size * pt_id], num_sides_to_send_to);	// but compute both !

			// find determinant of matrix B
			double a = ((WedgeQueue[edge_spot]).vector1->boundary)*(((WedgeQueue[edge_spot]).vector1->t)*(*(WedgeQueue[edge_spot]).vector1));
			double b = ((WedgeQueue[edge_spot]).vector1->boundary)*(((WedgeQueue[edge_spot]).vector2->t)*(*(WedgeQueue[edge_spot]).vector2));
			double c = ((WedgeQueue[edge_spot]).vector2->boundary)*(((WedgeQueue[edge_spot]).vector1->t)*(*(WedgeQueue[edge_spot]).vector1));
			double d = ((WedgeQueue[edge_spot]).vector2->boundary)*(((WedgeQueue[edge_spot]).vector2->t)*(*(WedgeQueue[edge_spot]).vector2));
			double determinant = (a*d)-(b*c);

			// CASE 1: infinitely many solutions or no solutions
			if (determinant*determinant < 1e-26){ 
				if (	((WedgeQueue[edge_spot]).vector1->boundary.ycoeff) == 0 		){	// if vertical lines
					// if infinitely many solutions (i.e. same line):
					if (	(((WedgeQueue[edge_spot]).vector1->boundary.c) / ((WedgeQueue[edge_spot]).vector1->boundary.xcoeff )) == (((WedgeQueue[edge_spot]).vector2->boundary.c) / ((WedgeQueue[edge_spot]).vector2->boundary.xcoeff))	){
						// same line, hence no vertices in this subwedge, hence we remove this edge from WedgeQueue
						edge_spot++;
					} else { // no solutions: lines are parallel but not the same
						// split the edge in half
						// new ray is
						// double r1 = 0;
						double r2 = 1;

						// check that the ray is in the wedge:
						double v1, v2, v3,  v4, alpha1, alpha2, det;
						v1 = (WedgeQueue[edge_spot]).vector1->x;
						v2 = (WedgeQueue[edge_spot]).vector1->y;
						v3 = (WedgeQueue[edge_spot]).vector2->x;
						v4 = (WedgeQueue[edge_spot]).vector2->y;
						det = v1*v4-v3*v2;
						alpha1 = (r2*v2)/det;
						alpha2 = (v1*r2)/det;
						if ((alpha1 < 0) || (alpha2 < 0)){		// if ray is not in the wedge
							r2 = -1;
						}

						// now we have our new ray (r1,r2)
						// add a new subedge to represent second half of this edge
						double idx;
						if (((WedgeQueue[edge_spot]).vector2->index) != 1){
							idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
						} else {
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
						}
						rays.emplace_back(0,r2, idx);
						WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);
						// then adjust this current edge to represent first hald of this edge
						((WedgeQueue[edge_spot]).vector2) = &rays.back();

					}

				} else /* not vertical lines */if (	(((WedgeQueue[edge_spot]).vector1->boundary.c) / ((WedgeQueue[edge_spot]).vector1->boundary.ycoeff )) == (((WedgeQueue[edge_spot]).vector2->boundary.c) / ((WedgeQueue[edge_spot]).vector2->boundary.ycoeff))	){
					// same line, hence no vertices in this subwedge, hence we move on from this edge from WedgeQueue
					edge_spot++;
				} else {	// no solutions
					// split the edge in half
					// new ray is
					double r1, r2;
					if (	(WedgeQueue[edge_spot]).vector1->boundary.xcoeff == 0		){
						r1 = 1;
						r2 = 0;
					} else {
						r1 = ((WedgeQueue[edge_spot]).vector1->boundary.xcoeff);
						r2 = (-1) / ((WedgeQueue[edge_spot]).vector1->boundary.ycoeff);
						// normalize:
						double length = sqrt(r1*r1 + r2*r2);
						r1 /= length;
						r2 /= length;
					}
					// check that the ray is in the wedge:
					double v1, v2, v3,  v4, alpha1, alpha2, det;
					v1 = (WedgeQueue[edge_spot]).vector1->x;
					v2 = (WedgeQueue[edge_spot]).vector1->y;
					v3 = (WedgeQueue[edge_spot]).vector2->x;
					v4 = (WedgeQueue[edge_spot]).vector2->y;
					det = v1*v4-v3*v2;
					alpha1 = (r1*v4 - r2*v2)/det;
					alpha2 = (v1*r2 - v3*r1)/det;
					if ((alpha1 < 0) || (alpha2 < 0)){		// if ray is not in the wedge
						r1 *= -1;
						r2 *= -1;
					}

					// now we have our new ray (r1,r2)
					// add a new subedge to represent second half of this edge
					double idx;
					if (((WedgeQueue[edge_spot]).vector2->index) != 1){
						idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
					} else {
						idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
					}
					rays.emplace_back(r1,r2, idx);
					WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);
					// then adjust this current edge to represent first hald of this edge
					((WedgeQueue[edge_spot]).vector2) = &rays.back();
					

				}
			// CASE 2: unique solution
			} else {
				// determine unique solution:
				double lambda1 = (a*d - b*d)/determinant;
				double lambda2 = (a*d - c*a)/determinant;
				point intersection = p + (lambda1 * ((WedgeQueue[edge_spot]).vector1->t)) * (*(WedgeQueue[edge_spot]).vector1) + (lambda2 * ((WedgeQueue[edge_spot]).vector2->t)) * (*(WedgeQueue[edge_spot]).vector2);

				if ( (lambda1 < 0) || (lambda2 < 0)){	// if unique solution is not in the subwedge
					// we split the subwedge
					double idx;
					if (((WedgeQueue[edge_spot]).vector2->index) != 1){
						idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
					} else {
						idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
					}
					rays.emplace_back((p - intersection) / p.distance(intersection), idx);

					// add a new subedge to represent second half of this edge
					WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

					// then adjust this current edge to represent first half of this edge
					(WedgeQueue[edge_spot]).vector2 = &rays.back();


				} else {	// the unique solution IS in the subwedge
					// now we determine whether the unique solution is in the cell or not
					bool in_cell = true;
					double min_distance = p.distance(intersection);		// distance from p to intersection
					for (auto it = mypoints.begin(); it != mypoints.end(); ++it){		// loop through input points and find closest input point to endpoint and corresponding distance
						double check_dist = (*it).distance(intersection);
						if ( check_dist < min_distance){	// if we have found a closer point
							if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
								in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
								break;
							}
						}
					}
					if (in_cell) {	// we have found a vertex
						if (in_region(intersection, boundary_lines)){	// if is in region ("world")...
							// save vertex data
							mycells.back().add_vertex(intersection, (WedgeQueue[edge_spot]).vector1->index, (WedgeQueue[edge_spot]).vector1->boundary);

							// save neighbour data
							if (((WedgeQueue[edge_spot]).vector1->nbr) != -1){
								neighbours.emplace_back((WedgeQueue[edge_spot]).vector1->nbr);
								add_id++;
							}
							edge_spot++;

						} else {	// otherwise it is not in region ("world")...
							double idx;
							if (((WedgeQueue[edge_spot]).vector2->index) != 1){
								idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
							} else {
								idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
							}
							rays.emplace_back((intersection - p) / p.distance(intersection), idx);

							// add a new subedge to represent second half of this edge
							WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

							// then adjust this current edge to represent first half of this edge
							(WedgeQueue[edge_spot]).vector2 = (&rays.back());
						}

					} else {		// we have not found a vertex and we split
						double idx;
						if (((WedgeQueue[edge_spot]).vector2->index) != 1){
							idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
						} else {
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
						}
						rays.emplace_back((intersection - p) / p.distance(intersection), idx);

						// add a new subedge to represent second half of this edge
						WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

						// then adjust this current edge to represent first half of this edge
						(WedgeQueue[edge_spot]).vector2 = (&rays.back());
					}
				}
			}
		}	// end WedgeQueue loop: done determining cell


		// now we determine if we should save neighbour information for found neighbours
		//		(depends on if the point p lies on a boundary side shared with another processor)
		switch(num_sides_to_send_to){	// (which depends on the number of sides we send information to)
			case 1:	// 1 shared boundary side
			{
				if(!send_info[2*pt_id]){	// not on boundary
					mycells.back().sort_vertices();
					break;
				} else {					// on boundary, so record neighbours as backups
					for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
						send_info[2*(*nbr_it)+1] = true;
					}
					giveups[0][counters[0]*2] = p.x;		// record giveup in an easily MPI_sendable format
					giveups[0][counters[0]*2+1] = p.y;
					giveup_indices.emplace_back(pt_id);		// record giveup index for use later
					counters[0]++;							// count number of giveups
					if (counters[0] == allocated_space_for_giveups[0]){		// if we have hit the limit for allocated giveup space
						allocated_space_for_giveups[0] += reallocation_difference_for_giveups;	// allocate more space
						giveups[0] = (double*)realloc(giveups[0], allocated_space_for_giveups[0] * 2* sizeof(double));
					}
					mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
					break;
				}
			}
			case 2:	// 2 shared boundary sides
			{
				int option = 0 + (send_info[4*pt_id] ? 1 : 0) + (send_info[4*pt_id+1] ? 2 : 0);
				switch(option){				// switch based on which boundary sides the point p lies on
					case 0:					// not on boundary
						mycells.back().sort_vertices();
						break;
					case 1:					// only on first boundary
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[4*(*nbr_it)+2] = true;
						}
						giveups[0][2*counters[0]] = p.x;		// record giveup (to be sent)
						giveups[0][2*counters[0]+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[0]++;							// count number of giveups
						if (counters[0] == allocated_space_for_giveups[0]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[0] += reallocation_difference_for_giveups;	// allocate more space
							giveups[0] = (double*)realloc(giveups[0], allocated_space_for_giveups[0] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 2:					// only on second boundary
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[4*(*nbr_it)+3] = true;
						}
						giveups[1][2*counters[1]] = p.x;		// record giveup (to be sent)
						giveups[1][2*counters[1]+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[1]++;							// count number of giveups
						if (counters[1] == allocated_space_for_giveups[1]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[1] += reallocation_difference_for_giveups;	// allocate more space
							giveups[1] = (double*)realloc(giveups[1], allocated_space_for_giveups[1] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 3:					// on both boundaries (corner)
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[4*(*nbr_it)+2] = true;
							send_info[4*(*nbr_it)+3] = true;
						}
						corner_giveups[0] = p.x;		// record giveup (to be sent)
						corner_giveups[1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						MPI_Isend(&corner_giveups[0], 2, MPI_DOUBLE, corner_processes_to_send_to[0], 1, cartesianCOMM, &did_they_recv_corner_giveups[0]);
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
				}
				break;
			}
			case 3:	// 3 shared boundary sides
			{
				int option = 0 + (send_info[6*pt_id] ? 1 : 0) + (send_info[6*pt_id+1] ? 2 : 0) + (send_info[6*pt_id+2] ? 4 : 0);
				switch(option){				// switch based on which boundary sides the point p lies on
					case 0:		// not on boundary
						mycells.back().sort_vertices();
						break;
					case 1:		// on first boundary side
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[6*(*nbr_it)+3] = true;
						}
						giveups[0][counters[0]*2] = p.x;		// record giveup (to be sent)
						giveups[0][counters[0]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[0]++;							// count number of giveups
						if (counters[0] == allocated_space_for_giveups[0]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[0] += reallocation_difference_for_giveups;	// allocate more space
							giveups[0] = (double*)realloc(giveups[0], allocated_space_for_giveups[0] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 2:		// on second boundary side
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[6*(*nbr_it)+4] = true;
						}
						giveups[1][counters[1]*2] = p.x;		// record giveup (to be sent)
						giveups[1][counters[1]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[1]++;							// count number of giveups
						if (counters[1] == allocated_space_for_giveups[1]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[1] += reallocation_difference_for_giveups;	// allocate more space
							giveups[1] = (double*)realloc(giveups[1], allocated_space_for_giveups[1] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 3:		// on first and second boundary side (corner)
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[6*(*nbr_it)+3] = true;
							send_info[6*(*nbr_it)+4] = true;
						}
						corner_giveups[0] = p.x;		// record giveup (to be sent)
						corner_giveups[1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// determine which process to send corner to:
						if(!we_send_to_original_copy[0]) MPI_Isend(&corner_giveups[0], 2, MPI_DOUBLE, my_neighbours[5], 1, cartesianCOMM, &did_they_recv_corner_giveups[0]);
						else if(!we_send_to_original_copy[1]) MPI_Isend(&corner_giveups[0], 2, MPI_DOUBLE, my_neighbours[4], 1, cartesianCOMM, &did_they_recv_corner_giveups[0]);
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 4:		// on third boundary side
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[6*(*nbr_it)+5] = true;
						}
						giveups[2][counters[2]*2] = p.x;		// record giveup (to be sent)
						giveups[2][counters[2]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[2]++;							// count number of giveups
						if (counters[2] == allocated_space_for_giveups[2]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[2] += reallocation_difference_for_giveups;	// allocate more space
							giveups[2] = (double*)realloc(giveups[2], allocated_space_for_giveups[2] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 5:		// on first and third boundary side (corner)
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[6*(*nbr_it)+3] = true;
							send_info[6*(*nbr_it)+5] = true;
						}
						corner_giveups[2] = p.x;		// record giveup (to be sent)
						corner_giveups[3] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// determine which process to send corner to:
						if(!we_send_to_original_copy[0]) MPI_Isend(&corner_giveups[2], 2, MPI_DOUBLE, my_neighbours[7], 1, cartesianCOMM, &did_they_recv_corner_giveups[1]);
						else if(!we_send_to_original_copy[1]) MPI_Isend(&corner_giveups[2], 2, MPI_DOUBLE, my_neighbours[6], 1, cartesianCOMM, &did_they_recv_corner_giveups[1]);
						else if(!we_send_to_original_copy[2]) MPI_Isend(&corner_giveups[2], 2, MPI_DOUBLE, my_neighbours[6], 1, cartesianCOMM, &did_they_recv_corner_giveups[0]);
						else MPI_Isend(&corner_giveups[2], 2, MPI_DOUBLE, my_neighbours[4], 1, cartesianCOMM, &did_they_recv_corner_giveups[0]);

						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 6:		// on second and third boundary side (corner)
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[6*(*nbr_it)+4] = true;
							send_info[6*(*nbr_it)+5] = true;
						}
						corner_giveups[4] = p.x;		// record giveup (to be sent)
						corner_giveups[5] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// determine which process to send corner to:
						if(!we_send_to_original_copy[2]) MPI_Isend(&corner_giveups[4], 2, MPI_DOUBLE, my_neighbours[7], 1, cartesianCOMM, &did_they_recv_corner_giveups[1]);
						else if(!we_send_to_original_copy[3]) MPI_Isend(&corner_giveups[4], 2, MPI_DOUBLE, my_neighbours[5], 1, cartesianCOMM, &did_they_recv_corner_giveups[1]);

						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
				}
				break;
			}
			case 4:	// 4 shared boundary sides
			{
				int option = 0 + (send_info[8*pt_id] ? 1 : 0) + (send_info[8*pt_id+1] ? 2 : 0) + (send_info[8*pt_id+2] ? 3 : 0) + (send_info[8*pt_id+3] ? 6 : 0);
				
							//		7_____6_____8
							//		|			|
							//		1			2
							//		|			|
							//		|___________|
							//		4	 3		5
			
				switch(option){				// switch based on which boundary sides the point p lies on
					case 0:
						mycells.back().sort_vertices();
						break;
					case 1:		// on left boundary
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+4] = true;
						}
						giveups[0][counters[0]*2] = p.x;		// record giveup (to be sent)
						giveups[0][counters[0]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[0]++;							// count number of giveups
						if (counters[0] == allocated_space_for_giveups[0]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[0] += reallocation_difference_for_giveups;	// allocate more space
							giveups[0] = (double*)realloc(giveups[0], allocated_space_for_giveups[0] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 2:		// on right boundary
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+5] = true;
						}
						giveups[1][counters[1]*2] = p.x;		// record giveup (to be sent)
						giveups[1][counters[1]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[1]++;							// count number of giveups
						if (counters[1] == allocated_space_for_giveups[1]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[1] += reallocation_difference_for_giveups;	// allocate more space
							giveups[1] = (double*)realloc(giveups[1], allocated_space_for_giveups[1] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 3:		// on bottom boundary
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+6] = true;
						}
						giveups[2][counters[2]*2] = p.x;		// record giveup (to be sent)
						giveups[2][counters[2]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[2]++;							// count number of giveups
						if (counters[2] == allocated_space_for_giveups[2]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[2] += reallocation_difference_for_giveups;	// allocate more space
							giveups[2] = (double*)realloc(giveups[2], allocated_space_for_giveups[2] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 4:		// on left+bottom corner
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+4] = true;
							send_info[8*(*nbr_it)+6] = true;
						}
						corner_giveups[0] = p.x;		// record giveup (to be sent)
						corner_giveups[1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// send corner giveup:
						MPI_Isend(&corner_giveups[0], 2, MPI_DOUBLE, my_neighbours[4], 1, cartesianCOMM, &did_they_recv_corner_giveups[0]);
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 5:		// on right+bottom corner
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+5] = true;
							send_info[8*(*nbr_it)+6] = true;
						}
						corner_giveups[2] = p.x;		// record giveup (to be sent)
						corner_giveups[3] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// send corner giveup:
						MPI_Isend(&corner_giveups[2], 2, MPI_DOUBLE, my_neighbours[5], 1, cartesianCOMM, &did_they_recv_corner_giveups[1]);
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 6:		// on top boundary
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+7] = true;
						}
						giveups[3][counters[3]*2] = p.x;		// record giveup (to be sent)
						giveups[3][counters[3]*2+1] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						counters[3]++;							// count number of giveups
						if (counters[3] == allocated_space_for_giveups[3]){		// if we have hit the limit for allocated giveup space
							allocated_space_for_giveups[3] += reallocation_difference_for_giveups;	// allocate more space
							giveups[3] = (double*)realloc(giveups[3], allocated_space_for_giveups[3] * 2* sizeof(double));
						}
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 7:		// on top+left corner
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+4] = true;
							send_info[8*(*nbr_it)+7] = true;
						}
						corner_giveups[4] = p.x;		// record giveup (to be sent)
						corner_giveups[5] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// send corner giveup:
						MPI_Isend(&corner_giveups[4], 2, MPI_DOUBLE, my_neighbours[6], 1, cartesianCOMM, &did_they_recv_corner_giveups[2]);
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
					case 8:		// on top+right corner
					{
						for (auto nbr_it = neighbours.cbegin(); nbr_it != neighbours.cend(); ++nbr_it){
							send_info[8*(*nbr_it)+5] = true;
							send_info[8*(*nbr_it)+7] = true;
						}
						corner_giveups[6] = p.x;		// record giveup (to be sent)
						corner_giveups[7] = p.y;
						giveup_indices.emplace_back(pt_id);		// record giveup index for use later
						// send corner giveup:
						MPI_Isend(&corner_giveups[6], 2, MPI_DOUBLE, my_neighbours[7], 1, cartesianCOMM, &did_they_recv_corner_giveups[3]);
						mycells.back().vertices.resize(0);;		// delete cell information because we are giving up this cell
						break;
					}
				}
				break;
			}
		}

		pt_id++;	// move to the next point

	}	// end inital loop through mypoints

	et_cells1 = MPI_Wtime();

	std::vector<double *> backups;								// our backups that we will send
	std::vector<double *> recieved_giveups;						// giveups that we recieve
	std::vector<double *> recieved_backups;						// backups that we recieve
	std::vector<unsigned int> num_of_backup_points_to_send;				// number of backup points to send to each process that we communicate with
	std::vector<int> num_of_backup_points_to_send_ints;
	std::vector<MPI_Request> did_they_recieve_num_of_giveups;
	std::vector<MPI_Request> did_they_recieve_giveups;
	std::vector<MPI_Request> did_they_recieve_num_of_backups;
	std::vector<MPI_Request> did_they_recieve_backups;
	

	switch(num_sides_to_send_to){
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//	-------------------------------------------------------------			1 SIDE TO SEND TO
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		case 1:
		{
			// who are we recieving from ?
			size_t source_index;
			if (we_send_to_original_copy[0]){			// left
				source_index = 0;
			} else if (we_send_to_original_copy[1]){	// right
				source_index = 1;
			} else if (we_send_to_original_copy[2]){	// bottom
				source_index = 2;
			} else {									// top
				source_index = 3;
			}

			// send number of giveups
			did_they_recieve_num_of_giveups.resize(1);
			MPI_Isend(&counters[0], 1, MPI_INT, my_neighbours[source_index], 0, cartesianCOMM, &did_they_recieve_num_of_giveups.back());

			
			// test if we have recieved the number of giveups to recieve:
			int flag{0};
			int check_again{1};
			MPI_Test(&did_I_recv_num_of_giveups[source_index], &flag, MPI_STATUS_IGNORE);
			if (flag){
				// initate giveup recv
				recieved_giveups.emplace_back((double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_index] * 2 * sizeof(double)));
				if (recieved_giveups.back() == nullptr){	// error checking
					std::cerr << "Error allocating memory for recieved_giveups[0]\n";
					MPI_Abort(cartesianCOMM, myid);
				}
				check_again = 0;
				MPI_Irecv(recieved_giveups.back(), number_of_giveups_to_recieve[source_index] * 2, MPI_DOUBLE, my_neighbours[source_index], 1, cartesianCOMM, &did_I_recv_giveups[source_index]);
			}

			
			// send giveups
			did_they_recieve_giveups.resize(1);
			MPI_Isend(giveups[0], counters[0]*2, MPI_DOUBLE, my_neighbours[source_index], 1, cartesianCOMM, &did_they_recieve_giveups.back());
			

			// now arrange backup data to send
			backups.resize(1);


			// Call send1: sorting backups data into a buffer to be sent
			num_of_backup_points_to_send.resize(1, 0);
			num_of_backup_points_to_send_ints.resize(1, 0);
			st_send = MPI_Wtime();
			num_of_backup_points_to_send[0] = send1(&backups.back(), send_info, mypoints, counters, myid);
			et_send = MPI_Wtime();
			if (myid == print_rank) std::cout << "Rank " << myid <<": number of backup points to send is " << num_of_backup_points_to_send[0] << "\n";
			num_of_backup_points_to_send_ints[0] = (int)num_of_backup_points_to_send[0];
			
			// send the number of backup points to be sent later
			did_they_recieve_num_of_backups.resize(1);
			MPI_Isend(&num_of_backup_points_to_send_ints[0], 1, MPI_INT, my_neighbours[source_index], 2, cartesianCOMM, &did_they_recieve_num_of_backups.back());

			
			// send the backup points
			did_they_recieve_backups.resize(1);
			MPI_Isend(backups.back(), num_of_backup_points_to_send_ints[0] * 2, MPI_DOUBLE, my_neighbours[source_index], 3, cartesianCOMM, &did_they_recieve_backups.back());

			
			// test whether we have recieved the number of backups to recieve
			int flag2{0};
			int check_again2{1};
			MPI_Test(&did_I_recv_num_of_backups[source_index], &flag2, MPI_STATUS_IGNORE);
			MPI_Request did_I_recv_backups;
			if (flag2){
				// initiate backup recv
				recieved_backups.emplace_back((double *) malloc((long unsigned int) number_of_backups_to_recieve[source_index] * 2 * sizeof(double)));
				if (recieved_backups.back() == nullptr){
					std::cerr << "Error allocating memory for recieved_backups[0]\n";
					MPI_Abort(cartesianCOMM, myid);
				}
				check_again2 = 0;
				MPI_Irecv(recieved_backups.back(), number_of_backups_to_recieve[source_index] * 2, MPI_DOUBLE, my_neighbours[source_index], 3, cartesianCOMM, &did_I_recv_backups);
			}

			// wait on recv of number of giveups
			st_wait = MPI_Wtime();
			if (check_again){
				MPI_Wait(&did_I_recv_num_of_giveups[source_index], MPI_STATUS_IGNORE);
				// initate giveup recv
				recieved_giveups.emplace_back((double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_index] * 2 * sizeof(double)));
				//recieved_giveups.emplace_back((double *) malloc((long unsigned int) sizes_to_recv[0] * 2 * sizeof(double)));	// allocate mem
				if (recieved_giveups.back() == nullptr){	// error checking
					std::cerr << "Error allocating memory for recieved_giveups[0]\n";
					MPI_Abort(cartesianCOMM, myid);
				}
				MPI_Irecv(recieved_giveups.back(), number_of_giveups_to_recieve[source_index] * 2, MPI_DOUBLE, my_neighbours[source_index], 1, cartesianCOMM, &did_I_recv_giveups[source_index]);
			}

			
			// wait on recv of num of backups
			if (check_again2){
				MPI_Wait(&did_I_recv_num_of_backups[source_index], MPI_STATUS_IGNORE);
				// initiate backup recv
				recieved_backups.emplace_back((double *) malloc((long unsigned int) number_of_backups_to_recieve[source_index] * 2 * sizeof(double)));
				if (recieved_backups.back() == nullptr){
					std::cerr << "Error allocating memory for recieved_backups[0]\n";
					MPI_Abort(cartesianCOMM, myid);
				}
				MPI_Irecv(recieved_backups.back(), number_of_backups_to_recieve[source_index] * 2, MPI_DOUBLE, my_neighbours[source_index], 3, cartesianCOMM, &did_I_recv_backups);
			}


			// Wait on giveups and backups
			MPI_Wait(&did_I_recv_giveups[source_index], MPI_STATUS_IGNORE);
			MPI_Wait(&did_I_recv_backups, MPI_STATUS_IGNORE);
			et_wait = MPI_Wtime();

			// expand boundary lines to encompass the whole space
			expand_boundary_lines(boundary_lines, min, max, spacing);

			// compute the remaining cells
			std::vector<size_t> source_indices(1, source_index);
			compute_remaining_cells(mypoints, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve, mycells, giveup_indices, &giveups_time, &recompute_time);


			// we need to have RECIEVED giveups and backups before returning, but we do not need to have completed the sends
			// hence "did_they_recv" requests should be declared outside of this switch statement
			break;
		}
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//	--------------------------------------------------------------------------		2 SIDES TO SEND TO
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		case 2:
		{
			// who are we communicating with?
			std::vector<size_t> source_indices;
			unsigned int total_number_of_processes_to_communicate_with = 0;	// this will be either 2 or 3 (no corner / corner)
			for (int i = 0; i<8; i++){
				if (we_send_to_original_copy[i]){
					source_indices.emplace_back((size_t) i);
					total_number_of_processes_to_communicate_with++;
				}
			}

			if (myid == print_rank) {
				if (total_number_of_processes_to_communicate_with == 3) std::cout << "Rank " << myid <<": I will communicate with " << source_indices[0] << ", " << source_indices[1] << ", " << source_indices[2] << "\n";
				else std::cout << "Rank " << myid <<": I will communicate with " << source_indices[0] << ", " << source_indices[1] << "\n";
			}

			
			// send number of giveups:
			did_they_recieve_num_of_giveups.resize(2);
			MPI_Isend(&counters[0], 1, MPI_INT, my_neighbours[source_indices[0]], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[0]);
			MPI_Isend(&counters[1], 1, MPI_INT, my_neighbours[source_indices[1]], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[1]);
			if (myid == print_rank) std::cout << "Rank " << myid <<": number of giveups: sending " << counters[0] << " to " << my_neighbours[source_indices[0]] << " and " << counters[1] << " to " << my_neighbours[source_indices[1]] << "\n";


			// test if we have recieved the number of giveups to recieve:
			recieved_giveups.resize(total_number_of_processes_to_communicate_with, nullptr);
			int flag[] = {0,0};
			int check_again[] = {1,1};
			MPI_Test(&did_I_recv_num_of_giveups[source_indices[0]], &flag[0], MPI_STATUS_IGNORE);
			if (flag[0]){
				if (myid == print_rank) std::cout << "Rank " << myid <<": recieved num_giveups from " << my_neighbours[source_indices[0]] << ": " << number_of_giveups_to_recieve[source_indices[0]] << "\n";
				// initiate giveup recv
				recieved_giveups[0] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[0]] * 2 * sizeof(double));
				if (recieved_giveups[0] == nullptr){	// error checking
					std::cerr << "Error allocating memory for recieved_giveups[0]\n";
					MPI_Abort(cartesianCOMM, myid);
				}
				check_again[0] = 0;
				MPI_Irecv(recieved_giveups[0], number_of_giveups_to_recieve[source_indices[0]] * 2, MPI_DOUBLE, my_neighbours[source_indices[0]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[0]]);
			}

			MPI_Test(&did_I_recv_num_of_giveups[source_indices[1]], &flag[1], MPI_STATUS_IGNORE);
			if (flag[1]){
				if (myid == print_rank) std::cout << "Rank " << myid <<": recieved num_giveups from " << my_neighbours[source_indices[1]] << ": " << number_of_giveups_to_recieve[source_indices[1]] << "\n";
				// initiate giveup recv
				recieved_giveups[1] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[1]] * 2 * sizeof(double));
				if (recieved_giveups[1] == nullptr){	// error checking
					std::cerr << "Error allocating memory for recieved_giveups[0]\n";
					MPI_Abort(cartesianCOMM, myid);
				}
				check_again[1] = 0;
				MPI_Irecv(recieved_giveups[1], number_of_giveups_to_recieve[source_indices[1]] * 2, MPI_DOUBLE, my_neighbours[source_indices[1]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[1]]);
			}


			// send giveups
			did_they_recieve_giveups.resize(2);
			MPI_Isend(giveups[0], counters[0]*2, MPI_DOUBLE, my_neighbours[source_indices[0]], 1, cartesianCOMM, &did_they_recieve_giveups[0]);
			MPI_Isend(giveups[1], counters[1]*2, MPI_DOUBLE, my_neighbours[source_indices[1]], 1, cartesianCOMM, &did_they_recieve_giveups[1]);


			// now arrange backup data to send
			backups.resize(3, 0);
			num_of_backup_points_to_send.resize(3, 0);
			num_of_backup_points_to_send_ints.resize(3,0);
			st_send = MPI_Wtime();
			send2(backups, num_of_backup_points_to_send, send_info, mypoints, counters, myid);
			et_send = MPI_Wtime();
			num_of_backup_points_to_send_ints[0] = (int)num_of_backup_points_to_send[0];
			num_of_backup_points_to_send_ints[1] = (int)num_of_backup_points_to_send[1];
			num_of_backup_points_to_send_ints[2] = (int)num_of_backup_points_to_send[2];


			// send the number of backup points to be sent later, and send the backup points
			did_they_recieve_num_of_backups.resize(total_number_of_processes_to_communicate_with);
			did_they_recieve_backups.resize(total_number_of_processes_to_communicate_with);
			for (unsigned int i = 0; i < total_number_of_processes_to_communicate_with; i++){
				MPI_Isend(&num_of_backup_points_to_send_ints[i], 1, MPI_INT, my_neighbours[source_indices[i]], 2, cartesianCOMM, &did_they_recieve_num_of_backups[i]);
				MPI_Isend(backups[i], num_of_backup_points_to_send_ints[i]*2, MPI_DOUBLE, my_neighbours[source_indices[i]], 3, cartesianCOMM, &did_they_recieve_backups[i]);
			}


			// test whether we have recieved the number of backups to recieve
			int flag2[] = {0,0,0};
			int check_again2[] = {1,1, ((total_number_of_processes_to_communicate_with == 3) ? 1 : 0)};
			recieved_backups.resize(total_number_of_processes_to_communicate_with, nullptr);
			std::vector<MPI_Request> did_I_recv_backups(total_number_of_processes_to_communicate_with);
			for (unsigned int i = 0; i<total_number_of_processes_to_communicate_with; i++){
				MPI_Test(&did_I_recv_num_of_backups[source_indices[i]], &flag2[i], MPI_STATUS_IGNORE);
				if (flag2[i]){
					if (myid == print_rank) std::cout << "Rank " << myid <<": recieved num_backups from " << my_neighbours[source_indices[i]] << ": " << number_of_backups_to_recieve[source_indices[i]] << "\n";
					// initiate backup recv
					recieved_backups[i] = (double *) malloc((long unsigned int) number_of_backups_to_recieve[source_indices[i]] * 2 * sizeof(double));
					if (recieved_backups[i] == nullptr){
						std::cerr << "Error allocating memory for recieved_backups[]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again2[i] = 0;
					MPI_Irecv(recieved_backups[i], number_of_backups_to_recieve[source_indices[i]]*2, MPI_DOUBLE, my_neighbours[source_indices[i]], 3, cartesianCOMM, &did_I_recv_backups[i]);
				}
			}


			// wait if we still need to wait for number of giveups/backups
			st_wait = MPI_Wtime();
			while (check_again[0] || check_again[1] || check_again2[0] || check_again2[1] || check_again2[2]){
				if (check_again[0]){
					MPI_Test(&did_I_recv_num_of_giveups[source_indices[0]], &flag[0], MPI_STATUS_IGNORE);
					if (flag[0]){
						if (myid == print_rank) std::cout << "Rank " << myid <<": recieved num_giveups from " << my_neighbours[source_indices[0]] << ": " << number_of_giveups_to_recieve[source_indices[0]] << "\n";
						// initiate giveup recv
						recieved_giveups[0] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[0]] * 2 * sizeof(double));
						if (recieved_giveups[0] == nullptr){	// error checking
							std::cerr << "Error allocating memory for recieved_giveups[0]\n";
							MPI_Abort(cartesianCOMM, myid);
						}
						check_again[0] = 0;
						MPI_Irecv(recieved_giveups[0], number_of_giveups_to_recieve[source_indices[0]] * 2, MPI_DOUBLE, my_neighbours[source_indices[0]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[0]]);
					} 
				} 
				if (check_again[1]) {
					MPI_Test(&did_I_recv_num_of_giveups[source_indices[1]], &flag[1], MPI_STATUS_IGNORE);
					if (flag[1]){
						if (myid == print_rank) std::cout << "Rank " << myid <<": recieved num_giveups from " << my_neighbours[source_indices[1]] << ": " << number_of_giveups_to_recieve[source_indices[1]] << "\n";
						// initiate giveup recv
						recieved_giveups[1] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[1]] * 2 * sizeof(double));
						if (recieved_giveups[1] == nullptr){	// error checking
							std::cerr << "Error allocating memory for recieved_giveups[0]\n";
							MPI_Abort(cartesianCOMM, myid);
						}
						check_again[1] = 0;
						MPI_Irecv(recieved_giveups[1], number_of_giveups_to_recieve[source_indices[1]] * 2, MPI_DOUBLE, my_neighbours[source_indices[1]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[1]]);
					}
				}
				for (unsigned int i = 0; i<total_number_of_processes_to_communicate_with; i++){
					if (check_again2[i]){
						MPI_Test(&did_I_recv_num_of_backups[source_indices[i]], &flag2[i], MPI_STATUS_IGNORE);
						if (flag2[i]){
							if (myid == print_rank) std::cout << "Rank " << myid <<": recieved num_backups from " << my_neighbours[source_indices[i]] << ": " << number_of_backups_to_recieve[source_indices[i]] << "\n";
							// initiate backup recv
							recieved_backups[i] = (double *) malloc((long unsigned int) number_of_backups_to_recieve[source_indices[i]] * 2 * sizeof(double));
							if (recieved_backups[i] == nullptr){
								std::cerr << "Error allocating memory for recieved_backups[]\n";
								MPI_Abort(cartesianCOMM, myid);
							}
							check_again2[i] = 0;
							MPI_Irecv(recieved_backups[i], number_of_backups_to_recieve[source_indices[i]]*2, MPI_DOUBLE, my_neighbours[source_indices[i]], 3, cartesianCOMM, &did_I_recv_backups[i]);
						}
					}
				}
			}

			// Now we have initiated all recv's for backups and giveups to recieve
			// Wait to recv giveups and backups
			for (unsigned int i = 0; i < total_number_of_processes_to_communicate_with; i++){
				MPI_Wait(&did_I_recv_giveups[source_indices[i]], MPI_STATUS_IGNORE);
				MPI_Wait(&did_I_recv_backups[i], MPI_STATUS_IGNORE);
			}
			et_wait = MPI_Wtime();

			if (total_number_of_processes_to_communicate_with == 3){
				recieved_giveups[2] = &recieved_corner_giveups[source_indices[2]*2-8];
				number_of_giveups_to_recieve[source_indices[2]] = 1;
			}

			// expand boundary lines to encompass the whole space
			expand_boundary_lines(boundary_lines, min, max, spacing);

			// compute the new cells for the giveups we recieved
			compute_remaining_cells(mypoints, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve, mycells, giveup_indices, &giveups_time, &recompute_time);

			if (total_number_of_processes_to_communicate_with == 3) recieved_giveups.pop_back();

			// we need to have RECIEVED giveups and backups before returning, but we do not need to have completed the sends
			// hence "did_they_recv" requests should be declared outside of this switch statement
			break;
		}
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//	------------------------------------------------------------------------------------		3 SIDES TO SEND TO
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		case 3:
		{
			// who are we communicating with?
			std::vector<size_t> source_indices;
			for (int i = 0; i<8; i++){
				if (we_send_to_original_copy[i]){
					source_indices.emplace_back((size_t) i);
				}
			}

			// send number of giveups:
			did_they_recieve_num_of_giveups.resize(3);
			MPI_Isend(&counters[0], 1, MPI_INT, my_neighbours[source_indices[0]], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[0]);
			MPI_Isend(&counters[1], 1, MPI_INT, my_neighbours[source_indices[1]], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[1]);
			MPI_Isend(&counters[2], 1, MPI_INT, my_neighbours[source_indices[2]], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[2]);


			// test if we have recieved the number of giveups to recieve:
				recieved_giveups.resize(5, nullptr);
				int flag[] = {0,0,0};
				int check_again[] = {1,1,1};
				MPI_Test(&did_I_recv_num_of_giveups[source_indices[0]], &flag[0], MPI_STATUS_IGNORE);
				if (flag[0]){
					// initiate giveup recv
					recieved_giveups[0] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[0]] * 2 * sizeof(double));
					if (recieved_giveups[0] == nullptr){	// error checking
						std::cerr << "Error allocating memory for recieved_giveups[0]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again[0] = 0;
					MPI_Irecv(recieved_giveups[0], number_of_giveups_to_recieve[source_indices[0]] * 2, MPI_DOUBLE, my_neighbours[source_indices[0]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[0]]);
				}

				MPI_Test(&did_I_recv_num_of_giveups[source_indices[1]], &flag[1], MPI_STATUS_IGNORE);
				if (flag[1]){
					// initiate giveup recv
					recieved_giveups[1] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[1]] * 2 * sizeof(double));
					if (recieved_giveups[1] == nullptr){	// error checking
						std::cerr << "Error allocating memory for recieved_giveups[0]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again[1] = 0;
					MPI_Irecv(recieved_giveups[1], number_of_giveups_to_recieve[source_indices[1]] * 2, MPI_DOUBLE, my_neighbours[source_indices[1]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[1]]);
				}

				MPI_Test(&did_I_recv_num_of_giveups[source_indices[2]], &flag[2], MPI_STATUS_IGNORE);
				if (flag[2]){
					// initiate giveup recv
					recieved_giveups[2] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[2]] * 2 * sizeof(double));
					if (recieved_giveups[2] == nullptr){	// error checking
						std::cerr << "Error allocating memory for recieved_giveups[0]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again[2] = 0;
					MPI_Irecv(recieved_giveups[2], number_of_giveups_to_recieve[source_indices[2]] * 2, MPI_DOUBLE, my_neighbours[source_indices[2]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[2]]);
				}
			//


			// send giveups
			did_they_recieve_giveups.resize(3);
			MPI_Isend(giveups[0], counters[0]*2, MPI_DOUBLE, my_neighbours[source_indices[0]], 1, cartesianCOMM, &did_they_recieve_giveups[0]);
			MPI_Isend(giveups[1], counters[1]*2, MPI_DOUBLE, my_neighbours[source_indices[1]], 1, cartesianCOMM, &did_they_recieve_giveups[1]);
			MPI_Isend(giveups[2], counters[2]*2, MPI_DOUBLE, my_neighbours[source_indices[2]], 1, cartesianCOMM, &did_they_recieve_giveups[2]);


			// now arrange backup data to send
			backups.resize(6, 0);
			num_of_backup_points_to_send.resize(6, 0);
			num_of_backup_points_to_send_ints.resize(5,0);
			st_send = MPI_Wtime();
			send3(backups, num_of_backup_points_to_send, send_info, mypoints, counters, myid, we_send_to_original_copy);
			et_send = MPI_Wtime();
			num_of_backup_points_to_send_ints[0] = (int)num_of_backup_points_to_send[0];
			num_of_backup_points_to_send_ints[1] = (int)num_of_backup_points_to_send[1];
			num_of_backup_points_to_send_ints[2] = (int)num_of_backup_points_to_send[2];
			num_of_backup_points_to_send_ints[3] = (int)num_of_backup_points_to_send[3];
			num_of_backup_points_to_send_ints[4] = (int)num_of_backup_points_to_send[4];

			// send the number of backup points to be sent later, and send the backup points
			did_they_recieve_num_of_backups.resize(5);
			did_they_recieve_backups.resize(5);
			for (unsigned int i = 0; i < 5; i++){
				MPI_Isend(&num_of_backup_points_to_send_ints[i], 1, MPI_INT, my_neighbours[source_indices[i]], 2, cartesianCOMM, &did_they_recieve_num_of_backups[i]);
				MPI_Isend(backups[i], num_of_backup_points_to_send_ints[i]*2, MPI_DOUBLE, my_neighbours[source_indices[i]], 3, cartesianCOMM, &did_they_recieve_backups[i]);
			}


			// test whether we have recieved the number of backups to recieve
			int flag2[] = {0,0,0,0,0};
			int check_again2[] = {1,1,1,1,1};
			recieved_backups.resize(5, nullptr);
			std::vector<MPI_Request> did_I_recv_backups(5);
			for (unsigned int i = 0; i<5; i++){
				MPI_Test(&did_I_recv_num_of_backups[source_indices[i]], &flag2[i], MPI_STATUS_IGNORE);
				if (flag2[i]){
					// initiate backup recv
					recieved_backups[i] = (double *) malloc((long unsigned int) number_of_backups_to_recieve[source_indices[i]] * 2 * sizeof(double));
					if (recieved_backups[i] == nullptr){
						std::cerr << "Error allocating memory for recieved_backups[]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again2[i] = 0;
					MPI_Irecv(recieved_backups[i], number_of_backups_to_recieve[source_indices[i]]*2, MPI_DOUBLE, my_neighbours[source_indices[i]], 3, cartesianCOMM, &did_I_recv_backups[i]);
				}
			}


			// wait if we still need to wait for number of giveups/backups
			st_wait = MPI_Wtime();
			int we_need_to_wait = check_again[0] +  check_again[1] + check_again[2] + check_again2[0] + check_again2[1] + check_again2[2] + check_again2[3] + check_again2[4];
			while (we_need_to_wait){
				for (unsigned int i = 0; i < 3; i++){
					if (check_again[i]){
						MPI_Test(&did_I_recv_num_of_giveups[source_indices[i]], &flag[i], MPI_STATUS_IGNORE);
						if (flag[i]){
							// initiate giveup recv
							recieved_giveups[i] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[source_indices[i]] * 2 * sizeof(double));
							if (recieved_giveups[i] == nullptr){	// error checking
								std::cerr << "Error allocating memory for recieved_giveups[i]\n";
								MPI_Abort(cartesianCOMM, myid);
							}
							check_again[i] = 0;
							we_need_to_wait--;
							MPI_Irecv(recieved_giveups[i], number_of_giveups_to_recieve[source_indices[i]] * 2, MPI_DOUBLE, my_neighbours[source_indices[i]], 1, cartesianCOMM, &did_I_recv_giveups[source_indices[i]]);
						} 
					} 
				}
				for (unsigned int i = 0; i<5; i++){
					if (check_again2[i]){
						MPI_Test(&did_I_recv_num_of_backups[source_indices[i]], &flag2[i], MPI_STATUS_IGNORE);
						if (flag2[i]){
							// initiate backup recv
							recieved_backups[i] = (double *) malloc((long unsigned int) number_of_backups_to_recieve[source_indices[i]] * 2 * sizeof(double));
							if (recieved_backups[i] == nullptr){
								std::cerr << "Error allocating memory for recieved_backups[]\n";
								MPI_Abort(cartesianCOMM, myid);
							}
							check_again2[i] = 0;
							we_need_to_wait--;
							MPI_Irecv(recieved_backups[i], number_of_backups_to_recieve[source_indices[i]]*2, MPI_DOUBLE, my_neighbours[source_indices[i]], 3, cartesianCOMM, &did_I_recv_backups[i]);
						}
					}
				}
			}

			// Now we have initiated all recv's for backups and giveups to recieve
			// So we must wait for those recv's to finish
			for (unsigned int i = 0; i < 5; i++){
				MPI_Wait(&did_I_recv_giveups[source_indices[i]], MPI_STATUS_IGNORE);
				MPI_Wait(&did_I_recv_backups[i], MPI_STATUS_IGNORE);
			}
			et_wait = MPI_Wtime();

			recieved_giveups[3] = &recieved_corner_giveups[source_indices[3]*2-8];
			recieved_giveups[4] = &recieved_corner_giveups[source_indices[4]*2-8];
			number_of_giveups_to_recieve[source_indices[3]] = 1;
			number_of_giveups_to_recieve[source_indices[4]] = 1;

			// expand boundary lines to encompass the whole space
			expand_boundary_lines(boundary_lines, min, max, spacing);
			// compute the cells for the giveups we recieved
			compute_remaining_cells(mypoints, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve, mycells, giveup_indices, &giveups_time, &recompute_time);

			recieved_giveups.pop_back();
			recieved_giveups.pop_back();

			break;
		}
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//	-------------------------------------------------------------------------		4 SIDES TO SEND TO
		//	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		case 4:
		{
			// send number of giveups:
			did_they_recieve_num_of_giveups.resize(4);
			MPI_Isend(&counters[0], 1, MPI_INT, my_neighbours[0], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[0]);
			MPI_Isend(&counters[1], 1, MPI_INT, my_neighbours[1], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[1]);
			MPI_Isend(&counters[2], 1, MPI_INT, my_neighbours[2], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[2]);
			MPI_Isend(&counters[3], 1, MPI_INT, my_neighbours[3], 0, cartesianCOMM, &did_they_recieve_num_of_giveups[3]);

			// test if we have recieved the number of giveups to recieve:
			recieved_giveups.resize(8, nullptr);
			int flag[] = {0,0,0,0};
			int check_again[] = {1,1,1,1};
			for (unsigned int i = 0; i < 4; i++){
				MPI_Test(&did_I_recv_num_of_giveups[i], &flag[i], MPI_STATUS_IGNORE);
				if (flag[i]){
					// initiate giveup recv
					recieved_giveups[i] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[i] * 2 * sizeof(double));
					if (recieved_giveups[i] == nullptr){	// error checking
						std::cerr << "Error allocating memory for recieved_giveups[0]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again[i] = 0;
					MPI_Irecv(recieved_giveups[i], number_of_giveups_to_recieve[i] * 2, MPI_DOUBLE, my_neighbours[i], 1, cartesianCOMM, &did_I_recv_giveups[i]);
				}
			}


			// send giveups
			did_they_recieve_giveups.resize(4);
			MPI_Isend(giveups[0], counters[0]*2, MPI_DOUBLE, my_neighbours[0], 1, cartesianCOMM, &did_they_recieve_giveups[0]);
			MPI_Isend(giveups[1], counters[1]*2, MPI_DOUBLE, my_neighbours[1], 1, cartesianCOMM, &did_they_recieve_giveups[1]);
			MPI_Isend(giveups[2], counters[2]*2, MPI_DOUBLE, my_neighbours[2], 1, cartesianCOMM, &did_they_recieve_giveups[2]);
			MPI_Isend(giveups[3], counters[3]*2, MPI_DOUBLE, my_neighbours[3], 1, cartesianCOMM, &did_they_recieve_giveups[3]);


			// now arrange backup data to send
			backups.resize(8, 0);
			num_of_backup_points_to_send.resize(8, 0);
			num_of_backup_points_to_send_ints.resize(8, 0);
			st_send = MPI_Wtime();
			send4(backups, num_of_backup_points_to_send, send_info, mypoints, counters, myid);
			et_send = MPI_Wtime();
			num_of_backup_points_to_send_ints[0] = (int)num_of_backup_points_to_send[0];
			num_of_backup_points_to_send_ints[1] = (int)num_of_backup_points_to_send[1];
			num_of_backup_points_to_send_ints[2] = (int)num_of_backup_points_to_send[2];
			num_of_backup_points_to_send_ints[3] = (int)num_of_backup_points_to_send[3];
			num_of_backup_points_to_send_ints[4] = (int)num_of_backup_points_to_send[4];
			num_of_backup_points_to_send_ints[5] = (int)num_of_backup_points_to_send[5];
			num_of_backup_points_to_send_ints[6] = (int)num_of_backup_points_to_send[6];
			num_of_backup_points_to_send_ints[7] = (int)num_of_backup_points_to_send[7];

			// send the number of backup points to be sent later, and send the backup points
			did_they_recieve_num_of_backups.resize(8);
			did_they_recieve_backups.resize(8);
			for (unsigned int i = 0; i < 8; i++){
				MPI_Isend(&num_of_backup_points_to_send_ints[i], 1, MPI_INT, my_neighbours[i], 2, cartesianCOMM, &did_they_recieve_num_of_backups[i]);
				MPI_Isend(backups[i], num_of_backup_points_to_send_ints[i]*2, MPI_DOUBLE, my_neighbours[i], 3, cartesianCOMM, &did_they_recieve_backups[i]);
			}


			// test whether we have recieved the number of backups to recieve
			int flag2[] = {0,0,0,0,0,0,0,0};
			int check_again2[] = {1,1,1,1,1,1,1,1};
			recieved_backups.resize(8, nullptr);
			std::vector<MPI_Request> did_I_recv_backups(8);
			for (unsigned int i = 0; i<8; i++){
				MPI_Test(&did_I_recv_num_of_backups[i], &flag2[i], MPI_STATUS_IGNORE);
				if (flag2[i]){
					// initiate backup recv
					recieved_backups[i] = (double *) malloc((long unsigned int) number_of_backups_to_recieve[i] * 2 * sizeof(double));
					if (recieved_backups[i] == nullptr){
						std::cerr << "Error allocating memory for recieved_backups[]\n";
						MPI_Abort(cartesianCOMM, myid);
					}
					check_again2[i] = 0;
					MPI_Irecv(recieved_backups[i], number_of_backups_to_recieve[i]*2, MPI_DOUBLE, my_neighbours[i], 3, cartesianCOMM, &did_I_recv_backups[i]);
				}
			}


			// wait if we still need to wait for number of giveups/backups
			st_wait = MPI_Wtime();
			int we_need_to_wait = check_again[0] +  check_again[1] + check_again[2] + check_again[3] + check_again2[0] + check_again2[1] + check_again2[2] + check_again2[3] + check_again2[4] + check_again2[5] + check_again2[6] + check_again2[7];
			while (we_need_to_wait){
				for (unsigned int i = 0; i < 4; i++){
					if (check_again[i]){
						MPI_Test(&did_I_recv_num_of_giveups[i], &flag[i], MPI_STATUS_IGNORE);
						if (flag[i]){
							// initiate giveup recv
							recieved_giveups[i] = (double *) malloc((long unsigned int) number_of_giveups_to_recieve[i] * 2 * sizeof(double));
							if (recieved_giveups[i] == nullptr){	// error checking
								std::cerr << "Error allocating memory for recieved_giveups[i]\n";
								MPI_Abort(cartesianCOMM, myid);
							}
							check_again[i] = 0;
							we_need_to_wait--;
							MPI_Irecv(recieved_giveups[i], number_of_giveups_to_recieve[i] * 2, MPI_DOUBLE, my_neighbours[i], 1, cartesianCOMM, &did_I_recv_giveups[i]);
						} 
					} 
				}
				for (unsigned int i = 0; i<8; i++){
					if (check_again2[i]){
						MPI_Test(&did_I_recv_num_of_backups[i], &flag2[i], MPI_STATUS_IGNORE);
						if (flag2[i]){
							// initiate backup recv
							recieved_backups[i] = (double *) malloc((long unsigned int) number_of_backups_to_recieve[i] * 2 * sizeof(double));
							if (recieved_backups[i] == nullptr){
								std::cerr << "Error allocating memory for recieved_backups[]\n";
								MPI_Abort(cartesianCOMM, myid);
							}
							check_again2[i] = 0;
							we_need_to_wait--;
							MPI_Irecv(recieved_backups[i], number_of_backups_to_recieve[i]*2, MPI_DOUBLE, my_neighbours[i], 3, cartesianCOMM, &did_I_recv_backups[i]);
						}
					}
				}
			}


			// Now we have initiated all recv's for backups and giveups to recieve
			// So we must wait for those recv's to finish
			for (unsigned int i = 0; i < 8; i++){
				MPI_Wait(&did_I_recv_giveups[i], MPI_STATUS_IGNORE);
				MPI_Wait(&did_I_recv_backups[i], MPI_STATUS_IGNORE);
			}

			et_wait = MPI_Wtime();


			recieved_giveups[4] = &recieved_corner_giveups[0];
			recieved_giveups[5] = &recieved_corner_giveups[2];
			recieved_giveups[6] = &recieved_corner_giveups[4];
			recieved_giveups[7] = &recieved_corner_giveups[6];
			number_of_giveups_to_recieve[4] = 1;
			number_of_giveups_to_recieve[5] = 1;
			number_of_giveups_to_recieve[6] = 1;
			number_of_giveups_to_recieve[7] = 1;


			// expand boundary lines to encompass the whole space
			expand_boundary_lines(boundary_lines, min, max, spacing);
			std::vector<size_t> source_indices{0,1,2,3,4,5,6,7};
			// compute the cells for the giveups we received
			compute_remaining_cells(mypoints, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve, mycells, giveup_indices, &giveups_time, &recompute_time);

			recieved_giveups.pop_back();
			recieved_giveups.pop_back();
			recieved_giveups.pop_back();
			recieved_giveups.pop_back();

			break;
		}
	}

	et_total2 = MPI_Wtime();

	// must wait to delete giveups and backups until this data has certainly been read by other processes
	std::cout << myid << " completed\n";
	MPI_Barrier(cartesianCOMM);

	// print results to file
	std::string results_filename = "results_complex_parallel_" + std::to_string(num_points) + ".txt";
	double st_writing = MPI_Wtime();
	print_cells_to_file(results_filename, myid, size, mycells, cartesianCOMM);
	double et_writing = MPI_Wtime();

	// free malloc'd data
	for (auto it = giveups.begin(); it != giveups.end(); ++it){
		free (*it);
	}
	for (auto it = recieved_giveups.begin(); it != recieved_giveups.end(); ++it){
		free (*it);
	}
	for (auto it = recieved_backups.begin(); it != recieved_backups.end(); ++it){
		free (*it);
	}
	for (auto it = backups.begin(); it != backups.end(); ++it){
		free (*it);
	}
	free(send_info);
	free(we_send_to);
	free(we_send_to_original_copy);

	if (myid != 0){
		double timing_results[8];
		timing_results[0] = et_total1 - st_total1 + et_total2 - st_total2;		// total time
		timing_results[1] = et_readingpoints1 + et_readingpoints2 - st_readingpoints1 - st_readingpoints2;		// time to read through input points
		timing_results[2] = et_cells1 - st_cells1;	// time computing initial local diagram
		timing_results[3] = et_wait - st_wait;		// time waiting
		timing_results[4] = et_send - st_send;		// time assorting backups to be sent
		timing_results[5] = giveups_time;			// time computing cells for recieved giveups
		timing_results[6] = recompute_time;			// time recomputing some of our initial cells
		timing_results[7] = et_writing - st_writing;// time to write to file

		// send timing results to rank 0 to print to file
		MPI_Send(&timing_results, 8, MPI_DOUBLE, 0, 0, cartesianCOMM);

	} else {
		double timing_results[(long unsigned int)size*8];
		timing_results[0] = et_total1 - st_total1 + et_total2 - st_total2;		// total time
		timing_results[1] = et_readingpoints1 + et_readingpoints2 - st_readingpoints1 - st_readingpoints2;		// time to read through input points
		timing_results[2] = et_cells1 - st_cells1;	// time computing initial local diagram
		timing_results[3] = et_wait - st_wait;		// time waiting
		timing_results[4] = et_send - st_send;		// time assorting backups to be sent
		timing_results[5] = giveups_time;			// time computing cells for recieved giveups
		timing_results[6] = recompute_time;			// time recomputing some of our initial cells
		timing_results[7] = et_writing - st_writing;// time to write to file

		for (int i = 1; i < size; i++){
			MPI_Recv(&timing_results[8*i], 8, MPI_DOUBLE, i, 0, cartesianCOMM, MPI_STATUS_IGNORE);
		}

		// print timing results to file
		std::string timing_filename = "timing_results_complex_" + std::to_string(num_points) + ".txt";

		std::ofstream fp;
		fp.open(timing_filename);
		if(!fp){	// error checking
			std::cerr<<"Failed to open file for writing timing data";
			return -1;
		}
		fp << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		fp << "TIMING RESULTS (seconds)     Layout: " << dims[0] << " by " << dims[1] << "\n";
		fp << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		fp << "Process:   Total:   Read points:   Compute inital cells:   Wait on recvs:   Assort backups:   Compute giveup cells:   Recompute:   Write to file:\n";
		fp << std::fixed;
    	fp << std::setprecision(4);
		for (int i = 0; i < size; i++) {
			fp << "  " << i << "\t   " << timing_results[8*i] << "   " << timing_results[8*i+1] << "\t   " << timing_results[8*i+2] << "\t\t   " << timing_results[8*i+3] << "\t    " << timing_results[8*i+4] << "\t      " << timing_results[8*i+5] << "\t\t      " << timing_results[8*i+6] << "\t   " << timing_results[8*i+7] << "\n";
		}
		fp << "\n";
		fp.close();

		std::cout<< "\nProgram complete:\nSee results_complex_parallel_" + std::to_string(num_points) + ".txt for diagram results and timing_results_complex_" + std::to_string(num_points) + ".txt for timing results.\n";
		std::cout<<"The file results_complex_parallel_" + std::to_string(num_points) + ".txt can be read by the python matplotlib script Results_Plot.py\n";


		// now we test accuracy if the user has specified that we do so
		if (test_accuracy){
			std::string correct_filename;
			if(test_accuracy_filename == nullptr){
				correct_filename = "results_simple_parallel_" + std::to_string(number_of_points) + ".txt";
			} else {
				correct_filename = test_accuracy_filename;
			}
			std::cout<<"\n\nWe shall test accuracy of results against " << correct_filename << "\n";
			test_complex_accuracy(number_of_points, correct_filename, "results_complex_parallel_" + std::to_string(number_of_points) + ".txt");
		}
	}

	MPI_Finalize();
	return 0;
}


/**
 *	\brief This function reads the flags passed by the user in the command line.
 *
 *	@param [in] argc The number of arguments
 *	@param [in] argv The arguments passed
 *	@param [in] rank The rank of the calling process
 *	@param [in] size The number of processes in MPI_COMM_WORLD
 *	@param [out] points_filename The address of a char * to be filled with the pointer to the filename containing input points
 *	@param [out] spacing The address of a double to be filled with the requested boundary spacing
 *	@param [out] xdim The address of an int to be filled with the number of processes in the x-dimension of the processor layout
 *	@param [out] ydim The address of an int to be filled with the number of processes in the y-dimension of the processor layout.
 *	@return 0 if all required arguments are passed; -1 if not all required arguments are passed
 *	
 */
int parseArguments(int argc, char * argv[], int rank, int size, char ** points_filename, double * spacing, int * xdim, int * ydim){
	int opt;
	int exit = 0;	// exit program early ?
	int pu = 0;		// print usage ?
	int pfi = 0;	// print file info ?
	while ((opt = getopt(argc, argv, "p:id:hx:o:tu:")) != -1){
		switch(opt){
			case 'p':
				*points_filename = optarg;
				if (rank == 0) std::cout<<"Reading points from "<<*points_filename<<"\n";
				break;
			case 'h':
				pu = 1;
				exit = 1;
				break;
			case 'd':
				*spacing = atof(optarg);
				break;
			case 't':
				test_accuracy = true;
				break;
			case 'u':
				test_accuracy = true;
				test_accuracy_filename = optarg;
				break;
			case 'o':
				print_rank = atoi(optarg);
				break;
			case 'i':
				pfi = 1;
				exit = 1;
				break;
			case 'x':
				*xdim = atoi(optarg);
				if ((*xdim) <= 0){
					if (rank == 0) std::cout<<"Error: user specified x-dimension "<< *xdim <<" for processor grid. X-dimension must be a positive integer.\n";
					exit = 1;
				} else if ( (size % (*xdim)) != 0){
					if (rank == 0) std::cout<<"Error: user specified x-dimension "<< *xdim << " for processor grid, which is not a divisor of specified number of processes ("<<size<<"). Unable to form specified grid layout for processes.\n";
					exit = 1;
				} else {
					*ydim = size / (*xdim);
					if (rank == 0) std::cout<<"Using "<< *xdim<< " by "<< *ydim <<" processor grid layout\n";
				}
				break;
			default:
				if (rank == 0){
					std::cout<<"Unknown option passed\n";
					pu = 1;
				}
		}
	}


	if(*points_filename == nullptr){	// if no filename is specified to read input points
		if (rank == 0) std::cout<<"ERROR: User failed to specify filename for input points\n\n";
		pu = 1;		// print usage
		exit = 1;	// abort program early
	}
	if (*xdim == 0){					// if the user failed to specify an xdimension for the processor layout
		if (rank == 0) std::cout<<"ERROR: User failed to specify x-dimension of processor layout with flag -x [value]\n\n";
		pu = 1;		// print usage
		exit = 1;	// abort program early
	}
	if ((rank == 0) && pu) printUsage();		// print usage
	if ((rank == 0) && pfi) printFileInfo();	// print file information
	if(exit) {	// if we should abort the program early
		MPI_Finalize();
		return -1;
	}
	return 0;
}


/**
 *	\brief This function prints the usage statement.
 */
void printUsage(void){
	std::cout<<"Complex Parallel Voronoi Diagram program\nby: Eleanor Maher <mahere6@tcd.ie>\nThis program will calculate the 2D Voronoi diagram for inputted points within a convex region.\nusage: ";
	std::cout<<"mpirun -np [value] ./v_parallel_complex [options]\n";

	// options
	std::cout<<"        -p filename     : will cause the program to read input points from the txt file named [filename]\n";
	std::cout<<"        -d value        : will cause the program to generate boundary as a rectangle with padding of \n";
	std::cout<<"                          size [value] around extreme input points (default padding size: 2.0)\n";
	std::cout<<"        -h              : will print this usage statement\n";
	std::cout<<"        -i              : will print information about format of input files\n";
	std::cout<<"        -x value        : will specify the 2D processor layout by setting the number of processes in the x-dimension\n";
	std::cout<<"                          to be [value]. Thus the processor grid will be [value] by [total_number_of_processes / value]\n";
	std::cout<<"        -o value        : will cause the program to print information about the operation of the algorithm as running\n";
	std::cout<<"                          on rank [value]\n";
	std::cout<<"        -t              : will test the accuracy of the results generated against the cell data in the file named \n";
	std::cout<<"                             'results_simple_parallel_[value].txt'\n";
	std::cout<<"                          where [value] is the number of points specified after the flag -p\n";
	std::cout<<"        -u filename     : will test the accuracy of the results generated against the cell data in [filename].\n";

	// usage explanation
	std::cout<<"\n";
	std::cout<<"A filename with input points MUST be passed with the flag -p for the program to run.\n";
	std::cout<<"The default boundary is a rectangle surrounding the input points with spacing of default size 2.0 around extreme input points.\n";
	std::cout<<"The padding size for the default rectangle boundary can be adjusted with the flag -d\n";
	std::cout<<"The number of processes in the x-dimension MUST be specified. This must be a positive\n";
	std::cout<<"integer which divides the total number of processes specified (with the flag -np for OpenMPI).\n";
	std::cout<<"The flag -u overrides the flag -t.\n";
	std::cout<<"\n";
	std::cout<<"For more information about the format of input files for points and boundary lines, use the flag -i\n";
	std::cout<<"\n";
}


/**
 *	\brief This function prints information about the format of input files. It is called
 *	if the user passes the flag -i.
 */
void printFileInfo(void){
	std::cout<<"\n--- Input file information ---\n";
	std::cout<<"   Files containing input points should first include a positive integer\n";
	std::cout<<"   indicating the number of input points contained in the file, followed by a\n";
	std::cout<<"   space, tab, or newline. Input point values should be separated by spaces.\n";
	std::cout<<"   For example:\n";
	std::cout<<"   \t4\n\t0 1\n\t-3 4.8\n\t8 0.2\n\t4 -3\n   is suitable for a file of four input points, whereas\n";
	std::cout<<"   \t4\n\t0, 1\n\t-3, 4.8\n\t8, 0.2\n\t4, -3\n   is not suitable.\n\n";
	std::cout<<"   Files containing boundary line information should specify the boundary\n";
	std::cout<<"   lines as halfplanes, of the form ax + by <= c. The intersection of the\n";
	std::cout<<"   specified halfplanes should be a closed, convex region which contains all\n";
	std::cout<<"   input points. Files containing boundary line information should firsly\n";
	std::cout<<"   include a positive integer indicating the number of halfplanes to read.\n";
	std::cout<<"   Then, for each halfplane ax + by <= c, the values a, b, and c should be\n";
	std::cout<<"   included and separated by spaces.\n   For example:\n";
	std::cout<<"   \t4\n\t-1 0 14\n\t1 0 13\n\t0 -1 6.5\n\t0 1 9\n   would be suitable to input the following boundary halfplanes:\n";
	std::cout<<"   \tx >= -14\n\tx <= 13\n\ty >= -6.5\n\ty <= 9\n\n\n";

}


/**
 *	\brief This function takes a point p and a ray emanating from p and returns information
 *	on where the ray intersects with the boundary of the Voronoi cell of p.
 *
 *	In particular, this function returns the line upon which the segment of Voronoi cell wall
 *	that contains the intersection between the ray and the Voronoi cell boundary lies. Additionally,
 *	this function returns the distance to the intersection from the point. An ordered_vector object
 *	is inputted, and these two pieces of data are returned in the member variables boundary and t.
 *	First, the function checks that these pieces of information have not already been determined for
 *	the inputted ordered_vector. Then the function calls find_Intersection_with_Boundary(), which
 *	provides a starting point. We check if this starting intersection point is in the cell by
 *	comparing distances with all input points. If not, we consider a closer bisector line and
 *	the intersection with such line. Repeatedly check in this way until an intersection is found
 *	within the cell.
 *	
 *	@see find_Intersection_with_Boundary()
 *	@param [in] p_it A std::vector<point>::iterator referencing a point object from which the ray emanates
 *	@param [in, out] direction An ordered_vector object, a unit vector representing the direction
 *	of the ray. Upon return the member variables boundary and t will be populated with the 
 *	boundary line which the ray intersects with and the distance to that intersection, respectively.
 *	@param [in] input_points The input points provided by the user, for which we generate the Voronoi diagram
 *	@param [in] boundary_lines A std::vector of halfplanes describing the convex region
 *	@param [in, out] send_info An array of bools used to describe how each input point should be communicated to other processes
 *	@param [in] num_sides_to_send_to An int describing how many boundary sides this process will communicate through [1,4)
 *	@return 0
 */
int find_Endpoint_and_Line(std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines, bool * send_info, unsigned int num_sides_to_send_to){

	point p = *p_it;

	// first check if we have already computed endpoint
	if (direction.t > 0) return 0;


	// then find intersection of ray from p in [direction] with boundary of world
	unsigned int side = find_Intersection_with_Boundary(p, direction, boundary_lines);
	bool use_side = (side < num_sides_to_send_to ? true : false);		// This bool will inidcate if p's cell is on a boundary between another process's section

	bool foundEndpoint = false;		// Have we found the endpoint yet?
	std::vector<std::vector<point>::iterator> skip_items (1, p_it);		// Points that we don't want to consider as "closer" than p to suspected endpoint
	while (! foundEndpoint){
		point endpoint = p + (direction.t * direction);	// this is our suspected endpoint
		// check whether the endpoint we have found is in the cell by comparing distances to all other points
		bool in_cell = true;
		double min_distance = p.distance(endpoint);		// distance from p to endpoint
		std::vector<point>::iterator closeNeighbour = p_it;					// closest neighbour to endpoint
		int index = -1;		// used for recording index of nearest neighbour
		for (auto it = input_points.begin(); it != input_points.end(); ++it){		// loop through input points and find closest input point to endpoint and corresponding distance
			index++;
			double current_distance = (*it).distance(endpoint);		// distance from current input point to endpoint
			if ( current_distance < min_distance){	// if we have found a closer point
				// check whether we should skip this point or not (if the point is p or has been considered already)
				bool skip = false;
				for (auto& skip_it : skip_items){
					if (it == skip_it){
						skip = true;
						break;
					}
				}
				if (!skip){
					min_distance = current_distance;
					closeNeighbour = it;
					direction.nbr = index;
					in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
				}
			}
		}

		if (in_cell){	// we have found the endpoint and we are done
			foundEndpoint = true;
		} else {		// we have not found the endpoint; there is a closer input point to the current suspected endpoint
			// add closeNeighbour to skip_items
				skip_items.push_back(closeNeighbour);
			// we adjust suspected endpoint and boundary line
			// suspected boundary lines becomes the perpedicular bisector between p and closeNeighbour
				if ((p.y - (*closeNeighbour).y) != 0){	// if the bisector is not vertical
					direction.boundary = line{(p.x-(*closeNeighbour).x)/(p.y-(*closeNeighbour).y), 1, (p.y+(*closeNeighbour).y + (p.x * p.x - ((*closeNeighbour).x * (*closeNeighbour).x))/(p.y-(*closeNeighbour).y))*0.5};
				} else {	// the bisector is vertical
					direction.boundary = line{1, 0, (p.x + (*closeNeighbour).x)/2};
				}

			// suspected endpoint is now the intersection between the perpendicular bisector and ray from p
				direction.t = (direction.boundary.c - (direction.boundary)*p)/(direction.boundary * direction);

			// not on boundary so we won't return the boundary side
			use_side = false;
		}

	}

	if(use_side){	// if p's cell lies on a boundary shared with another process
		send_info[side] = true;		// p's cell needs to be recalculated by other processor with more information
	}

	return 0;
}


/**
 *	\brief This function finds the intersection of a ray from a given point with the boundary
 *	of the world. 
 *
 *	In particular, this function loops through the halfplanes defining the world until a 
 *	halfplane is found that the ray from the point intersects with. After this, the 
 *	function cycles through the remaining halfplanes and checks whether there is a
 *	closer intersection.
 *	The return value of the function indicates which boundary side the ray passes through
 *	(first), as the index [0,3] of the side in boundary_lines.
 *	
 *	@param [in] p A point object, the point from which the ray is shot
 *	@param [in, out] direction An ordered_vector object, a unit vector representing the direction
 *	of the ray. Upon return the member variables boundary and t will be populated with the 
 *	boundary line which the ray intersects with and the distance to that intersection, respectively.
 *	@param [in] boundary_lines A std::vector of halfplanes describing the convex region
 */
unsigned int find_Intersection_with_Boundary(const point p, ordered_vector& direction, std::vector<line>& boundary_lines){
	double ti;	// will hold the distance from p to boundary lines

	// cycle through halfplanes in boundary
	auto it = boundary_lines.begin();	// iterator through boundary lines
	unsigned int side = 0;				// index of boundary side that we are on
	bool foundFirstIntersection = false;	// have we found the first intersection with the boundary of the region (there will usually be two total)
	while ((! foundFirstIntersection)){		// first we find an initial intersection of the ray with boundary
		double parallel_check = ((*it)*direction);	// will be 0 if boundary line and direction vector are parallel
		if ( parallel_check != 0 ){	// if boundary line and direction vector are not parallel
			ti = ( (*it).c - ((*it)*p)) / parallel_check;	// distance from p to boundary line in direction of [direction]

			if (ti > 0){	// if ray intersects boundary line
				direction.t = ti;	// we have found first intersection
				direction.boundary = (*it);
				foundFirstIntersection = true;
			}
		}
		side++;	// move to the next side
		it++; // move to consider the next boundary line
	}
	
	side--;		// needed to ensure accurate side index because in loop side is incremented once after it is correctly found
	unsigned int closer = side;		// will hold the index of the side that we are checking to see if it is closer
	while (it != boundary_lines.end()){	// now we iterate through the remaining boundary lines and see if we can find a closer intersection
		closer++;
		double parallel_check = ((*it)*direction);	// will be 0 if boundary line and direction vector are parallel
		if ( parallel_check != 0 ){	// if boundary line and direction vector are not parallel
			ti = ( (*it).c - ((*it)*p)) / parallel_check;	// distance from p to boundary line in direction of [direction]

			if ((ti > 0) && (ti < direction.t)){	// if ray intersects boundary line AND at a point closer than before
				direction.t = ti;	// we have found a closer intersection
				direction.boundary = (*it);
				side = closer;		// adjust the index of the closer side
			}
		}
		it++; // move to consider the next boundary line
	}

	// done !
	return side;
}


/**
 *	\brief This function takes a point and checks whether the point lies within
 *	the region for which we determine the Voronoi diagram.
 *
 *	In particular, this function cycles through the halfplanes defined in 
 *	the form ax + by <= c and checks that the input point satifies these equations, 
 *	returning 0 if there is a halfplane that the point does not satisfy and 1 otherwise.
 *	Tolerance: this function will return zero if the point lies more than 1e-10 distance
 *	outside of the region.
 *	@param [in] pt A point object
 *	@param [in] boundary_lines A std::vector of halfplanes describing the convex region
 *	@return An int, 1 if the point lies in the convez region, 0 otherwise.
 */
int in_region(point pt, std::vector<line>& boundary_lines){
	// cycle through boundary planes
	for (auto it = boundary_lines.cbegin(); it != boundary_lines.cend(); ++it){
		if ((*it) * pt > (*it).c + 1e-10){	// check that the point is on the correct side of the halfplane
			return 0;
		}
	}
	return 1;
}


/**
 *	\brief This function determines which points to send as backups for a process that 
 *	communicates only with one other process.
 *
 *	In particular, this function loops through input_points and send_info, determining
 *	for each point if it must be sent as a backup; backups are placed in backup_buffer, 
 *	which is an array of doubles (a format easily sent with OpenMPI). The number of backups 
 *	to be sent is returned by the function.
 *	
 *	@param [out] backup_buffer An address of a double pointer to be filled with an array of doubles representing points to be sent as backups.
 *	@param [in] send_info An array of bools used to describe how each input point should be communicated to the other process
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] counters A std::vector<size_t> with only one element, representing the number of giveups to be given away to the other process.
 *	@param [in] myid The rank of the current process in cartesianCOMM
 *	@return The number of backup points to be sent
 */
unsigned int send1(double ** backup_buffer, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid){
	size_t allocated_space_for_backups = counters[0]+20;	// keep track of the amount of space we have allocated for backups
	size_t reallocation_difference_for_backups = 50;	// the number of backups to increase by if capacity is reached
	*backup_buffer = (double *) malloc(allocated_space_for_backups*2*sizeof(double));		// space for backups
	unsigned int spot = 0;	// used to index through send_info
	unsigned int backup_index{0};

	// loop through points and figure out where each one needs to be sent
	for (auto point_it = input_points.cbegin(); point_it != input_points.cend(); ++point_it){
		// most likely the point won't be a giveup...
		if (!send_info[spot*2]){	// if not giveup
			if (!send_info[spot*2+1]){					// not giveup and not backup
				spot++;
				continue;
			} else {									// only backup
				(*backup_buffer)[backup_index] = (*point_it).x;
				(*backup_buffer)[backup_index+1] = (*point_it).y;
				backup_index+=2;
				if (backup_index == allocated_space_for_backups*2){		// if we have reached capacity for backups
					allocated_space_for_backups += reallocation_difference_for_backups;
					*backup_buffer = (double*)realloc((*backup_buffer), allocated_space_for_backups * 2* sizeof(double));	// allocate more space
					
				}
			}
		} 
		spot++;
	}

	if (myid == print_rank) std::cout<<"Rank "<<myid<<": done sorting backup data for sending\n";

	backup_index /= 2;
	return backup_index;
}


/**
 *	\brief This function determines which points to send as backups for a process that 
 *	communicates with two processes sharing a boundary line.
 *
 *	In particular, this function loops through input_points and send_info, determining
 *	for each point if it must be sent as a backup and to which process; backups are added 
 *	to the back of the correct element in backup_buffers, which are arrays of doubles 
 *	(a format easily sent with OpenMPI). The number of backups to be sent to each process
 *	is stored in the corresponding element in backup_index.
 *	
 *	@param [out] backup_buffers An std::vector of double pointers to be filled with arrays of doubles representing points to be sent as backups, corresponding to other processes.
 *	@param [out] backup_index An std::vector of ints to be filled with the number of backups to be sent to each process respectively.
 *	@param [in] send_info An array of bools used to describe how each input point should be communicated to the other process
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] counters A std::vector<size_t> with only one element, representing the number of giveups to be given away to the other process.
 *	@param [in] myid The rank of the current process in cartesianCOMM
 */
void send2(std::vector<double *>& backup_buffers, std::vector<unsigned int>& backup_index, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid){

	std::vector<unsigned int> allocated_space_for_backups = {((unsigned int)counters[0]+30)*2,((unsigned int)counters[1]+30)*2,30};	// keep track of the amount of space we have allocated for backups
	unsigned int reallocation_difference_for_backups = 100;	// the number of backups to increase by if capacity is reached


	backup_buffers[0] = (double *) malloc(allocated_space_for_backups[0]*sizeof(double));	// first side
	backup_buffers[1] = (double *) malloc(allocated_space_for_backups[1]*sizeof(double));	// second side
	backup_buffers[2] = (double *) malloc(allocated_space_for_backups[2]*sizeof(double));					// corner between sides	(although we might not even have a corner...)


	unsigned int spot = 0;	// used to index through send_info

	// loop through points and figure out where each one needs to be sent
	for (auto point_it = input_points.cbegin(); point_it != input_points.cend(); ++point_it){
		// most likely the point won't be a giveup...
		if (!send_info[spot*4]){	// if not giveup to side1
			if (!send_info[spot*4+1]){	// not giveup to side2
				if (!send_info[spot*4+2]){	// not backup to side1
					if (!send_info[spot*4+3]){					// not giveup and not backup
						spot++;
						continue;
					} else {									// only backup to side2
						backup_buffers[1][backup_index[1]] = (*point_it).x;
						backup_buffers[1][backup_index[1]+1] = (*point_it).y;
						backup_index[1]+=2;
						if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
							allocated_space_for_backups[1] += reallocation_difference_for_backups;
							backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*4+3]){				// only backup for side1
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}
				} else {										// side1 and side2 backup (and hence also corner backup)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[1][backup_index[1]] = (*point_it).x;
					backup_buffers[1][backup_index[1]+1] = (*point_it).y;
					backup_index[1]+=2;
					if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
						allocated_space_for_backups[1] += reallocation_difference_for_backups;
						backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[2][backup_index[2]] = (*point_it).x;
					backup_buffers[2][backup_index[2]+1] = (*point_it).y;
					backup_index[2]+=2;
					if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
						allocated_space_for_backups[2] += 10;
						backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
						
					}
				}
			} else if (!send_info[spot*4+2]){					// only giveup to side2
				spot++;
				continue;
			} else {											// side2 giveup and side1 backup (and hence also corner backup)
				backup_buffers[0][backup_index[0]] = (*point_it).x;
				backup_buffers[0][backup_index[0]+1] = (*point_it).y;
				backup_index[0]+=2;
				if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
					allocated_space_for_backups[0] += reallocation_difference_for_backups;
					backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
					
				}

				backup_buffers[2][backup_index[2]] = (*point_it).x;
				backup_buffers[2][backup_index[2]+1] = (*point_it).y;
				backup_index[2]+=2;
				if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
					allocated_space_for_backups[2] += 10;
					backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
					
				}
			}
		} else if (!send_info[spot*4+1]){ // side1 giveup
			if (!send_info[spot*4+3]){							// only side1 giveup
				spot++;
				continue;
			} else {											// side1 giveup and side2 backup (and hence also corner backup)
				backup_buffers[1][backup_index[1]] = (*point_it).x;
				backup_buffers[1][backup_index[1]+1] = (*point_it).y;
				backup_index[1]+=2;
				if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
					allocated_space_for_backups[1] += reallocation_difference_for_backups;
					backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
					
				}

				backup_buffers[2][backup_index[2]] = (*point_it).x;
				backup_buffers[2][backup_index[2]+1] = (*point_it).y;
				backup_index[2]+=2;
				if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
					allocated_space_for_backups[2] += 10;
					backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
					
				}
			}
		} else {												// corner (side1+side2 giveup and side1 backup and side2 backup)
			backup_buffers[0][backup_index[0]] = (*point_it).x;
			backup_buffers[0][backup_index[0]+1] = (*point_it).y;
			backup_index[0]+=2;
			if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
				allocated_space_for_backups[0] += reallocation_difference_for_backups;
				backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
				
			}

			backup_buffers[1][backup_index[1]] = (*point_it).x;
			backup_buffers[1][backup_index[1]+1] = (*point_it).y;
			backup_index[1]+=2;
			if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
				allocated_space_for_backups[1] += reallocation_difference_for_backups;
				backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
				
			}
		}
		spot++;
	}

	if (myid == print_rank) std::cout<<"Rank "<<myid<<": done sorting backup data for sending\n";


	backup_index[0]/=2;
	backup_index[1]/=2;
	backup_index[2]/=2;

	return;
}


/**
 *	\brief This function determines which points to send as backups for a process that 
 *	communicates with two processes sharing a boundary line.
 *
 *	In particular, this function loops through input_points and send_info, determining
 *	for each point if it must be sent as a backup and to which process; backups are added 
 *	to the back of the correct element in backup_buffers, which are arrays of doubles 
 *	(a format easily sent with OpenMPI). The number of backups to be sent to each process
 *	is stored in the corresponding element in backup_index.
 *
 *	@param [out] backup_buffers An std::vector of double pointers to be filled with arrays of doubles representing points to be sent as backups, corresponding to other processes.
 *	@param [out] backup_index An std::vector of ints to be filled with the number of backups to be sent to each process respectively.
 *	@param [in] send_info An array of bools used to describe how each input point should be communicated to the other process
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] counters A std::vector<size_t> with only one element, representing the number of giveups to be given away to the other process.
 *	@param [in] myid The rank of the current process in cartesianCOMM
 *	@param [in] we_send_to_original_copy An array of 8 bools indicating which of the surrounding 8 processes the calling process communicates with.
 */
void send3(std::vector<double *>& backup_buffers, std::vector<unsigned int>& backup_index, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid, bool * we_send_to_original_copy){


	std::vector<unsigned int> allocated_space_for_backups = {((unsigned int)counters[0]+30)*2, ((unsigned int)counters[1]+30)*2, ((unsigned int)counters[2]+30)*2, 30, 30, 30};	// keep track of the amount of space we have allocated for backups
	unsigned int reallocation_difference_for_backups = 100;	// the number of backups to increase by if capacity is reached

	backup_buffers[0] = (double *) malloc(allocated_space_for_backups[0]*sizeof(double));	// first side
	backup_buffers[1] = (double *) malloc(allocated_space_for_backups[1]*sizeof(double));	// second side
	backup_buffers[2] = (double *) malloc(allocated_space_for_backups[2]*sizeof(double));	// third side
	
	backup_buffers[3] = (double *) malloc (allocated_space_for_backups[3]*sizeof(double));		//corner12			// corner between sides		(although we might not even have this corner...)
	backup_buffers[4] = (double *) malloc (allocated_space_for_backups[4]*sizeof(double));		//corner23			// corner between sides		(although we might not even have this corner...)
	backup_buffers[5] = (double *) malloc (allocated_space_for_backups[5]*sizeof(double));		//corner31			// corner between sides		(although we might not even have this corner...)


	unsigned int spot = 0;	// used to index through send_info
	// loop through points and figure out where each one needs to be sent
	for (auto point_it = input_points.cbegin(); point_it != input_points.cend(); ++point_it){
		// most likely the point won't be a giveup...
		if (!send_info[spot*6]){ // not side1 giveup
			if (!send_info[spot*6+1]){ // not side2 giveup
				if (!send_info[spot*6+2]){ // not side3 giveup
					if (!send_info[spot*6+3]){ // not side1 backup
						if (!send_info[spot*6+4]){	// not side2 backup
							if (!send_info[spot*6+5]){								// not giveup or backup
								spot++;
								continue;
							} else {												// side3 backup
								backup_buffers[2][backup_index[2]] = (*point_it).x;
								backup_buffers[2][backup_index[2]+1] = (*point_it).y;
								backup_index[2]+=2;
								if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
									allocated_space_for_backups[2] += reallocation_difference_for_backups;
									backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
									
								}
							}
						} else if (!send_info[spot*6+5]){							// side 2 backup
							backup_buffers[1][backup_index[1]] = (*point_it).x;
							backup_buffers[1][backup_index[1]+1] = (*point_it).y;
							backup_index[1]+=2;
							if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
								allocated_space_for_backups[1] += reallocation_difference_for_backups;
								backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
								
							}
						} else {													// side 2 and side 3 backup (hence corner23 backup)
							backup_buffers[1][backup_index[1]] = (*point_it).x;
							backup_buffers[1][backup_index[1]+1] = (*point_it).y;
							backup_index[1]+=2;
							if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
								allocated_space_for_backups[1] += reallocation_difference_for_backups;
								backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[2][backup_index[2]] = (*point_it).x;
							backup_buffers[2][backup_index[2]+1] = (*point_it).y;
							backup_index[2]+=2;
							if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
								allocated_space_for_backups[2] += reallocation_difference_for_backups;
								backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[4][backup_index[4]] = (*point_it).x;
							backup_buffers[4][backup_index[4]+1] = (*point_it).y;
							backup_index[4]+=2;
							if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
								allocated_space_for_backups[4] += 10;
								backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
								
							}
						}
					} else if (!send_info[spot*6+4]){ // not side2 backup
						if (!send_info[spot*6+5]){									// side1 backup
							backup_buffers[0][backup_index[0]] = (*point_it).x;
							backup_buffers[0][backup_index[0]+1] = (*point_it).y;
							backup_index[0]+=2;
							if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
								allocated_space_for_backups[0] += reallocation_difference_for_backups;
								backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
								
							}
						} else {													// side1 and side3 backup (hence corner31 backup)
							backup_buffers[0][backup_index[0]] = (*point_it).x;
							backup_buffers[0][backup_index[0]+1] = (*point_it).y;
							backup_index[0]+=2;
							if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
								allocated_space_for_backups[0] += reallocation_difference_for_backups;
								backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[2][backup_index[2]] = (*point_it).x;
							backup_buffers[2][backup_index[2]+1] = (*point_it).y;
							backup_index[2]+=2;
							if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
								allocated_space_for_backups[2] += reallocation_difference_for_backups;
								backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[5][backup_index[5]] = (*point_it).x;
							backup_buffers[5][backup_index[5]+1] = (*point_it).y;
							backup_index[5]+=2;
							if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
								allocated_space_for_backups[5] += 10;
								backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
								
							}
						}
					} else {														// side1 and side2 backup (hence corner12 backup)
						backup_buffers[0][backup_index[0]] = (*point_it).x;
						backup_buffers[0][backup_index[0]+1] = (*point_it).y;
						backup_index[0]+=2;
						if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
							allocated_space_for_backups[0] += reallocation_difference_for_backups;
							backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[1][backup_index[1]] = (*point_it).x;
						backup_buffers[1][backup_index[1]+1] = (*point_it).y;
						backup_index[1]+=2;
						if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
							allocated_space_for_backups[1] += reallocation_difference_for_backups;
							backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[3][backup_index[3]] = (*point_it).x;
						backup_buffers[3][backup_index[3]+1] = (*point_it).y;
						backup_index[3]+=2;
						if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
							allocated_space_for_backups[3] += 10;
							backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*6+3]){ // not side1 backup
					if (!send_info[spot*6+4]){										// side3 giveup
						spot++;
						continue;
					} else {														// side3 giveup and side2 backup (hence corner23 backup)
						backup_buffers[1][backup_index[1]] = (*point_it).x;
						backup_buffers[1][backup_index[1]+1] = (*point_it).y;
						backup_index[1]+=2;
						if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
							allocated_space_for_backups[1] += reallocation_difference_for_backups;
							backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[4][backup_index[4]] = (*point_it).x;
						backup_buffers[4][backup_index[4]+1] = (*point_it).y;
						backup_index[4]+=2;
						if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
							allocated_space_for_backups[4] += 10;
							backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*6+4]){									// side3 giveup and side1 backup (hence corner31 backup)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[5][backup_index[5]] = (*point_it).x;
					backup_buffers[5][backup_index[5]+1] = (*point_it).y;
					backup_index[5]+=2;
					if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
						allocated_space_for_backups[5] += 10;
						backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
						
					}
				} else {															// side3 giveup and side1 backup and side2 backup (hence corners backups)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[1][backup_index[1]] = (*point_it).x;
					backup_buffers[1][backup_index[1]+1] = (*point_it).y;
					backup_index[1]+=2;
					if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
						allocated_space_for_backups[1] += reallocation_difference_for_backups;
						backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += 10;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[5][backup_index[5]] = (*point_it).x;
					backup_buffers[5][backup_index[5]+1] = (*point_it).y;
					backup_index[5]+=2;
					if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
						allocated_space_for_backups[5] += 10;
						backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
						
					}
				}
			} else if (!send_info[spot*6+2]){ // not side3 giveup
				if (!send_info[spot*6+3]){ // not side1 backup
					if (!send_info[spot*6+3+5]){									// side2 giveup
						spot++;
						continue;
					} else {														// side2 giveup and side3 backup (hence corner23 backup)
						backup_buffers[2][backup_index[2]] = (*point_it).x;
						backup_buffers[2][backup_index[2]+1] = (*point_it).y;
						backup_index[2]+=2;
						if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
							allocated_space_for_backups[2] += reallocation_difference_for_backups;
							backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[4][backup_index[4]] = (*point_it).x;
						backup_buffers[4][backup_index[4]+1] = (*point_it).y;
						backup_index[4]+=2;
						if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
							allocated_space_for_backups[4] += 10;
							backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*6+5]){									// side2 giveup and side1 backup (hence corner12 backup)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += 10;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}
				} else {															// side2 giveup and side1 backup and side3 backup (hence corners backups)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[2][backup_index[2]] = (*point_it).x;
					backup_buffers[2][backup_index[2]+1] = (*point_it).y;
					backup_index[2]+=2;
					if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
						allocated_space_for_backups[2] += reallocation_difference_for_backups;
						backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += 10;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[5][backup_index[5]] = (*point_it).x;
					backup_buffers[5][backup_index[5]+1] = (*point_it).y;
					backup_index[5]+=2;
					if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
						allocated_space_for_backups[5] += 10;
						backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
						
					}
				}
			} else {																// giveup to side2 and side3 (thus corner giveup) and backups to 2 and 3
				backup_buffers[1][backup_index[1]] = (*point_it).x;
				backup_buffers[1][backup_index[1]+1] = (*point_it).y;
				backup_index[1]+=2;
				if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
					allocated_space_for_backups[1] += reallocation_difference_for_backups;
					backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
					
				}

				backup_buffers[2][backup_index[2]] = (*point_it).x;
				backup_buffers[2][backup_index[2]+1] = (*point_it).y;
				backup_index[2]+=2;
				if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
					allocated_space_for_backups[2] += reallocation_difference_for_backups;
					backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
					
				}
			}
		} else if (!send_info[spot*6+1]){ // side1 giveup
			if (!send_info[spot*6+2]){ // not side2 or side3 givecup
				if (!send_info[spot*6+4]){ // not side2 backup
					if (!send_info[spot*6+5]){										// side1 giveup
						spot++;
						continue;
					} else {														// side1 giveup and side3 backup (hence corner31 backup)
						backup_buffers[2][backup_index[2]] = (*point_it).x;
						backup_buffers[2][backup_index[2]+1] = (*point_it).y;
						backup_index[2]+=2;
						if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
							allocated_space_for_backups[2] += reallocation_difference_for_backups;
							backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[5][backup_index[5]] = (*point_it).x;
						backup_buffers[5][backup_index[5]+1] = (*point_it).y;
						backup_index[5]+=2;
						if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
							allocated_space_for_backups[5] += 10;
							backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*6+5]){									// side1 giveup and side2 backup (hence corner12 backup)
					backup_buffers[1][backup_index[1]] = (*point_it).x;
					backup_buffers[1][backup_index[1]+1] = (*point_it).y;
					backup_index[1]+=2;
					if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
						allocated_space_for_backups[1] += reallocation_difference_for_backups;
						backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += 10;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}
				} else {															// side1 giveup and side2 backup and side3 backup (hence corners backup)
					backup_buffers[1][backup_index[1]] = (*point_it).x;
					backup_buffers[1][backup_index[1]+1] = (*point_it).y;
					backup_index[1]+=2;
					if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
						allocated_space_for_backups[1] += reallocation_difference_for_backups;
						backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[2][backup_index[2]] = (*point_it).x;
					backup_buffers[2][backup_index[2]+1] = (*point_it).y;
					backup_index[2]+=2;
					if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
						allocated_space_for_backups[2] += reallocation_difference_for_backups;
						backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += 10;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[5][backup_index[5]] = (*point_it).x;
					backup_buffers[5][backup_index[5]+1] = (*point_it).y;
					backup_index[5]+=2;
					if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
						allocated_space_for_backups[5] += 10;
						backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
						
					}
				}
			} else {																	// side1 and side3 giveup (thus the corner and backups)
				backup_buffers[0][backup_index[0]] = (*point_it).x;
				backup_buffers[0][backup_index[0]+1] = (*point_it).y;
				backup_index[0]+=2;
				if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
					allocated_space_for_backups[0] += reallocation_difference_for_backups;
					backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
					
				}

				backup_buffers[2][backup_index[2]] = (*point_it).x;
				backup_buffers[2][backup_index[2]+1] = (*point_it).y;
				backup_index[2]+=2;
				if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
					allocated_space_for_backups[2] += reallocation_difference_for_backups;
					backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
					
				}
			}
		} else {																	// side1 and side2 giveup (thus the corner and backups)
			backup_buffers[0][backup_index[0]] = (*point_it).x;
			backup_buffers[0][backup_index[0]+1] = (*point_it).y;
			backup_index[0]+=2;
			if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
				allocated_space_for_backups[0] += reallocation_difference_for_backups;
				backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
				
			}

			backup_buffers[1][backup_index[1]] = (*point_it).x;
			backup_buffers[1][backup_index[1]+1] = (*point_it).y;
			backup_index[1]+=2;
			if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
				allocated_space_for_backups[1] += reallocation_difference_for_backups;
				backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
				
			}
		}
		spot++;
	}


	// Now we need to get rid of the backup data that we actually don't need....
	// one of these corners is not a corner...
	if (!we_send_to_original_copy[0]){						// not corner23
		free(backup_buffers[4]);
		backup_buffers.erase(backup_buffers.begin() + 4);
		backup_index.erase(backup_index.begin()+4);
	} else if (!we_send_to_original_copy[1]){ 				// not corner23
		free(backup_buffers[4]);
		backup_buffers.erase(backup_buffers.begin() + 4);
		backup_index.erase(backup_index.begin()+4);
	} else if (!we_send_to_original_copy[2]){				// not corner12
		free(backup_buffers[3]);
		backup_buffers.erase(backup_buffers.begin() + 3);
		backup_index.erase(backup_index.begin()+3);

		// need to flip last two backup buffers and index to get the proper order
		unsigned int temp = backup_index[3];
		backup_index[3] = backup_index[4];
		backup_index[4] = temp;

		double * temp_ptr = backup_buffers[3];
		backup_buffers[3] = backup_buffers[4];
		backup_buffers[4] = temp_ptr;
	} else {												// not corner12
		free(backup_buffers[3]);
		backup_buffers.erase(backup_buffers.begin() + 3);
		backup_index.erase(backup_index.begin()+3);

		// need to flip last two backup buffers and index to get the proper order
		unsigned int temp = backup_index[3];
		backup_index[3] = backup_index[4];
		backup_index[4] = temp;

		double * temp_ptr = backup_buffers[3];
		backup_buffers[3] = backup_buffers[4];
		backup_buffers[4] = temp_ptr;
	}


	if (myid == print_rank) std::cout<<"Rank "<<myid<<": done sorting backup data for sending\n";


	backup_index[0]/=2;
	backup_index[1]/=2;
	backup_index[2]/=2;
	backup_index[3]/=2;
	backup_index[4]/=2;

	return;
}


/**
 *	\brief This function determines which points to send as backups for a process that 
 *	communicates with two processes sharing a boundary line.
 *
 *	In particular, this function loops through input_points and send_info, determining
 *	for each point if it must be sent as a backup and to which process; backups are added 
 *	to the back of the correct element in backup_buffers, which are arrays of doubles 
 *	(a format easily sent with OpenMPI). The number of backups to be sent to each process
 *	is stored in the corresponding element in backup_index.
 *	
 *	@param [in] giveup_buffers An std::vector of arrays of doubles representing points which are "giveups" to the other corresponding process.
 *	@param [out] backup_buffers An std::vector of double pointers to be filled with arrays of doubles representing points to be sent as backups, corresponding to other processes.
 *	@param [out] backup_index An std::vector of ints to be filled with the number of backups to be sent to each process respectively.
 *	@param [in] send_info An array of bools used to describe how each input point should be communicated to the other process
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] counters A std::vector<size_t> with only one element, representing the number of giveups to be given away to the other process.
 *	@param [in] cells The struct holding cell information for this process
 *	@param [in] myid The rank of the current process in cartesianCOMM
 */
void send4(std::vector<double *>& backup_buffers, std::vector<unsigned int>& backup_index, bool * send_info, std::vector<point>& input_points, std::vector<size_t>& counters, int myid){


			//		6______3____ 7
			//		 |			|
			//		0|			| 1
			//		 |			|
			//		 |__________|
			//		 4	  2		 5

	std::vector<unsigned int> allocated_space_for_backups = {((unsigned int)counters[0]+30)*2, ((unsigned int)counters[1]+30)*2, ((unsigned int)counters[2]+30)*2, ((unsigned int)counters[3]+30)*2, 30, 30, 30, 30};	// keep track of the amount of space we have allocated for backups
	unsigned int reallocation_difference_for_backups = 100;	// the number of backups to increase by if capacity is reached


	backup_buffers[0] = (double *) malloc(allocated_space_for_backups[0]*sizeof(double));	// first side
	backup_buffers[1] = (double *) malloc(allocated_space_for_backups[1]*sizeof(double));	// second side
	backup_buffers[2] = (double *) malloc(allocated_space_for_backups[2]*sizeof(double));	// third side
	backup_buffers[3] = (double *) malloc(allocated_space_for_backups[3]*sizeof(double));	// fourth side
	
	backup_buffers[4] = (double *) malloc (allocated_space_for_backups[4]*sizeof(double));		// corner between sides
	backup_buffers[5] = (double *) malloc (allocated_space_for_backups[5]*sizeof(double));		// corner between sides
	backup_buffers[6] = (double *) malloc (allocated_space_for_backups[6]*sizeof(double));		// corner between sides
	backup_buffers[7] = (double *) malloc (allocated_space_for_backups[7]*sizeof(double));		// corner between sides


	unsigned int spot = 0;	// used to index through send_info
	// loop through points and figure out where each one needs to be sent
	for (auto point_it = input_points.cbegin(); point_it != input_points.cend(); ++point_it){
		// most likely the point won't be a giveup...
		if (!send_info[spot*8]){	// not side0 giveup
			if (!send_info[spot*8+1]){	// not side1 giveup
				if (!send_info[spot*8]+2){	// not side2 giveup
					if (!send_info[spot*8+3]){	// not side3 giveup
						if (!send_info[spot*8+4]){	// not side0 backup
							if (!send_info[spot*8+5]){	// not side1 backup
								if (!send_info[spot*8+6]){	// not side2 backup
									if (!send_info[spot*8+7]){									// not giveup or backup
										spot++;
										continue;
									} else {													// backup for side3
										backup_buffers[3][backup_index[3]] = (*point_it).x;
										backup_buffers[3][backup_index[3]+1] = (*point_it).y;
										backup_index[3]+=2;
										if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
											allocated_space_for_backups[3] += reallocation_difference_for_backups;
											backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
											
										}
									}
								} else if (!send_info[spot*8+7]){								// backup for side2
									backup_buffers[2][backup_index[2]] = (*point_it).x;
									backup_buffers[2][backup_index[2]+1] = (*point_it).y;
									backup_index[2]+=2;
									if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
										allocated_space_for_backups[2] += reallocation_difference_for_backups;
										backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
										
									}
								} else {														// backup for side2 and side3
									backup_buffers[2][backup_index[2]] = (*point_it).x;
									backup_buffers[2][backup_index[2]+1] = (*point_it).y;
									backup_index[2]+=2;
									if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
										allocated_space_for_backups[2] += reallocation_difference_for_backups;
										backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
										
									}

									backup_buffers[3][backup_index[3]] = (*point_it).x;
									backup_buffers[3][backup_index[3]+1] = (*point_it).y;
									backup_index[3]+=2;
									if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
										allocated_space_for_backups[3] += reallocation_difference_for_backups;
										backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
										
									}
								}
							} else if (!send_info[spot*8+6]){	// backup for side1
								if (!send_info[spot*8+7]){										// backup for side1
									backup_buffers[1][backup_index[1]] = (*point_it).x;
									backup_buffers[1][backup_index[1]+1] = (*point_it).y;
									backup_index[1]+=2;
									if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
										allocated_space_for_backups[1] += reallocation_difference_for_backups;
										backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
										
									}
								} else {														// backup for side1 and side3 (hence also backup for corner7)
									backup_buffers[1][backup_index[1]] = (*point_it).x;
									backup_buffers[1][backup_index[1]+1] = (*point_it).y;
									backup_index[1]+=2;
									if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
										allocated_space_for_backups[1] += reallocation_difference_for_backups;
										backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
										
									}

									backup_buffers[3][backup_index[3]] = (*point_it).x;
									backup_buffers[3][backup_index[3]+1] = (*point_it).y;
									backup_index[3]+=2;
									if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
										allocated_space_for_backups[3] += reallocation_difference_for_backups;
										backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
										
									}

									backup_buffers[7][backup_index[7]] = (*point_it).x;
									backup_buffers[7][backup_index[7]+1] = (*point_it).y;
									backup_index[7]+=2;
									if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
										allocated_space_for_backups[7] += 10;
										backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
										
									}
								}
							} else if (!send_info[spot*8+7]){									// backup for side1 and side2 (hence also backup for corner5)
								backup_buffers[1][backup_index[1]] = (*point_it).x;
								backup_buffers[1][backup_index[1]+1] = (*point_it).y;
								backup_index[1]+=2;
								if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
									allocated_space_for_backups[1] += reallocation_difference_for_backups;
									backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[2][backup_index[2]] = (*point_it).x;
								backup_buffers[2][backup_index[2]+1] = (*point_it).y;
								backup_index[2]+=2;
								if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
									allocated_space_for_backups[2] += reallocation_difference_for_backups;
									backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[5][backup_index[5]] = (*point_it).x;
								backup_buffers[5][backup_index[5]+1] = (*point_it).y;
								backup_index[5]+=2;
								if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
									allocated_space_for_backups[5] += 10;
									backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
									
								}
							} else {															// backup for side1 and side2 and side3  (hence also backup for corner5 and corner7)
								backup_buffers[1][backup_index[1]] = (*point_it).x;
								backup_buffers[1][backup_index[1]+1] = (*point_it).y;
								backup_index[1]+=2;
								if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
									allocated_space_for_backups[1] += reallocation_difference_for_backups;
									backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[2][backup_index[2]] = (*point_it).x;
								backup_buffers[2][backup_index[2]+1] = (*point_it).y;
								backup_index[2]+=2;
								if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
									allocated_space_for_backups[2] += reallocation_difference_for_backups;
									backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[3][backup_index[3]] = (*point_it).x;
								backup_buffers[3][backup_index[3]+1] = (*point_it).y;
								backup_index[3]+=2;
								if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
									allocated_space_for_backups[3] += reallocation_difference_for_backups;
									backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[5][backup_index[5]] = (*point_it).x;
								backup_buffers[5][backup_index[5]+1] = (*point_it).y;
								backup_index[5]+=2;
								if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
									allocated_space_for_backups[5] += 10;
									backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[7][backup_index[7]] = (*point_it).x;
								backup_buffers[7][backup_index[7]+1] = (*point_it).y;
								backup_index[7]+=2;
								if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
									allocated_space_for_backups[7] += 10;
									backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
									
								}
							}
						} else if (!send_info[spot*8+5]){	// backup for side0
							if (!send_info[spot*8+6]) {
								if (!send_info[spot*8+7]){										// backup for side0
									backup_buffers[0][backup_index[0]] = (*point_it).x;
									backup_buffers[0][backup_index[0]+1] = (*point_it).y;
									backup_index[0]+=2;
									if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
										allocated_space_for_backups[0] += reallocation_difference_for_backups;
										backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
										
									}
								} else {														// backup for side0 and side3 (hence also backup for corner6)
									backup_buffers[0][backup_index[0]] = (*point_it).x;
									backup_buffers[0][backup_index[0]+1] = (*point_it).y;
									backup_index[0]+=2;
									if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
										allocated_space_for_backups[0] += reallocation_difference_for_backups;
										backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
										
									}

									backup_buffers[3][backup_index[3]] = (*point_it).x;
									backup_buffers[3][backup_index[3]+1] = (*point_it).y;
									backup_index[3]+=2;
									if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
										allocated_space_for_backups[3] += reallocation_difference_for_backups;
										backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
										
									}

									backup_buffers[6][backup_index[6]] = (*point_it).x;
									backup_buffers[6][backup_index[6]+1] = (*point_it).y;
									backup_index[6]+=2;
									if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
										allocated_space_for_backups[6] += 10;
										backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
										
									}
								}
							} else if (!send_info[spot*8+7]){									// backup for side0 and side2 (hence also backup for corner4)
								backup_buffers[0][backup_index[0]] = (*point_it).x;
								backup_buffers[0][backup_index[0]+1] = (*point_it).y;
								backup_index[0]+=2;
								if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
									allocated_space_for_backups[0] += reallocation_difference_for_backups;
									backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[2][backup_index[2]] = (*point_it).x;
								backup_buffers[2][backup_index[2]+1] = (*point_it).y;
								backup_index[2]+=2;
								if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
									allocated_space_for_backups[2] += reallocation_difference_for_backups;
									backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[4][backup_index[4]] = (*point_it).x;
								backup_buffers[4][backup_index[4]+1] = (*point_it).y;
								backup_index[4]+=2;
								if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
									allocated_space_for_backups[4] += 10;
									backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
									
								}
							} else {															// backup for side0 and side2 and side3  (hence also backup for corner4 and corner6)
								backup_buffers[0][backup_index[0]] = (*point_it).x;
								backup_buffers[0][backup_index[0]+1] = (*point_it).y;
								backup_index[0]+=2;
								if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
									allocated_space_for_backups[0] += reallocation_difference_for_backups;
									backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[2][backup_index[2]] = (*point_it).x;
								backup_buffers[2][backup_index[2]+1] = (*point_it).y;
								backup_index[2]+=2;
								if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
									allocated_space_for_backups[2] += reallocation_difference_for_backups;
									backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[3][backup_index[3]] = (*point_it).x;
								backup_buffers[3][backup_index[3]+1] = (*point_it).y;
								backup_index[3]+=2;
								if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
									allocated_space_for_backups[3] += reallocation_difference_for_backups;
									backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[4][backup_index[4]] = (*point_it).x;
								backup_buffers[4][backup_index[4]+1] = (*point_it).y;
								backup_index[4]+=2;
								if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
									allocated_space_for_backups[4] += 10;
									backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[6][backup_index[6]] = (*point_it).x;
								backup_buffers[6][backup_index[6]+1] = (*point_it).y;
								backup_index[6]+=2;
								if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
									allocated_space_for_backups[6] += 10;
									backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
									
								}
							}
						} else if (!send_info[spot*8+6]){										// backup for side0 and side1
							if (!send_info[spot*8+7]){
								backup_buffers[0][backup_index[0]] = (*point_it).x;
								backup_buffers[0][backup_index[0]+1] = (*point_it).y;
								backup_index[0]+=2;
								if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
									allocated_space_for_backups[0] += reallocation_difference_for_backups;
									backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[1][backup_index[1]] = (*point_it).x;
								backup_buffers[1][backup_index[1]+1] = (*point_it).y;
								backup_index[1]+=2;
								if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
									allocated_space_for_backups[1] += reallocation_difference_for_backups;
									backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
									
								}
							} else {															// backup for side0 and side1 and side3 (hence also backup for corner6 and corner7)
								backup_buffers[0][backup_index[0]] = (*point_it).x;
								backup_buffers[0][backup_index[0]+1] = (*point_it).y;
								backup_index[0]+=2;
								if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
									allocated_space_for_backups[0] += reallocation_difference_for_backups;
									backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[1][backup_index[1]] = (*point_it).x;
								backup_buffers[1][backup_index[1]+1] = (*point_it).y;
								backup_index[1]+=2;
								if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
									allocated_space_for_backups[1] += reallocation_difference_for_backups;
									backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[3][backup_index[3]] = (*point_it).x;
								backup_buffers[3][backup_index[3]+1] = (*point_it).y;
								backup_index[3]+=2;
								if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
									allocated_space_for_backups[3] += reallocation_difference_for_backups;
									backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[6][backup_index[6]] = (*point_it).x;
								backup_buffers[6][backup_index[6]+1] = (*point_it).y;
								backup_index[6]+=2;
								if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
									allocated_space_for_backups[6] += 10;
									backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
									
								}

								backup_buffers[7][backup_index[7]] = (*point_it).x;
								backup_buffers[7][backup_index[7]+1] = (*point_it).y;
								backup_index[7]+=2;
								if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
									allocated_space_for_backups[7] += 10;
									backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
									
								}
							}
						} else if (!send_info[spot*8+7]){										// backup for side0 and side1 and side2 (hence also backup for corner4 and corner5)
							backup_buffers[0][backup_index[0]] = (*point_it).x;
							backup_buffers[0][backup_index[0]+1] = (*point_it).y;
							backup_index[0]+=2;
							if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
									allocated_space_for_backups[0] += reallocation_difference_for_backups;
									backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
									
								}

							backup_buffers[1][backup_index[1]] = (*point_it).x;
							backup_buffers[1][backup_index[1]+1] = (*point_it).y;
							backup_index[1]+=2;
							if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
								allocated_space_for_backups[1] += reallocation_difference_for_backups;
								backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[2][backup_index[2]] = (*point_it).x;
							backup_buffers[2][backup_index[2]+1] = (*point_it).y;
							backup_index[2]+=2;
							if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
								allocated_space_for_backups[2] += reallocation_difference_for_backups;
								backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[4][backup_index[4]] = (*point_it).x;
							backup_buffers[4][backup_index[4]+1] = (*point_it).y;
							backup_index[4]+=2;
							if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
								allocated_space_for_backups[4] += 10;
								backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[5][backup_index[5]] = (*point_it).x;
							backup_buffers[5][backup_index[5]+1] = (*point_it).y;
							backup_index[5]+=2;
							if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
								allocated_space_for_backups[5] += 10;
								backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
								
							}

						} else {																// backup for everything...  (hence also backup for all corner)
							backup_buffers[0][backup_index[0]] = (*point_it).x;
							backup_buffers[0][backup_index[0]+1] = (*point_it).y;
							backup_index[0]+=2;
							if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
								allocated_space_for_backups[0] += reallocation_difference_for_backups;
								backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[1][backup_index[1]] = (*point_it).x;
							backup_buffers[1][backup_index[1]+1] = (*point_it).y;
							backup_index[1]+=2;
							if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
								allocated_space_for_backups[1] += reallocation_difference_for_backups;
								backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[2][backup_index[2]] = (*point_it).x;
							backup_buffers[2][backup_index[2]+1] = (*point_it).y;
							backup_index[2]+=2;
							if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
								allocated_space_for_backups[2] += reallocation_difference_for_backups;
								backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[3][backup_index[3]] = (*point_it).x;
							backup_buffers[3][backup_index[3]+1] = (*point_it).y;
							backup_index[3]+=2;
							if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
								allocated_space_for_backups[3] += reallocation_difference_for_backups;
								backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[4][backup_index[4]] = (*point_it).x;
							backup_buffers[4][backup_index[4]+1] = (*point_it).y;
							backup_index[4]+=2;
							if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
								allocated_space_for_backups[4] += 10;
								backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[5][backup_index[5]] = (*point_it).x;
							backup_buffers[5][backup_index[5]+1] = (*point_it).y;
							backup_index[5]+=2;
							if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
								allocated_space_for_backups[5] += 10;
								backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[6][backup_index[6]] = (*point_it).x;
							backup_buffers[6][backup_index[6]+1] = (*point_it).y;
							backup_index[6]+=2;
							if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
								allocated_space_for_backups[6] += 10;
								backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[7][backup_index[7]] = (*point_it).x;
							backup_buffers[7][backup_index[7]+1] = (*point_it).y;
							backup_index[7]+=2;
							if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
								allocated_space_for_backups[7] += 10;
								backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
								
							}
						}
					} else if (!send_info[spot*8+4]){	// side3 giveup
						if (!send_info[spot*8+5]){												// side3 giveup (we assume not side2 backup)
							spot++;
							continue;
						} else {																// side3 giveup and side1 backup  (hence also backup for corner7)
							backup_buffers[1][backup_index[1]] = (*point_it).x;
							backup_buffers[1][backup_index[1]+1] = (*point_it).y;
							backup_index[1]+=2;
							if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
								allocated_space_for_backups[1] += reallocation_difference_for_backups;
								backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[7][backup_index[7]] = (*point_it).x;
							backup_buffers[7][backup_index[7]+1] = (*point_it).y;
							backup_index[7]+=2;
							if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
								allocated_space_for_backups[7] += 10;
								backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
								
							}
						}
					} else if (!send_info[spot*8+5]){											// side3 giveup and side0 backup (hence also backup for corner6)
						backup_buffers[0][backup_index[0]] = (*point_it).x;
						backup_buffers[0][backup_index[0]+1] = (*point_it).y;
						backup_index[0]+=2;
						if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
							allocated_space_for_backups[0] += reallocation_difference_for_backups;
							backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[6][backup_index[6]] = (*point_it).x;
						backup_buffers[6][backup_index[6]+1] = (*point_it).y;
						backup_index[6]+=2;
						if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
							allocated_space_for_backups[6] += 10;
							backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
							
						}
					} else {																	// side3 giveup and side0 backup and side1 backup (hence also backup for corner6 and corner7)
						backup_buffers[0][backup_index[0]] = (*point_it).x;
						backup_buffers[0][backup_index[0]+1] = (*point_it).y;
						backup_index[0]+=2;
						if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
							allocated_space_for_backups[0] += reallocation_difference_for_backups;
							backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[1][backup_index[1]] = (*point_it).x;
						backup_buffers[1][backup_index[1]+1] = (*point_it).y;
						backup_index[1]+=2;
						if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
							allocated_space_for_backups[1] += reallocation_difference_for_backups;
							backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[6][backup_index[6]] = (*point_it).x;
						backup_buffers[6][backup_index[6]+1] = (*point_it).y;
						backup_index[6]+=2;
						if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
							allocated_space_for_backups[6] += 10;
							backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[7][backup_index[7]] = (*point_it).x;
						backup_buffers[7][backup_index[7]+1] = (*point_it).y;
						backup_index[7]+=2;
						if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
							allocated_space_for_backups[7] += 10;
							backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*8+4]){
					if (!send_info[spot*8+5]){												// side2 giveup (we assume no side3 backup)(we also assume not side3 giveup)
						spot++;
						continue;
					} else {																// side2 giveup and side1 backup  (hence also backup for corner5)
						backup_buffers[1][backup_index[1]] = (*point_it).x;
						backup_buffers[1][backup_index[1]+1] = (*point_it).y;
						backup_index[1]+=2;
						if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
							allocated_space_for_backups[1] += reallocation_difference_for_backups;
							backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[5][backup_index[5]] = (*point_it).x;
						backup_buffers[5][backup_index[5]+1] = (*point_it).y;
						backup_index[5]+=2;
						if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
							allocated_space_for_backups[5] += 10;
							backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*8+5]){											// side2 giveup and side0 backup (hence also backup for corner4)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}
				} else {																	// side2 giveup and side0 backup and side1 backup (hence also backup for corner4 and corner5)
					backup_buffers[0][backup_index[0]] = (*point_it).x;
					backup_buffers[0][backup_index[0]+1] = (*point_it).y;
					backup_index[0]+=2;
					if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
						allocated_space_for_backups[0] += reallocation_difference_for_backups;
						backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[1][backup_index[1]] = (*point_it).x;
					backup_buffers[1][backup_index[1]+1] = (*point_it).y;
					backup_index[1]+=2;
					if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
						allocated_space_for_backups[1] += reallocation_difference_for_backups;
						backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[5][backup_index[5]] = (*point_it).x;
					backup_buffers[5][backup_index[5]+1] = (*point_it).y;
					backup_index[5]+=2;
					if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
						allocated_space_for_backups[5] += 10;
						backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
						
					}
				}
			} else if (!send_info[spot*8+2]){	// side1 giveup
				if (!send_info[spot*8+3]){
					if (!send_info[spot*8+6]){
						if (!send_info[spot*8+7]){											// side1 giveup (we assume no side0 backup)
							spot++;
							continue;
						} else {															// side1 giveup and side3 backup  (hence also backup for corner7)
							backup_buffers[3][backup_index[3]] = (*point_it).x;
							backup_buffers[3][backup_index[3]+1] = (*point_it).y;
							backup_index[3]+=2;
							if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
								allocated_space_for_backups[3] += reallocation_difference_for_backups;
								backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
								
							}

							backup_buffers[7][backup_index[7]] = (*point_it).x;
							backup_buffers[7][backup_index[7]+1] = (*point_it).y;
							backup_index[7]+=2;
							if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
								allocated_space_for_backups[7] += 10;
								backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
								
							}
						}
					} else if (!send_info[spot*8+7]){										// side1 giveup and side2 backup  (hence also backup for corner5)
						backup_buffers[2][backup_index[2]] = (*point_it).x;
						backup_buffers[2][backup_index[2]+1] = (*point_it).y;
						backup_index[2]+=2;
						if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
							allocated_space_for_backups[2] += reallocation_difference_for_backups;
							backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[5][backup_index[5]] = (*point_it).x;
						backup_buffers[5][backup_index[5]+1] = (*point_it).y;
						backup_index[5]+=2;
						if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
							allocated_space_for_backups[5] += 10;
							backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
							
						}
					} else {																// side1 giveup and side2 backup and side3 backup  (hence also backup for corner5 and corner7)
						backup_buffers[2][backup_index[2]] = (*point_it).x;
						backup_buffers[2][backup_index[2]+1] = (*point_it).y;
						backup_index[2]+=2;
						if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
							allocated_space_for_backups[2] += reallocation_difference_for_backups;
							backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[5][backup_index[5]] = (*point_it).x;
						backup_buffers[5][backup_index[5]+1] = (*point_it).y;
						backup_index[5]+=2;
						if (backup_index[5] == allocated_space_for_backups[5]){		// if we have reached capacity for backups
							allocated_space_for_backups[5] += 10;
							backup_buffers[5] = (double*)realloc(backup_buffers[5], allocated_space_for_backups[5] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[3][backup_index[3]] = (*point_it).x;
						backup_buffers[3][backup_index[3]+1] = (*point_it).y;
						backup_index[3]+=2;
						if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
							allocated_space_for_backups[3] += reallocation_difference_for_backups;
							backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[7][backup_index[7]] = (*point_it).x;
						backup_buffers[7][backup_index[7]+1] = (*point_it).y;
						backup_index[7]+=2;
						if (backup_index[7] == allocated_space_for_backups[7]){		// if we have reached capacity for backups
							allocated_space_for_backups[7] += 10;
							backup_buffers[7] = (double*)realloc(backup_buffers[7], allocated_space_for_backups[7] * sizeof(double));	// allocate more space
							
						}
					}
				} else {																	// side1 and side3 corner giveup (hence backup for side1 and side3)
					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += reallocation_difference_for_backups;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[1][backup_index[1]] = (*point_it).x;
					backup_buffers[1][backup_index[1]+1] = (*point_it).y;
					backup_index[1]+=2;
					if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
						allocated_space_for_backups[1] += reallocation_difference_for_backups;
						backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
						
					}
				}
			} else {																		// side1 and side2 corner giveup (hence backup for side1 and side2)
				backup_buffers[1][backup_index[1]] = (*point_it).x;
				backup_buffers[1][backup_index[1]+1] = (*point_it).y;
				backup_index[1]+=2;
				if (backup_index[1] == allocated_space_for_backups[1]){		// if we have reached capacity for backups
					allocated_space_for_backups[1] += reallocation_difference_for_backups;
					backup_buffers[1] = (double*)realloc(backup_buffers[1], allocated_space_for_backups[1] * sizeof(double));	// allocate more space
					
				}

				backup_buffers[2][backup_index[2]] = (*point_it).x;
				backup_buffers[2][backup_index[2]+1] = (*point_it).y;
				backup_index[2]+=2;
				if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
					allocated_space_for_backups[2] += reallocation_difference_for_backups;
					backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
					
				}
			}
		} else if (!send_info[spot*8+2]) {	// side0 giveup
			if (!send_info[spot*8+3]){
				if (!send_info[spot*8+6]){
					if (!send_info[spot*8+7]){												// side0 giveup
						spot++;
						continue;
					} else {																// side0 giveup and side3 backup  (hence also backup for corner6)
						backup_buffers[3][backup_index[3]] = (*point_it).x;
						backup_buffers[3][backup_index[3]+1] = (*point_it).y;
						backup_index[3]+=2;
						if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
							allocated_space_for_backups[3] += reallocation_difference_for_backups;
							backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
							
						}

						backup_buffers[6][backup_index[6]] = (*point_it).x;
						backup_buffers[6][backup_index[6]+1] = (*point_it).y;
						backup_index[6]+=2;
						if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
							allocated_space_for_backups[6] += 10;
							backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
							
						}
					}
				} else if (!send_info[spot*8+7]){											// side0 giveup and side2 backup (hence also backup for corner4)
					backup_buffers[2][backup_index[2]] = (*point_it).x;
					backup_buffers[2][backup_index[2]+1] = (*point_it).y;
					backup_index[2]+=2;
					if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
						allocated_space_for_backups[2] += reallocation_difference_for_backups;
						backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}
				} else {																	// side0 giveup and side2 backup and side3 backup  (hence also backup for corner4 and corner6)
					backup_buffers[2][backup_index[2]] = (*point_it).x;
					backup_buffers[2][backup_index[2]+1] = (*point_it).y;
					backup_index[2]+=2;
					if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
						allocated_space_for_backups[2] += reallocation_difference_for_backups;
						backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[4][backup_index[4]] = (*point_it).x;
					backup_buffers[4][backup_index[4]+1] = (*point_it).y;
					backup_index[4]+=2;
					if (backup_index[4] == allocated_space_for_backups[4]){		// if we have reached capacity for backups
						allocated_space_for_backups[4] += 10;
						backup_buffers[4] = (double*)realloc(backup_buffers[4], allocated_space_for_backups[4] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[3][backup_index[3]] = (*point_it).x;
					backup_buffers[3][backup_index[3]+1] = (*point_it).y;
					backup_index[3]+=2;
					if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
						allocated_space_for_backups[3] += reallocation_difference_for_backups;
						backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
						
					}

					backup_buffers[6][backup_index[6]] = (*point_it).x;
					backup_buffers[6][backup_index[6]+1] = (*point_it).y;
					backup_index[6]+=2;
					if (backup_index[6] == allocated_space_for_backups[6]){		// if we have reached capacity for backups
						allocated_space_for_backups[6] += 10;
						backup_buffers[6] = (double*)realloc(backup_buffers[6], allocated_space_for_backups[6] * sizeof(double));	// allocate more space
						
					}
				}
			} else {																		// side0 and side3 corner giveup  (hence also backup for side0 and side3)
				backup_buffers[3][backup_index[3]] = (*point_it).x;
				backup_buffers[3][backup_index[3]+1] = (*point_it).y;
				backup_index[3]+=2;
				if (backup_index[3] == allocated_space_for_backups[3]){		// if we have reached capacity for backups
					allocated_space_for_backups[3] += reallocation_difference_for_backups;
					backup_buffers[3] = (double*)realloc(backup_buffers[3], allocated_space_for_backups[3] * sizeof(double));	// allocate more space
					
				}

				backup_buffers[0][backup_index[0]] = (*point_it).x;
				backup_buffers[0][backup_index[0]+1] = (*point_it).y;
				backup_index[0]+=2;
				if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
					allocated_space_for_backups[0] += reallocation_difference_for_backups;
					backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
					
				}
			}
		} else {																			// side0 and side2 corner giveup  (hence also backup for side0 and side2)
			backup_buffers[0][backup_index[0]] = (*point_it).x;
			backup_buffers[0][backup_index[0]+1] = (*point_it).y;
			backup_index[0]+=2;
			if (backup_index[0] == allocated_space_for_backups[0]){		// if we have reached capacity for backups
				allocated_space_for_backups[0] += reallocation_difference_for_backups;
				backup_buffers[0] = (double*)realloc(backup_buffers[0], allocated_space_for_backups[0] * sizeof(double));	// allocate more space
				
			}

			backup_buffers[2][backup_index[2]] = (*point_it).x;
			backup_buffers[2][backup_index[2]+1] = (*point_it).y;
			backup_index[2]+=2;
			if (backup_index[2] == allocated_space_for_backups[2]){		// if we have reached capacity for backups
				allocated_space_for_backups[2] += reallocation_difference_for_backups;
				backup_buffers[2] = (double*)realloc(backup_buffers[2], allocated_space_for_backups[2] * sizeof(double));	// allocate more space
				
			}
		}

		spot++;
	}


	if (myid == print_rank) std::cout<<"Rank "<<myid<<": done sorting backup data for sending\n";

	backup_index[0]/=2;
	backup_index[1]/=2;
	backup_index[2]/=2;
	backup_index[3]/=2;
	backup_index[4]/=2;
	backup_index[5]/=2;
	backup_index[6]/=2;
	backup_index[7]/=2;


	return;
}


/**
 *	\brief This function adjusts boundary_lines to be filled with the halfplanes
 *	determining the final convex region.
 *
 *	
 *	@param [out] boundary_lines An std::vector of halfplanes to be filled with halfplanes describing the entire convex region
 *	@param [in] min An array of two doubles specifying the minimum x and y values respectively of the entire set of input points.
 *	@param [in] max An array of two doubles specifying the maximum x and y values respectively of the entire set of input points.
 *	@param [in] spacing A double specifying the user-requested spacing between extreme x and y values and the boundary halfplanes.
 */
void expand_boundary_lines(std::vector<line>& boundary_lines, double * min, double * max, double spacing){

	boundary_lines.clear();

	boundary_lines.emplace_back(1, 0, max[0] + spacing);
	boundary_lines.emplace_back(-1, 0, spacing - min[0]);
	boundary_lines.emplace_back(0, 1, max[1] + spacing);
	boundary_lines.emplace_back(0, -1, spacing - min[1]);

	return;
}


/**
 *	\brief This function computes the Voronoi cells for points received as "giveups" to the calling process.
 *
 *	
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] boundary_lines An std::vector of halfplanes to be filled with halfplanes describing the entire convex region
 *	@param [in] recieved_giveups An std::vector of pointers to double arrays, each being the giveups sent by a particular process
 *	@param [in] number_of_giveups_to_recieve A std::vector of ints representing the number of giveups recieved from each process
 *	@param [in] source_indices A std::vector indicating which neighbour [0,7] the calling process communicates with
 *	@param [in] recieved_backups A std::vector of pointers to double arrays, each being the backups sent by a particular process
 *	@param [in] number_of_backups_to_recieve A std::vector of ints representing the number of backups recieved from each process
 *	@param [in, out] mycells A struct filled with cell data for the points that the calling process is responsible for
 *	@param [in] giveup_indices A std::vector of ints indicating the indices of points from input_points which the calling process has sent to other processes to generate the cells for.
 */
void compute_remaining_cells(std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve, std::vector<cell_info>& mycells, std::vector<unsigned int>& giveup_indices, double * giveups_time, double * recompute_time){
	
	double st_giveups, et_giveups, st_recompute, et_recompute;		// doubles for timing function
	st_giveups = MPI_Wtime();

	size_t current_source = 0;					// which source are we computing the giveup cells for
	std::vector<int> mypoints_to_recalculate;	// indices of points from our initial set of points whose cells need to be recalculated
	mypoints_to_recalculate.reserve(10);
	// iterate through recieved giveup buffers
	for(auto buffer_it = recieved_giveups.cbegin(); buffer_it != recieved_giveups.cend(); ++buffer_it){

		for (int point_spot = 0; point_spot < number_of_giveups_to_recieve[source_indices[current_source]]*2; point_spot+=2){
			point p{(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]};	// the current point whose cell we calculate
			mycells.emplace_back(p);

			std::vector<ordered_vector> rays(4);
			rays.reserve(20);
			rays[0] = ordered_vector{0,1, 1};
			rays[1] = ordered_vector{1,0, 2};
			rays[2] = ordered_vector{0,-1, 3};
			rays[3] = ordered_vector{-1,0, 4};

			// Create a queue of edges to cycle through
			std::vector<subwedge> WedgeQueue(4);
			WedgeQueue.reserve(30);
			WedgeQueue[0] = subwedge{&rays[3], &rays[0]};	// order is counterclockwise starting at noon
			WedgeQueue[1] = subwedge{&rays[2], &rays[3]};
			WedgeQueue[2] = subwedge{&rays[1], &rays[2]};
			WedgeQueue[3] = subwedge{&rays[0], &rays[1]};


			size_t edge_spot = 0;		// loop through remaining edges
			while(WedgeQueue.begin() + (long)edge_spot != WedgeQueue.end()){
				// find endpoints and bisecting lines
				find_Endpoint_and_Line_advanced(p, *buffer_it + point_spot, *(WedgeQueue[edge_spot]).vector1, input_points, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve);
				find_Endpoint_and_Line_advanced(p, *buffer_it + point_spot, *(WedgeQueue[edge_spot]).vector2, input_points, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve);

				// find determinant of matrix B
				double a = ((WedgeQueue[edge_spot]).vector1->boundary)*(((WedgeQueue[edge_spot]).vector1->t)*(*(WedgeQueue[edge_spot]).vector1));
				double b = ((WedgeQueue[edge_spot]).vector1->boundary)*(((WedgeQueue[edge_spot]).vector2->t)*(*(WedgeQueue[edge_spot]).vector2));
				double c = ((WedgeQueue[edge_spot]).vector2->boundary)*(((WedgeQueue[edge_spot]).vector1->t)*(*(WedgeQueue[edge_spot]).vector1));
				double d = ((WedgeQueue[edge_spot]).vector2->boundary)*(((WedgeQueue[edge_spot]).vector2->t)*(*(WedgeQueue[edge_spot]).vector2));
				double determinant = (a*d)-(b*c);

				// CASE 1: infinitely many solutions or no solutions
				if (determinant*determinant < 1e-26){ 
					if (	((WedgeQueue[edge_spot]).vector1->boundary.ycoeff) == 0 		){	// if vertical lines
						// if infinitely many solutions (i.e. same line):
						if (	(((WedgeQueue[edge_spot]).vector1->boundary.c) / ((WedgeQueue[edge_spot]).vector1->boundary.xcoeff )) == (((WedgeQueue[edge_spot]).vector2->boundary.c) / ((WedgeQueue[edge_spot]).vector2->boundary.xcoeff))	){
							// same line, hence no vertices in this subwedge, hence we remove this edge from WedgeQueue
							edge_spot++;
						} else { // no solutions: lines are parallel but not the same
							// split the edge in half
							// new ray is
							// double r1 = 0;
							double r2 = 1;

							// check that the ray is in the wedge:
							double v1, v2, v3,  v4, alpha1, alpha2, det;
							v1 = (WedgeQueue[edge_spot]).vector1->x;
							v2 = (WedgeQueue[edge_spot]).vector1->y;
							v3 = (WedgeQueue[edge_spot]).vector2->x;
							v4 = (WedgeQueue[edge_spot]).vector2->y;
							det = v1*v4-v3*v2;
							alpha1 = (r2*v2)/det;
							alpha2 = (v1*r2)/det;
							if ((alpha1 < 0) || (alpha2 < 0)){		// if ray is not in the wedge
								r2 = -1;
							}

							// now we have our new ray (r1,r2)
							// add a new subedge to represent second half of this edge
							double idx;
							if (((WedgeQueue[edge_spot]).vector2->index) != 1){
								idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
							} else {
								idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
							}
							rays.emplace_back(0,r2, idx);
							WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);
							// then adjust this current edge to represent first hald of this edge
							((WedgeQueue[edge_spot]).vector2) = &rays.back();
						}

					} else /* not vertical lines */if (	(((WedgeQueue[edge_spot]).vector1->boundary.c) / ((WedgeQueue[edge_spot]).vector1->boundary.ycoeff )) == (((WedgeQueue[edge_spot]).vector2->boundary.c) / ((WedgeQueue[edge_spot]).vector2->boundary.ycoeff))	){
						// same line, hence no vertices in this subwedge, hence we remove this edge from WedgeQueue
						edge_spot++;
					} else {	// no solutions
						// split the edge in half
						// new ray is
						double r1, r2;
						if (	(WedgeQueue[edge_spot]).vector1->boundary.xcoeff == 0		){
							r1 = 1;
							r2 = 0;
						} else {
							r1 = ((WedgeQueue[edge_spot]).vector1->boundary.xcoeff);
							r2 = (-1) / ((WedgeQueue[edge_spot]).vector1->boundary.ycoeff);
							// normalize:
							double length = sqrt(r1*r1 + r2*r2);
							r1 /= length;
							r2 /= length;
						}
						// check that the ray is in the wedge:
						double v1, v2, v3,  v4, alpha1, alpha2, det;
						v1 = (WedgeQueue[edge_spot]).vector1->x;
						v2 = (WedgeQueue[edge_spot]).vector1->y;
						v3 = (WedgeQueue[edge_spot]).vector2->x;
						v4 = (WedgeQueue[edge_spot]).vector2->y;
						det = v1*v4-v3*v2;
						alpha1 = (r1*v4 - r2*v2)/det;
						alpha2 = (v1*r2 - v3*r1)/det;
						if ((alpha1 < 0) || (alpha2 < 0)){		// if ray is not in the wedge
							r1 *= -1;
							r2 *= -1;
						}

						// now we have our new ray (r1,r2)
						// add a new subedge to represent second half of this edge
						double idx;
						if (((WedgeQueue[edge_spot]).vector2->index) != 1){
							idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
						} else {
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
						}
						rays.emplace_back(r1,r2, idx);
						WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);
						// then adjust this current edge to represent first hald of this edge
						((WedgeQueue[edge_spot]).vector2) = &rays.back();
						
					}
				// CASE 2: unique solution
				} else {
					// determine unique solution:
					double lambda1 = (a*d - b*d)/determinant;
					double lambda2 = (a*d - c*a)/determinant;
					point intersection = p + (lambda1 * ((WedgeQueue[edge_spot]).vector1->t)) * (*(WedgeQueue[edge_spot]).vector1) + (lambda2 * ((WedgeQueue[edge_spot]).vector2->t)) * (*(WedgeQueue[edge_spot]).vector2);

					if ( (lambda1 < 0) || (lambda2 < 0)){	// if unique solution is not in the subwedge
						// we split the subwedge
						double idx;
						if (((WedgeQueue[edge_spot]).vector2->index) != 1){
							idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
						} else {
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
						}
						rays.emplace_back((p - intersection) / p.distance(intersection), idx);

						// add a new subedge to represent second half of this edge
						WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

						// then adjust this current edge to represent first half of this edge
						(WedgeQueue[edge_spot]).vector2 = &rays.back();


					} else {	// the unique solution IS in the subwedge
						// now we determine whether the unique solution is in the cell or not
						bool in_cell = true;
						double min_distance = p.distance(intersection);		// distance from p to intersection


						// check backup points for a closer point to intersection
						size_t current_source = 0;
						for (auto buffer_it = recieved_backups.begin(); buffer_it != recieved_backups.end(); ++buffer_it){
							for (int point_spot = 0; point_spot < number_of_backups_to_recieve[source_indices[current_source]]*2; point_spot+=2){
								double check_dist = (intersection).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
								if ( check_dist < min_distance){	// if we have found a closer point
									if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
										in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
										break;
									}
								}
							}
							current_source++;
						}

						// check giveup points for a closer point to intersection
						if (in_cell){
							current_source = 0;
							for (auto buffer_it = recieved_giveups.begin(); buffer_it != recieved_giveups.end(); ++buffer_it){
								for (int point_spot = 0; point_spot < number_of_giveups_to_recieve[source_indices[current_source]]*2; point_spot+=2){
									double check_dist = (intersection).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
									if ( check_dist < min_distance){	// if we have found a closer point
										if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
											in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
											break;
										}
									}
								}
								current_source++;
							}
						}

						// check input points for a closer point to intersection
						if (in_cell){
							for (auto it = input_points.begin(); it != input_points.end(); ++it){		// loop through input points and find closest input point to endpoint and corresponding distance
								double check_dist = (*it).distance(intersection);
								if ( check_dist < min_distance){	// if we have found a closer point
									if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
										in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
										break;
									}
								}
							}
						}


						if (in_cell) {	// we have found a vertex
							if (in_region(intersection, boundary_lines)){	// if in region ("world")...
								// save vertex data
								mycells.back().add_vertex(intersection, (WedgeQueue[edge_spot]).vector1->index, (WedgeQueue[edge_spot]).vector1->boundary);

								// save neighbour data
								if (((WedgeQueue[edge_spot]).vector1->nbr) != -1){	// if the neighbour is one of mypoints. is it a giveup?
									bool is_it_a_giveup = false;
									for (auto giveup_it = giveup_indices.begin(); giveup_it != giveup_indices.end(); ++giveup_it){	// loop through giveup indices and see if there is a match
										if ((int)(*giveup_it) != ((WedgeQueue[edge_spot]).vector1->nbr)){
											continue;
										} else { // we have a match
											is_it_a_giveup = true;
											break;
										}
									}

									if (!is_it_a_giveup){	// if the point is not a giveup, the point needs to be recalculated
										mypoints_to_recalculate.emplace_back((WedgeQueue[edge_spot]).vector1->nbr);
									}
								}
								
								edge_spot++;
							} else {	// we are not in the world...
								double idx;
								if (((WedgeQueue[edge_spot]).vector2->index) != 1){
									idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
								} else {
									idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
								}
								rays.emplace_back((intersection - p) / p.distance(intersection), idx);

								// add a new subedge to represent second half of this edge
								WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

								// then adjust this current edge to represent first half of this edge
								(WedgeQueue[edge_spot]).vector2 = (&rays.back());
							}

						} else {		// we have not found a vertex and we split
							double idx;
							if (((WedgeQueue[edge_spot]).vector2->index) != 1){
								idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
							} else {
								idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
							}
							rays.emplace_back((intersection - p) / p.distance(intersection), idx);

							// add a new subedge to represent second half of this edge
							WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

							// then adjust this current edge to represent first half of this edge
							(WedgeQueue[edge_spot]).vector2 = (&rays.back());

						}
					}
				}
			}	// end WedgeQueue loop

			mycells.back().sort_vertices();

		}	// end loop through this current giveup buffer
		current_source++;
	}
	et_giveups = MPI_Wtime();

	// now we need to deal with the points to recalculate...
	st_recompute = MPI_Wtime();
	recompute_some_of_mypoints(input_points, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve, mycells, mypoints_to_recalculate);
	et_recompute = MPI_Wtime();

	*giveups_time = et_giveups - st_giveups;
	*recompute_time = et_recompute - st_recompute;

	return;
}


/**
 *	\brief This function is called by compute_remaining_cells(). It is a modified version of find_Endpoint_and_Line()
 *	that loops through the recieved giveups and backups as well as the process's initial input points.
 *
 *	
 *	@param [in] p A point object from which the ray emanates
 *	@param [in] p_address The address of the point p in its array in received_giveups
 *	@param [in, out] direction An ordered_vector object, a unit vector representing the direction
 *	of the ray. Upon return the member variables boundary and t will be populated with the 
 *	boundary line which the ray intersects with and the distance to that intersection, respectively.
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] boundary_lines An std::vector of halfplanes to be filled with halfplanes describing the entire convex region
 *	@param [in] recieved_giveups An std::vector of pointers to double arrays, each being the giveups sent by a particular process
 *	@param [in] number_of_giveups_to_recieve A std::vector of ints representing the number of giveups recieved from each process
 *	@param [in] source_indices A std::vector indicating which neighbour [0,7] the calling process communicates with
 *	@param [in] recieved_backups A std::vector of pointers to double arrays, each being the backups sent by a particular process
 *	@param [in] number_of_backups_to_recieve A std::vector of ints representing the number of backups recieved from each process
 */
void find_Endpoint_and_Line_advanced(point p, double * p_address, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve){

	// first check if we have already computed endpoint
	if (direction.t > 0) return;

	// then find intersection of ray from p in [direction] with boundary of world
	find_Intersection_with_Boundary_advanced(p, direction, boundary_lines);

	bool foundEndpoint = false;		// Have we found the endpoint yet?
	std::vector<double *> skip_items (1, p_address);			// Points from received_giveups and received_backups that we don't want to consider as "closer" than p to suspected endpoint
	std::vector<std::vector<point>::iterator> skip_iterators;	// Points from input_points that we don't want to consider as "closer" than p to suspected endpoint

	while (! foundEndpoint){
		point endpoint = p + (direction.t * direction);	// this is our endpoint
		// check whether the endpoint we have found is in the cell
		bool in_cell = true;
		double min_distance = p.distance(endpoint);		// distance from p to endpoint
		point closeNeighbour = p;					// closest neighbour to endpoint
		double * closeNeighbourAddress;				// location of closest neighbour
		size_t current_source = 0;					// iterate through recieved_giveups and recieved_backups entries

		// iterate through received_giveups and look for a closer neighbour
		for (auto buffer_it = recieved_giveups.begin(); buffer_it != recieved_giveups.end(); ++buffer_it){
			for (int point_spot = 0; point_spot < number_of_giveups_to_recieve[source_indices[current_source]]*2; point_spot+=2){

				double current_distance = (endpoint).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
				if ( current_distance < min_distance){	// if we have found a closer point
					// check whether we should skip this point or not
					bool skip = false;
					for (auto& skip_it : skip_items){
						if (*buffer_it + point_spot == skip_it){
							skip = true;
							break;
						}
					}


					if (!skip){		// make sure that we aren't considering repeat items as a closer input point
						min_distance = current_distance;
						closeNeighbourAddress = *buffer_it + point_spot;
						direction.nbr = -1;	// we are not using a neighbour from input_points
						in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
					}
				}
			}
			current_source++;
		}

		current_source = 0;

		// iterate through received_backups and look for a closer neighbour
		for (auto buffer_it = recieved_backups.begin(); buffer_it != recieved_backups.end(); ++buffer_it){
			for (int point_spot = 0; point_spot < number_of_backups_to_recieve[source_indices[current_source]]*2; point_spot+=2){

				double current_distance = (endpoint).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
				if ( current_distance < min_distance){	// if we have found a closer point

					// check whether we should skip this point or not
					bool skip = false;
					for (auto& skip_it : skip_items){
						if (*buffer_it + point_spot == skip_it){
							skip = true;
							break;
						}
					}

					if (!skip){		// make sure that we aren't considering repeat items as a closer input point
						min_distance = current_distance;
						closeNeighbourAddress = *buffer_it + point_spot;
						direction.nbr = -1;	// we are not using a neighbour from mypoints
						in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
					}
				}
			}
			current_source++;
		}

		bool use_close_neighbour_it = false;	// will be true if we find a closer neighbour in input_points
		std::vector<point>::iterator closeNeighbour_it;		// reference to closer neighbour
		int index = -1;		// used for recording index of nearer neighbour in input_points

		// iterate through input_points and look for a closer neighbour
		for (auto it = input_points.begin(); it != input_points.end(); ++it){
			index++;
			double current_distance = (*it).distance(endpoint);		// distance from current input point to endpoint
			if ( current_distance < min_distance){	// if we have found a closer point

				// check whether we should skip this point or not
				bool skip = false;
				for (auto& skip_it : skip_iterators){
					if (it == skip_it){
						skip = true;
						break;
					}
				}

				if (!skip){		// make sure that we aren't considering repeat items as a closer input point
					min_distance = current_distance;
					closeNeighbour_it = it;		// record closer neighbour information
					direction.nbr = index;
					in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
					use_close_neighbour_it = true;
				}
			}
		}


		if (in_cell){	// we have found the endpoint and we are done
			foundEndpoint = true;
		} else {		// we have not found the endpoint; there is a closer input point to the current suspected endpoint
			// add closeNeighbour to skip_items
				if (use_close_neighbour_it){
					skip_iterators.push_back(closeNeighbour_it);	// don't want to consider this input point again
					closeNeighbour = *closeNeighbour_it;
				} else {
					skip_items.push_back(closeNeighbourAddress);	// don't want to consider this point again
					closeNeighbour = point{  closeNeighbourAddress[0], closeNeighbourAddress[1]     };
				}
			// we adjust suspected endpoint and boundary line
			// suspected boundary lines becomes the perpedicular bisector between p and closeNeighbour
				if ((p.y - (closeNeighbour).y) == 0){	// the the bisector is vertical
					direction.boundary = line{1, 0, (p.x + (closeNeighbour).x)/2};
				} else {
					direction.boundary = line{(p.x-(closeNeighbour).x)/(p.y-(closeNeighbour).y), 1, (p.y+(closeNeighbour).y + (p.x * p.x - ((closeNeighbour).x * (closeNeighbour).x))/(p.y-(closeNeighbour).y))*0.5};
				}

			// suspected endpoint is now the intersection between the perpendicular bisector and ray from p
				direction.t = (direction.boundary.c - (direction.boundary)*p)/(direction.boundary * direction);
		}
	}


	return;
}


/**
 *	\brief This function is called by find_Endpoint_and_Line_advanced() and find_Endpoint_and_Line_recomputation.
 *	It is a modified version of find_Intersection_with_Boundary() that doesn't return the intersecting side index in boundary_lines.
 *
 *	
 *	@param [in] p A point object from which the ray emanates
 *	@param [in, out] direction An ordered_vector object, a unit vector representing the direction
 *	of the ray. Upon return the member variables boundary and t will be populated with the 
 *	boundary line which the ray intersects with and the distance to that intersection, respectively.
 *	@param [in] boundary_lines An std::vector of halfplanes to be filled with halfplanes describing the entire convex region
 */
void find_Intersection_with_Boundary_advanced(const point p, ordered_vector& direction, std::vector<line>& boundary_lines){
	double ti;

	// cycle through halfplanes in boundary
	auto it = boundary_lines.begin();
	bool foundFirstIntersection = false;
	while ((! foundFirstIntersection) ){	// first we find an initial intersection of the ray with boundary
		double parallel_check = ((*it)*direction);	// will be 0 if boundary line and direction vector are parallel
		if ( parallel_check != 0 ){	// if boundary line and direction vector are not parallel
			ti = ( (*it).c - ((*it)*p)) / parallel_check;	// distance from p to boundary line in direction of [direction]

			if (ti > 0){	// if ray intersects boundary line
				direction.t = ti;	// we have found first intersection
				direction.boundary = (*it);
				foundFirstIntersection = true;
			}
		}
		it++; // move to consider the next boundary line
	}
	

	while (it != boundary_lines.end()){	// now we iterate through the remaining boundary lines and see if we can find a closer intersection
		
		double parallel_check = ((*it)*direction);	// will be 0 if boundary line and direction vector are parallel
		if ( parallel_check != 0 ){	// if boundary line and direction vector are not parallel
			ti = ( (*it).c - ((*it)*p)) / parallel_check;	// distance from p to boundary line in direction of [direction]

			if ((ti > 0) && (ti < direction.t)){	// if ray intersects boundary line AND at a point closer than before
				
				direction.t = ti;	// we have found a closer intersection
				direction.boundary = (*it);
			}
		}
		it++; // move to consider the next boundary line
	}

	// done !
	return;
}


/**
 *	\brief This function is called by compute_remaining_cells() in order to recompute the cell information
 *	for points in input_points which are not giveups to other processes and have been affected by the 
 *	introduction of new points as giveups and backups received from other processes.
 *
 *	
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] boundary_lines An std::vector of halfplanes to be filled with halfplanes describing the entire convex region
 *	@param [in] recieved_giveups An std::vector of pointers to double arrays, each being the giveups sent by a particular process
 *	@param [in] number_of_giveups_to_recieve A std::vector of ints representing the number of giveups recieved from each process
 *	@param [in] source_indices A std::vector indicating which neighbour [0,7] the calling process communicates with
 *	@param [in] recieved_backups A std::vector of pointers to double arrays, each being the backups sent by a particular process
 *	@param [in] number_of_backups_to_recieve A std::vector of ints representing the number of backups recieved from each process
 *	@param [in, out] mycells A struct filled with cell data for the points that the calling process is responsible for
 *	@param [in] mypoints_to_recalculate A std::vector of ints indicating the indices of points from input_points which the calling process must recalculate cells for.
 */
void recompute_some_of_mypoints(std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve, std::vector<cell_info>& mycells, std::vector<int>& mypoints_to_recalculate){


	for (auto point_it = mypoints_to_recalculate.begin(); point_it != mypoints_to_recalculate.end(); ++point_it){

		// see if this index has already appeared before, because indices may have been recorded more than once
		bool move_on = false;
		for (auto it = mypoints_to_recalculate.begin(); it != point_it; ++it){
			if ((*it) == (*point_it)){
				move_on = true;
				break;
			}
		}
		if (move_on) continue;	// we've already computed this point again so move to next index

		point p {input_points[(size_t)(*point_it)]};
		mycells[(size_t)*point_it].vertices.clear();	// clear the cell data we've already computed

		std::vector<ordered_vector> rays(4);
		rays.reserve(20);
		rays[0] = ordered_vector{0,1, 1};
		rays[1] = ordered_vector{1,0, 2};
		rays[2] = ordered_vector{0,-1, 3};
		rays[3] = ordered_vector{-1,0, 4};

		// Create a queue of edges to cycle through
		std::vector<subwedge> WedgeQueue(4);
		WedgeQueue.reserve(30);
		WedgeQueue[0] = subwedge{&rays[3], &rays[0]};	// order is counterclockwise starting at noon
		WedgeQueue[1] = subwedge{&rays[2], &rays[3]};
		WedgeQueue[2] = subwedge{&rays[1], &rays[2]};
		WedgeQueue[3] = subwedge{&rays[0], &rays[1]};


		size_t edge_spot = 0;		// loop through remaining edges
		while(WedgeQueue.begin() + (long)edge_spot != WedgeQueue.end()){

			// find endpoints and bisecting lines
			find_Endpoint_and_Line_recomputation(p, input_points.begin() + (*point_it), *(WedgeQueue[edge_spot]).vector1, input_points, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve);
			find_Endpoint_and_Line_recomputation(p, input_points.begin() + (*point_it), *(WedgeQueue[edge_spot]).vector2, input_points, boundary_lines, recieved_giveups, number_of_giveups_to_recieve, source_indices, recieved_backups, number_of_backups_to_recieve);

			// find determinant of matrix B
			double a = ((WedgeQueue[edge_spot]).vector1->boundary)*(((WedgeQueue[edge_spot]).vector1->t)*(*(WedgeQueue[edge_spot]).vector1));
			double b = ((WedgeQueue[edge_spot]).vector1->boundary)*(((WedgeQueue[edge_spot]).vector2->t)*(*(WedgeQueue[edge_spot]).vector2));
			double c = ((WedgeQueue[edge_spot]).vector2->boundary)*(((WedgeQueue[edge_spot]).vector1->t)*(*(WedgeQueue[edge_spot]).vector1));
			double d = ((WedgeQueue[edge_spot]).vector2->boundary)*(((WedgeQueue[edge_spot]).vector2->t)*(*(WedgeQueue[edge_spot]).vector2));
			double determinant = (a*d)-(b*c);

			// CASE 1: infinitely many solutions or no solutions
			if (determinant*determinant < 1e-26){ 
				if (	((WedgeQueue[edge_spot]).vector1->boundary.ycoeff) == 0 		){	// if vertical lines
					// if infinitely many solutions (i.e. same line):
					if (	(((WedgeQueue[edge_spot]).vector1->boundary.c) / ((WedgeQueue[edge_spot]).vector1->boundary.xcoeff )) == (((WedgeQueue[edge_spot]).vector2->boundary.c) / ((WedgeQueue[edge_spot]).vector2->boundary.xcoeff))	){
						// same line, hence no vertices in this subwedge, hence we remove this edge from WedgeQueue
						edge_spot++;
					} else { // no solutions: lines are parallel but not the same
						// split the edge in half
						// new ray is
						// double r1 = 0;
						double r2 = 1;

						// check that the ray is in the wedge:
						double v1, v2, v3,  v4, alpha1, alpha2, det;
						v1 = (WedgeQueue[edge_spot]).vector1->x;
						v2 = (WedgeQueue[edge_spot]).vector1->y;
						v3 = (WedgeQueue[edge_spot]).vector2->x;
						v4 = (WedgeQueue[edge_spot]).vector2->y;
						det = v1*v4-v3*v2;
						alpha1 = (r2*v2)/det;
						alpha2 = (v1*r2)/det;
						if ((alpha1 < 0) || (alpha2 < 0)){		// if ray is not in the wedge
							r2 = -1;
						}

						// now we have our new ray (r1,r2)
						// add a new subedge to represent second half of this edge
						double idx;
						if (((WedgeQueue[edge_spot]).vector2->index) != 1){
							idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
						} else {
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
						}
						rays.emplace_back(0,r2, idx);
						WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);
						// then adjust this current edge to represent first hald of this edge
						((WedgeQueue[edge_spot]).vector2) = &rays.back();

					}

				} else /* not vertical lines */if (	(((WedgeQueue[edge_spot]).vector1->boundary.c) / ((WedgeQueue[edge_spot]).vector1->boundary.ycoeff )) == (((WedgeQueue[edge_spot]).vector2->boundary.c) / ((WedgeQueue[edge_spot]).vector2->boundary.ycoeff))	){
					// same line, hence no vertices in this subwedge, hence we remove this edge from WedgeQueue
					edge_spot++;
				} else {	// no solutions
					// split the edge in half
					// new ray is
					double r1, r2;
					if (	(WedgeQueue[edge_spot]).vector1->boundary.xcoeff == 0		){
						r1 = 1;
						r2 = 0;
					} else {
						r1 = ((WedgeQueue[edge_spot]).vector1->boundary.xcoeff);
						r2 = (-1) / ((WedgeQueue[edge_spot]).vector1->boundary.ycoeff);
						// normalize:
						double length = sqrt(r1*r1 + r2*r2);
						r1 /= length;
						r2 /= length;
					}
					// check that the ray is in the wedge:
					double v1, v2, v3,  v4, alpha1, alpha2, det;
					v1 = (WedgeQueue[edge_spot]).vector1->x;
					v2 = (WedgeQueue[edge_spot]).vector1->y;
					v3 = (WedgeQueue[edge_spot]).vector2->x;
					v4 = (WedgeQueue[edge_spot]).vector2->y;
					det = v1*v4-v3*v2;
					alpha1 = (r1*v4 - r2*v2)/det;
					alpha2 = (v1*r2 - v3*r1)/det;
					if ((alpha1 < 0) || (alpha2 < 0)){		// if ray is not in the wedge
						r1 *= -1;
						r2 *= -1;
					}

					// now we have our new ray (r1,r2)
					// add a new subedge to represent second half of this edge
					double idx;
					if (((WedgeQueue[edge_spot]).vector2->index) != 1){
						idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
					} else {
						idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
					}
					rays.emplace_back(r1,r2, idx);
					WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);
					// then adjust this current edge to represent first hald of this edge
					((WedgeQueue[edge_spot]).vector2) = &rays.back();

				}
			// CASE 2: unique solution
			} else {
				// determine unique solution:
				double lambda1 = (a*d - b*d)/determinant;
				double lambda2 = (a*d - c*a)/determinant;
				point intersection = p + (lambda1 * ((WedgeQueue[edge_spot]).vector1->t)) * (*(WedgeQueue[edge_spot]).vector1) + (lambda2 * ((WedgeQueue[edge_spot]).vector2->t)) * (*(WedgeQueue[edge_spot]).vector2);

				if ( (lambda1 < 0) || (lambda2 < 0)){	// if unique solution is not in the subwedge
					// we split the subwedge
					double idx;
					if (((WedgeQueue[edge_spot]).vector2->index) != 1){
						idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
					} else {
						idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
					}
					rays.emplace_back((p - intersection) / p.distance(intersection), idx);

					// add a new subedge to represent second half of this edge
					WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

					// then adjust this current edge to represent first half of this edge
					(WedgeQueue[edge_spot]).vector2 = &rays.back();


				} else {	// the unique solution IS in the subwedge
					// now we determine whether the unique solution is in the cell or not
					bool in_cell = true;
					double min_distance = p.distance(intersection);		// distance from p to intersection

					// check backup points for a closer point to intersection
					size_t current_source = 0;
					for (auto buffer_it = recieved_backups.begin(); buffer_it != recieved_backups.end(); ++buffer_it){
						for (int point_spot = 0; point_spot < number_of_backups_to_recieve[source_indices[current_source]]*2; point_spot+=2){
							double check_dist = (intersection).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
							if ( check_dist < min_distance){	// if we have found a closer point
								if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
									in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
									break;
								}
							}
						}
						current_source++;
					}

					// check giveup points for a closer point to intersection
					if (in_cell){
						current_source = 0;
						for (auto buffer_it = recieved_giveups.begin(); buffer_it != recieved_giveups.end(); ++buffer_it){
							for (int point_spot = 0; point_spot < number_of_giveups_to_recieve[source_indices[current_source]]*2; point_spot+=2){
								double check_dist = (intersection).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
								if ( check_dist < min_distance){	// if we have found a closer point
									if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
										in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
										break;
									}
								}
							}
							current_source++;
						}
					}

					// check input points for a closer point to intersection
					if (in_cell){
						for (auto it = input_points.begin(); it != input_points.end(); ++it){		// loop through input points and find closest input point to endpoint and corresponding distance
							double check_dist = (*it).distance(intersection);
							if ( check_dist < min_distance){	// if we have found a closer point
								if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
									in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
									break;
								}
							}
						}
					}

					if (in_cell) {	// we have found a vertex
						if (in_region(intersection, boundary_lines)){	// if we are in region ("world")...
							// save vertex data
							mycells[(size_t)*point_it].add_vertex(intersection, (WedgeQueue[edge_spot]).vector1->index, (WedgeQueue[edge_spot]).vector1->boundary);
							edge_spot++;

						} else {	// we are not in the world...
							double idx;
							if (((WedgeQueue[edge_spot]).vector2->index) != 1){
								idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
							} else {
								idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
							}
							rays.emplace_back((intersection - p) / p.distance(intersection), idx);

							// add a new subedge to represent second half of this edge
							WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

							// then adjust this current edge to represent first half of this edge
							(WedgeQueue[edge_spot]).vector2 = (&rays.back());
						}

					} else {		// we have not found a vertex and we split
						double idx;
						if (((WedgeQueue[edge_spot]).vector2->index) != 1){
							idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
						} else {
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 5)/2;
						}
						rays.emplace_back((intersection - p) / p.distance(intersection), idx);

						// add a new subedge to represent second half of this edge
						WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

						// then adjust this current edge to represent first half of this edge
						(WedgeQueue[edge_spot]).vector2 = (&rays.back());
					}
				}
			}
		}	// end WedgeQueue loop


		mycells[(size_t)*point_it].sort_vertices();
	}


	return;
}


/**
 *	\brief This function is called by recompute_some_of_mypoints(). It is a modified version of find_Endpoint_and_Line()
 *	that loops through the recieved giveups and backups as well as the process's initial input points.
 *
 *	
 *	@param [in] p_it A std::vector<point>::iterator referencing a point object from which the ray emanates
 *	@param [in, out] direction An ordered_vector object, a unit vector representing the direction
 *	of the ray. Upon return the member variables boundary and t will be populated with the 
 *	boundary line which the ray intersects with and the distance to that intersection, respectively.
 *	@param [in] input_points An std::vector<point> of the input points that this process is initially responsible for
 *	@param [in] boundary_lines An std::vector of halfplanes to be filled with halfplanes describing the entire convex region
 *	@param [in] recieved_giveups An std::vector of pointers to double arrays, each being the giveups sent by a particular process
 *	@param [in] number_of_giveups_to_recieve A std::vector of ints representing the number of giveups recieved from each process
 *	@param [in] source_indices A std::vector indicating which neighbour [0,7] the calling process communicates with
 *	@param [in] recieved_backups A std::vector of pointers to double arrays, each being the backups sent by a particular process
 *	@param [in] number_of_backups_to_recieve A std::vector of ints representing the number of backups recieved from each process
 */
void find_Endpoint_and_Line_recomputation(point p, std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines, std::vector<double *>& recieved_giveups, std::vector<int>& number_of_giveups_to_recieve, std::vector<size_t>& source_indices, std::vector<double *>& recieved_backups, std::vector<int>& number_of_backups_to_recieve){

	// first check if we have already computed endpoint
	if (direction.t > 0) return;


	// then find intersection of ray from p in [direction] with boundary of world
	find_Intersection_with_Boundary_advanced(p, direction, boundary_lines);

	bool foundEndpoint = false;
	std::vector<double *> skip_items;			// Points from received_giveups and received_backups that we don't want to consider as "closer" than p to suspected endpoint
	skip_items.reserve(10);
	std::vector<std::vector<point>::iterator> skip_iterators(1, p_it);	// Points from input_points that we don't want to consider as "closer" than p to suspected endpoint

	while (! foundEndpoint){
		point endpoint = p + (direction.t * direction);	// this is our endpoint
		// check whether the endpoint we have found is in the cell
		bool in_cell = true;
		double min_distance = p.distance(endpoint);		// distance from p to endpoint
		point closeNeighbour = p;					// closest neighbour to endpoint
		double * closeNeighbourAddress;				// location of closest neighbour
		size_t current_source = 0;					// iterate through recieved_giveups and recieved_backups entries

		// iterate through received_giveups and look for a closer neighbour
		for (auto buffer_it = recieved_giveups.begin(); buffer_it != recieved_giveups.end(); ++buffer_it){
			for (int point_spot = 0; point_spot < number_of_giveups_to_recieve[source_indices[current_source]]*2; point_spot+=2){

				double current_distance = (endpoint).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
				if ( current_distance < min_distance){	// if we have found a closer point

					// check whether we should skip this point or not
					bool skip = false;
					for (auto& skip_it : skip_items){
						if (*buffer_it + point_spot == skip_it){
							skip = true;
							break;
						}
					}

					if (!skip){		// make sure that we aren't considering repeat items as a closer input point
						min_distance = current_distance;
						closeNeighbourAddress = *buffer_it + point_spot;
						in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
					}
				}
			}
			current_source++;
		}

		current_source = 0;

		// iterate through received_backups and look for a closer neighbour
		for (auto buffer_it = recieved_backups.begin(); buffer_it != recieved_backups.end(); ++buffer_it){
			for (int point_spot = 0; point_spot < number_of_backups_to_recieve[source_indices[current_source]]*2; point_spot+=2){

				double current_distance = (endpoint).distance(	(*buffer_it)[point_spot], (*buffer_it)[point_spot+1]		);		// distance from current input point to endpoint
				if ( current_distance < min_distance){	// if we have found a closer point

					// check whether we should skip this point or not
					bool skip = false;
					for (auto& skip_it : skip_items){
						if (*buffer_it + point_spot == skip_it){
							skip = true;
							break;
						}
					}

					if (!skip){		// make sure that we aren't considering repeat items as a closer input point
						min_distance = current_distance;
						closeNeighbourAddress = *buffer_it + point_spot;
						in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
					}
				}
			}
			current_source++;
		}

		bool use_close_neighbour_it = false;	// will be true if we find a closer neighbour in input_points
		std::vector<point>::iterator closeNeighbour_it;		// reference to closer neighbour

		// iterate through input_points and look for a closer neighbour
		for (auto it = input_points.begin(); it != input_points.end(); ++it){
			double current_distance = (*it).distance(endpoint);		// distance from current input point to endpoint
			if ( current_distance < min_distance){	// if we have found a closer point

				// check whether we should skip this point or not
				bool skip = false;
				for (auto& skip_it : skip_iterators){
					if (it == skip_it){
						skip = true;
						break;
					}
				}

				if (!skip){	// make sure that we aren't considering repeat items as a closer input point
					min_distance = current_distance;
					closeNeighbour_it = it;		// record closer neighbour information
					in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
					use_close_neighbour_it = true;
				}
			}
		}


		if (in_cell){	// we have found the endpoint and we are done
			foundEndpoint = true;
		} else {		// we have not found the endpoint; there is a closer input point to the current suspected endpoint
			// add closeNeighbour to skip_items
				if (use_close_neighbour_it){
					skip_iterators.push_back(closeNeighbour_it);	// don't want to consider this input point again
					closeNeighbour = *closeNeighbour_it;
				} else {
					skip_items.push_back(closeNeighbourAddress);	// don't want to consider this point again
					closeNeighbour = point{  closeNeighbourAddress[0], closeNeighbourAddress[1]     };
				}
			// we adjust suspected endpoint and boundary line
			// suspected boundary lines becomes the perpedicular bisector between p and closeNeighbour
				if ((p.y - (closeNeighbour).y) == 0){	// the the bisector is vertical
					direction.boundary = line{1, 0, (p.x + (closeNeighbour).x)/2};
				} else {
					direction.boundary = line{(p.x-(closeNeighbour).x)/(p.y-(closeNeighbour).y), 1, (p.y+(closeNeighbour).y + (p.x * p.x - ((closeNeighbour).x * (closeNeighbour).x))/(p.y-(closeNeighbour).y))*0.5};
				}

			// suspected endpoint is now the intersection between the perpendicular bisector and ray from p
				direction.t = (direction.boundary.c - (direction.boundary)*p)/(direction.boundary * direction);
		}
	}
	return;
}


/**
 *	\brief This function prints the results of simple_parallel_voronoi.cc to a file.
 *	A collective function.
 *
 *	@param [in] filename The name of the file to print results to
 *	@param [in] rank The id of the calling process in the MPI Communicator
 *	@param [in] size The number of processes in the MPI Communicator
 *	@param [in] cells The process's computed cells for its assigned points
 *	@return 0 upon success, -1 if failed to open file for writing
 */
int print_cells_to_file(std::string filename, int rank, int size, std::vector<cell_info>& cells, MPI_Comm comm){

	for (int i = 0; i < size; i++){	// loop through ranks
		if (rank == i){
			std::ofstream fp;
			if (i == 0) fp.open(filename);
			else fp.open(filename, std::ios::app);
			
			if(!fp){	// error checking
				std::cerr<<"Failed to open file for writing final cell data";
				return -1;
			}


			// loop through cells
			for (auto cell_it = cells.begin(); cell_it != cells.end(); ++cell_it){

				if ( (*cell_it).vertices.empty() ) continue;			// skip to next iteration if cell is not ours anymore
				fp << (*cell_it).p.x << " " << (*cell_it).p.y << " ";	// print the point

				// loop through vertices in cell:
				for (auto vert_it = (*cell_it).vertices.begin(); vert_it != (*cell_it).vertices.end(); ++vert_it){
					fp <<(*vert_it).x << " " << (*vert_it).y << " ";
				}
				fp << "\n";
			}

			fp.close();	// close file
		}

		MPI_Barrier(comm);	// synchronize

	}

	return 0;
}


/**
 *	\brief This function tests the accuracy of the cells in one file against the cells in another file.
 *
 *	@param [in] correct_filename The filename containing accurate cell information
 *	@param [in] filename_to_test The filename whose cell data we wish to test for accuracy
 *	@return 0 upon success, 1 if failed to open a file
 */
int test_complex_accuracy(int num_points, std::string correct_filename, std::string filename_to_test){

	// read in points from correct_filename
	std::vector<double> x_values;			// contains x values of points
	std::vector<double> y_values;			// contains y values of points
	std::vector<double> cells_vertices;		// contains cell vertices
	std::vector<unsigned int> locations;		// stores the start of vertex data for each cell

	x_values.reserve((size_t)num_points);			// reserve space for data
	y_values.reserve((size_t)num_points);
	cells_vertices.reserve((size_t)num_points*14);
	locations.reserve((size_t)num_points+1);
	locations.emplace_back(0);		// the first vertex for the first cell starts here

	unsigned int number_of_vertices_read{0};

	std::fstream fp;
	fp.open(correct_filename, std::ios::in);	// open file
	if (!fp.is_open()){
		std::cout<<"Error opening " << correct_filename << "\n";
		return 1;
	}

	for (int i = 0; i<num_points; i++){	// loop through number of points
		std::string one_line;
		std::getline(fp, one_line);	// read one line from correct_filename

		std::stringstream stream(one_line);
		std::string one_word;

		stream >> one_word;	// read in x_value of point
		x_values.emplace_back(std::stod(one_word));	// store xvalue of point

		stream >> one_word;	// read in y_value of point
		y_values.emplace_back(std::stod(one_word));	// store yvalue of point

		// read in vertices
		while(stream >> one_word){
			cells_vertices.emplace_back(std::stod(one_word));
			number_of_vertices_read++;
		}

		// done reading vertices, so store location data
		locations.emplace_back(number_of_vertices_read);
	}
	fp.close();


	// now we have stored all of the correct data
	// we slowly parse through filename_to_test data and see if it is accurate.

	fp.open(filename_to_test, std::ios::in);
	if (!fp.is_open()){
		std::cout<<"Error opening " << filename_to_test << "\n";
		return 1;
	}

	double x,y;
	std::vector<double> current_cell_vertices;
	current_cell_vertices.reserve(20);
	unsigned int number_of_vertices_in_this_cell{0};
	unsigned int total_number_of_cell_mismatches{0};
	unsigned int total_number_of_points_without_matches{0};

	for (int i = 0; i<num_points; i++){	// loop through number of points

		//cout<<"READ CELL ----------------------------------\n";
		std::string one_line;
		std::getline(fp, one_line);	// read one line from filename_to_test

		std::stringstream stream(one_line);
		std::string one_word;

		stream >> one_word;	// read in x_value of point
		try{
			x = std::stod(one_word);	// store xvalue of point
		} catch (const std::invalid_argument& e) {
			std::cout << "Invalid input: cannot convert to a number. Skipping cell\n";
			continue;
		} catch (const std::out_of_range& e) {
			std::cout << "Input is out of range for a double. Skipping cell\n";
			continue;
		}

		stream >> one_word;	// read in y_value of point
		try{
			y = std::stod(one_word);	// store yvalue of point
		} catch (const std::invalid_argument& e) {
			std::cout << "Invalid input: cannot convert to a number. Skipping cell\n";
			continue;
		} catch (const std::out_of_range& e) {
			std::cout << "Input is out of range for a double. Skipping cell\n";
			continue;
		}

		// read in vertices
		while(stream >> one_word){
			try{
				current_cell_vertices.emplace_back(std::stod(one_word));
			} catch (const std::invalid_argument& e) {
				std::cout << "Invalid input: cannot convert to a number. Skipping cell\n";
				continue;
			} catch (const std::out_of_range& e) {
				std::cout << "Input is out of range for a double. Skipping cell\n";
				continue;
			}
			number_of_vertices_in_this_cell++;		// count number of vertices in this cell
		}

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
								std::cout<<"Vertex mismatch ! compared " << current_cell_vertices[k] << " with " << cells_vertices[locations[j+k]] << "\n";
								mismatch = true;		// we have found a vertex mismatch
								break;
							}
						}
						if (mismatch){
							total_number_of_cell_mismatches++;	// increment the total number of mismatches found
						}

					} else {	// mismatch number of vertices.... yikes
						total_number_of_cell_mismatches++;	// increment the total number of mismatches found
						std::cout<<"Number of vertices mismatch !\n";
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
	std::cout<<"\n\nProgram complete\n";
	if(total_number_of_points_without_matches != 0) std::cout<<"Total number of points from " << filename_to_test << " that we couldn't find in " << correct_filename << "\n\t"<< total_number_of_points_without_matches << "\n\n";
	std::cout<<"Total number of cell mismatches: " << total_number_of_cell_mismatches;
	if (total_number_of_cell_mismatches > 20) std::cout<<"\t Be careful to ensure that boundaries used for both simple and complex programs were the same!";
	std::cout<<"\n";

	return 0;
}
