/**
 * @file simple_parallel_voronoi.cc
 * @brief An implementation of Daniel Reem's Projector Algorithm. See
 * Daniel Reem. The projector algorithm: A simple parallel algorithm for computing voronoi diagrams and delaunay graphs.
 * Theoretical Computer Science, 970:114054, 2023. 
 * ISSN = 0304-3975
 * DOI = {https://doi.org/10.1016/j.tcs.2023.114054}
 * URL = {https://www.sciencedirect.com/science/article/pii/S0304397523003675}
 * 
 * This version can be used for timing purposes.
 * 
 * This parallel implementation simply divides the input points as equally as possible
 * among processes and lets each process compute cells independently. No other load-balancing
 * effort is made.
 * 
 * This file can be compiled with
 *   mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o v_parallel_simple simple_parallel_voronoi.cc voronoi_functions.o -lm -lstdc++
 * along with the header files voronoi.h and voronoi_functions.h, and the object file voronoi_functions.o.
 * @author E. Maher
 * @version 2.0
 * @date 02-07-2024
 */

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "voronoi.h"
#include "voronoi_functions.h"


int main(int argc, char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	double starttime_total, endtime_total, starttime_readingpoints, endtime_readingpoints;	// used for timing
	double starttime_cells, endtime_cells, starttime_writing, endtime_writing;				// used for timing

	starttime_total = MPI_Wtime();

	// read user input from command line
	char * points_filename{nullptr};		// filename to read input points from
	char * boundary_filename{nullptr};		// filename to read boundary lines from
	double spacing{2.0};					// spacing size for default boundary lines
	std::vector<line> boundary_lines;		// vector to populate with boundary halfplanes

	int should_we_end_program = 0;
	if(rank == 0) should_we_end_program = parseArguments0(argc, argv, &points_filename, &boundary_filename, &spacing);
	else should_we_end_program = parseArguments(argc, argv, &points_filename, &boundary_filename, &spacing);

	if (should_we_end_program) return 0;

	// read input points
	starttime_readingpoints = MPI_Wtime();
	std::vector<point> input_points = read_input_points(points_filename);
	endtime_readingpoints = MPI_Wtime();
	int ntotal = input_points.size();

	// determine boundary lines
	if (boundary_filename == nullptr) generateDefaultBoundary(input_points, boundary_lines, spacing);
	else readBoundaryLines(boundary_filename, boundary_lines);

	// determine which points belong to this process
	int s, e = 0;
	size_t nlocal = determine_start_and_end(rank, size, ntotal, &s, &e);

	auto start_it= input_points.begin() + s; 
	auto end_it= input_points.begin() + e; 

	// calculate cells for points
	std::vector<cell_info> cells;		//	vector to hold cell data
	cells.reserve(nlocal);

	// loop through input points
	starttime_cells = MPI_Wtime();
	for (auto point_it = start_it; point_it != end_it; ++point_it){
		cells.emplace_back(*point_it);
		point p{*point_it};

		std::vector<ordered_vector> rays(3);
		rays.reserve(20);
		rays[0] = ordered_vector{0,1, 1};
		rays[1] = ordered_vector{cos(3.14159265358979323846/6), (-1)*sin(3.14159265358979323846/6), 2};
		rays[2] = ordered_vector{(-1)*cos(3.14159265358979323846/6), (-1)*sin(3.14159265358979323846/6), 3};

		// Create a queue of edges to cycle through
		std::vector<subwedge> WedgeQueue(3);
		WedgeQueue.reserve(20);
		WedgeQueue[0] = subwedge{&rays[2], &rays[0]};	// order is counterclockwise starting at noon
		WedgeQueue[1] = subwedge{&rays[1], &rays[2]};
		WedgeQueue[2] = subwedge{&rays[0], &rays[1]};


		size_t edge_spot = 0;		// loop through remaining edges
		while(WedgeQueue.begin() + (long)edge_spot != WedgeQueue.end()){
			// find endpoints and bisecting lines
			find_Endpoint_and_Line(point_it, *(WedgeQueue[edge_spot]).vector1, input_points, boundary_lines);
			find_Endpoint_and_Line(point_it, *(WedgeQueue[edge_spot]).vector2, input_points, boundary_lines);

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
						double idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
						if (((WedgeQueue[edge_spot]).vector2->index) == 1){
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 4)/2;
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
					double idx = (((WedgeQueue[edge_spot]).vector1->index) + ((WedgeQueue[edge_spot]).vector2->index))/2;
					if (((WedgeQueue[edge_spot]).vector2->index) == 1){
						idx = (((WedgeQueue[edge_spot]).vector1->index) + 4)/2;
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
						idx = (((WedgeQueue[edge_spot]).vector1->index) + 4)/2;
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
					for (auto it = input_points.begin(); it != input_points.end(); ++it){		// loop through input points and find closest input point to endpoint and corresponding distance
						double check_dist = (*it).distance(intersection);
						if ( check_dist < min_distance){	// if we have found a closer point
							if ((check_dist - min_distance)*(check_dist - min_distance) > 1e-20){	// make sure that we aren't considering any points on the circle centred at intersection and passing through p
								in_cell = false;	// we have found a closer input point, hence we are not in the cell yet
								break;
							}
						}
					}
					if (in_cell) {	// we have found a vertex
						if (!in_region(intersection, boundary_lines)){	// if not in region...
							double idx;
							if (((WedgeQueue[edge_spot]).vector2->index) != 1){
								idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
							} else {
								idx = (((WedgeQueue[edge_spot]).vector1->index) + 4)/2;
							}
							rays.emplace_back((intersection - p) / p.distance(intersection), idx);

							// add a new subedge to represent second half of this edge
							WedgeQueue.emplace_back(&rays.back(), (WedgeQueue[edge_spot]).vector2);

							// then adjust this current edge to represent first half of this edge
							(WedgeQueue[edge_spot]).vector2 = (&rays.back());
						} else {
							// save vertex data
							cells.back().add_vertex(intersection, (WedgeQueue[edge_spot]).vector1->index, (WedgeQueue[edge_spot]).vector1->boundary);
							edge_spot++;
						}

					} else {		// we have not found a vertex and we split
						double idx = ((WedgeQueue[edge_spot]).vector1->index + (WedgeQueue[edge_spot]).vector2->index)/2;
						if (((WedgeQueue[edge_spot]).vector2->index) == 1){
							idx = (((WedgeQueue[edge_spot]).vector1->index) + 4)/2;
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

		cells.back().sort_vertices();

	}	// end iteration through local points
	endtime_cells = MPI_Wtime();

	std::cout<<"Rank " << rank << " Complete\n";

	// print everything to file
	starttime_writing = MPI_Wtime();
	print_cells_to_file("results_simple_parallel_" + std::to_string(ntotal) + ".txt", rank, size, cells);
	endtime_writing = MPI_Wtime();
	if (rank == 0){
		std::cout<<"See results_simple_parallel_" + std::to_string(ntotal) + ".txt for diagram results and timing_results_simple_" + std::to_string(ntotal) + ".txt for timing results.\nThe file results_simple_parallel_" + std::to_string(ntotal) + ".txt can be read by the python matplotlib script Results_Plot.py\n\n";
	}

	endtime_total = MPI_Wtime();

	if (rank != 0){		// send timing results to rank 0 to print
		double timing_results[] = {endtime_readingpoints - starttime_readingpoints, endtime_cells - starttime_cells, endtime_writing - starttime_writing, endtime_total - starttime_total};
		MPI_Send(&timing_results, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

	} else {			// recieve timing results to rank 0 to print
		double timing_results[(long unsigned int)size*4];
		timing_results[0] = endtime_readingpoints - starttime_readingpoints;
		timing_results[1] = endtime_cells - starttime_cells;
		timing_results[2] = endtime_writing - starttime_writing;
		timing_results[3] = endtime_total - starttime_total;

		for (int i = 1; i < size; i++){
			MPI_Recv(&timing_results[4*i], 4, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// print timing results to file
		std::string timing_filename = "timing_results_simple_" + std::to_string(ntotal) + ".txt";

		std::ofstream fp;
		fp.open(timing_filename);
		if(!fp){	// error checking
			std::cerr<<"Failed to open file for writing timing data";
			return -1;
		}
		fp << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		fp << "TIMING RESULTS (seconds)\n";
		fp << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		fp << "Process:    Reading points:    Computing cells:    Writing to file:    Total program time:\n";
		fp << std::fixed;
    	fp << std::setprecision(5);
		for (int i = 0; i< size; i++) {
			fp << "  " << i << "\t\t" << timing_results[4*i] << "\t\t   " << timing_results[4*i+1] << "\t\t" << timing_results[4*i+2] << "\t\t    " << timing_results[4*i+3] << "\n";
		}
		fp << "\n";
		fp.close();
	}

	MPI_Finalize();

	return 0;
}	// end main function
