/**
 * @file voronoi_functions.cc
 * @brief This file contains functions for the program contained in simple_parallel_voronoi.cc,
 * and is indended to be compiled into an object file voronoi_functions.o.
 * 
 * The program contained in simple_parallel_voronoi.cc is an implementation of Daniel Reem's Projector Algorithm. See
 * Daniel Reem. The projector algorithm: A simple parallel algorithm for computing voronoi diagrams and delaunay graphs.
 * Theoretical Computer Science, 970:114054, 2023. 
 * ISSN = 0304-3975
 * DOI = {https://doi.org/10.1016/j.tcs.2023.114054}
 * URL = {https://www.sciencedirect.com/science/article/pii/S0304397523003675}
 * 
 * The header file voronoi_functions.h contains function declarations of the functions contained in this file.
 *   mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o voronoi_functions.o -c voronoi_functions.cc -lm
 * This file can be compiled with
 *   mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o voronoi_functions.o -c voronoi_functions.cc -lm
 * along with the header files voronoi.h and voronoi_functions.h.
 * @author E. Maher
 * @version 1.0
 * @date 02-07-2024
 */


#include <stdio.h>  
#include <unistd.h>  
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "voronoi3.0.h"		// function prototypes here
#include <cmath>
#include <mpi.h>
#include "voronoi_functions.h"


/**
 *	\brief This function reads boundary lines written in a file.
 */
int readBoundaryLines(char * boundary_filename, std::vector<line>& boundary_lines){
	std::ifstream fp;
	fp.open(boundary_filename, std::ifstream::in);    // open file to read
	if(! fp.is_open()){     // error check: make sure that the file is open
		std::cerr<<"Error: Could not open "<<boundary_filename<<" for reading boundary lines\n";
		exit(0);
	}
	// read the number of boundary lines
	int numberOfLines;
	fp>>numberOfLines;
	if(numberOfLines<3){  // error checking: make sure that number of lines is at least 3
		std::cout<<"Error: number of boundary lines in file must be greater than 2. Number of lines read was "<<numberOfLines<<"\n";
		fp.close();
		exit(0);
	}

	// read lines
		for (int it = 0; it < numberOfLines; it++){
			double x, y, c;
			fp >> x;
			fp >> y;
			fp >> c;
			boundary_lines.emplace_back(x, y, c);
		}
	fp.close();
	
	return 0;
}


/**
 *	\brief This function generates a default boundary region for the inputted points.
 *	
 *	The default boundary is simply a rectangle encompassing the inputted points with
 *	excess space as specified by the user with the flag -d val (default spacing is 2).
 *	The global variable boundary_lines (std::vector<line>) is populated with lines
 * 	of the form ax + by = c which represent halfplanes ax + by <= c.
 *	@param [in] points input points
 */
int generateDefaultBoundary(std::vector<point>& points, std::vector<line>& boundary_lines, double spacing){
	// determine extreme x and y coordinates
		double minx{points[0].x};
		double miny{points[0].y};
		double maxx{minx};
		double maxy{miny};
		for (auto it = points.begin()+1; it != points.end(); ++it){
			if ((*it).x < minx){        // new minimum x coordinate
				minx = (*it).x;
			} else if((*it).x > maxx){  // new maximum x coordinate
				maxx = (*it).x;
			}

			if ((*it).y < miny){        // new minimum y coordinate
				miny = (*it).y;
			} else if((*it).y > maxy){  // new maximum y coordinate
				maxy = (*it).y;
			}
		}
	boundary_lines.emplace_back(-1, 0, (-1)*(minx-spacing));
	boundary_lines.emplace_back(1, 0, maxx+spacing);
	boundary_lines.emplace_back(0, -1, (-1)*(miny-spacing));
	boundary_lines.emplace_back(0, 1, maxy+spacing);

	return 0;
}


/**
 *	\brief This function generates reads the command line arguments passed by the user for rank 0 in MPI_COMM_WORLD.
 *	
 *	@param [in] argc Number of arguments passed at command line
 *	@param [in] argv Arguments passed at command line
 *	@param [out] points_filename Address of character pointer to be filled with pointer to filename to read input points from
 *	@param [out] boundary_filename Address of character pointer to be filled with pointer to filename to read boundary lines from
 *	@param [out] spacing Address of a double to fill with requested spacing for computer generated boundary lines
 *	@return 0 if all necessary flags passed; -1 if program should end early
 */
int parseArguments0(int argc, char * argv[], char ** points_filename, char ** boundary_filename, double * spacing){
	int opt;
	int exit = 0;	// should we exit the program early ?
	int pu = 0;		// should we print the usage statement ?
	while ((opt = getopt(argc, argv, "p:b:id:h")) != -1){
		switch(opt){
			case 'p':
				*points_filename = optarg;
				std::cout<<"Reading points from "<<*points_filename<<"\n";
				break;
			case 'b':
				*boundary_filename = optarg;
				std::cout<<"Reading boundary lines from "<<*boundary_filename<<"\n";
				break;
			case 'h':
				pu = 1;
				exit = 1;
				break;
			case 'd':
				*spacing = atof(optarg);
				break;
			case 'i':
				printFileInfo();
				exit = 1;
				break;
			default:
				std::cout<<"Unknown option passed\n";
				pu = 1;
				return -1;
		}
	}

	if(*points_filename == nullptr){
		std::cout<<"ERROR: User failed to specify filename for input points\n\n";
		pu = 1;
		exit = 1;
	}
	if (pu) printUsage();
	if(exit){
		MPI_Finalize();
		return -1;
	}
	return 0;
}


/**
 *	\brief This function generates reads the command line arguments passed by the user for non-zero ranks in MPI_COMM_WORLD.
 *	
 *	@param [in] argc Number of arguments passed at command line
 *	@param [in] argv Arguments passed at command line
 *	@param [out] points_filename Address of character pointer to be filled with pointer to filename to read input points from
 *	@param [out] boundary_filename Address of character pointer to be filled with pointer to filename to read boundary lines from
 *	@param [out] spacing Address of a double to fill with requested spacing for computer generated boundary lines
 *	@return 0 if all necessary flags passed; -1 if program should end early
 */
int parseArguments(int argc, char * argv[], char ** points_filename, char ** boundary_filename, double * spacing){
	int opt;
	int exit = 0;
	while ((opt = getopt(argc, argv, "p:b:id:h")) != -1){
		switch(opt){
			case 'p':
				*points_filename = optarg;
				break;
			case 'b':
				*boundary_filename = optarg;
				break;
			case 'h':
				exit = 1;
				break;
			case 'd':
				*spacing = atof(optarg);
				break;
			case 'i':
				exit = 1;
				break;
			default:
				std::cout<<"Unknown option passed\n";
				return -1;
		}
	}

	if(exit){
		MPI_Finalize();
		return -1;
	}
	if(points_filename == nullptr){
		MPI_Finalize();
		return -1;
	}
	return 0;
}


/**
 *	\brief This function prints the usage statement.
 */
void printUsage(void){
	std::cout<<"Simple Parallel Voronoi Diagram program\nby: Eleanor Maher <mahere6@tcd.ie>\n";
	std::cout<<"This program will calculate the 2D Voronoi diagram in parallel for inputted points within a convex region.\nusage: ";
	std::cout<<"mpirun -np [number of processes] ./v_parallel_simple [options]\n";

	// options
	std::cout<<"        -p filename     : will cause the program to read input points from the txt file named [filename]\n";
	std::cout<<"        -b filename     : will cause the program to read boundary lines from the txt file named [filename]\n";
	std::cout<<"        -d value        : will cause the program to use the default boundary of a rectangle with padding\n";
	std::cout<<"                            of size [value] around extreme input points (default padding size: 2.0)\n";
	std::cout<<"        -h              : will print this usage statement\n";
	std::cout<<"        -i              : will print information about format of input files\n";

	// usage explanation
	std::cout<<"\n";
	std::cout<<"A filename with input points MUST be passed with the flag -p for the program to run.\n";
	std::cout<<"If no filename with boundary lines is specified, a default boundary will be used.\n";
	std::cout<<"This default boundary is a rectangle surrounding the input points with spacing of default size 2.0 around extreme input points.\n";
	std::cout<<"The padding size for the default rectangle boundary can be adjusted with the flag -d\n";
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
	std::cout<<"   \tx >= -14\n\tx <= 13\n\ty >= -6.5\n\ty <= 9\n";

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
 *	@param [out] plot A timestep_plot_data object which is adjusted and used to record some operations in this function.
 *	@return 0
 */
int find_Endpoint_and_Line(std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines){

	point p = *p_it;

	// first check if we have already computed endpoint
	if (direction.t > 0){
		return 0;
	}

	// then find intersection of ray from p in [direction] with boundary of world
	find_Intersection_with_Boundary(p, direction, boundary_lines);
	bool foundEndpoint = false;
	std::vector<std::vector<point>::iterator> skip_items (1, p_it);
	while (! foundEndpoint){
		point endpoint = p + (direction.t * direction);	// this is our endpoint
		// check whether the endpoint we have found is in the cell
		bool in_cell = true;
		double min_distance = p.distance(endpoint);		// distance from p to endpoint
		std::vector<point>::iterator closeNeighbour = p_it;					// closest neighbour to endpoint
		for (auto it = input_points.begin(); it != input_points.end(); ++it){		// loop through input points and find closest input point to endpoint and corresponding distance
			
			double current_distance = (*it).distance(endpoint);		// distance from current input point to endpoint
			if ( current_distance < min_distance){	// if we have found a closer point

				// check whether we should skip this point or not
				bool skip = false;
				for (auto& skip_it : skip_items){
					if (it == skip_it){
						skip = true;
						break;
					}
				}

				if (!skip){	// make sure that we aren't considering p itself as a closer input point
					min_distance = current_distance;
					closeNeighbour = it;
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
				if ((p.y - (*closeNeighbour).y) == 0){	// the the bisector is vertical
					direction.boundary = line{1, 0, (p.x + (*closeNeighbour).x)/2};
				} else {
					direction.boundary = line{(p.x-(*closeNeighbour).x)/(p.y-(*closeNeighbour).y), 1, (p.y+(*closeNeighbour).y + (p.x * p.x - ((*closeNeighbour).x * (*closeNeighbour).x))/(p.y-(*closeNeighbour).y))*0.5};
				}

			// suspected endpoint is now the intersection between the perpendicular bisector and ray from p
				direction.t = (direction.boundary.c - (direction.boundary)*p)/(direction.boundary * direction);

		}
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
 *	
 *	@param [in] p A point object, the point from which the ray is shot
 *	@param [in, out] direction An ordered_vector object, a unit vector representing the direction
 *	of the ray. Upon return the member variables boundary and t will be populated with the 
 *	boundary line which the ray intersects with and the distance to that intersection, respectively.
 *	@param [out] plot A timestep_plot_data object which is adjusted and used to record some operations in this function.
 *	@return 0
 */
int find_Intersection_with_Boundary(const point p, ordered_vector& direction, std::vector<line>& boundary_lines){
	double ti;

	// cycle through halfplanes in boundary
	auto it = boundary_lines.begin();
	bool foundFirstIntersection = false;
	while ((! foundFirstIntersection) && (it != boundary_lines.end())){	// first we find an initial intersection of the ray with boundary
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
	return 0;
}


/**
 *	\brief This function prints the final results of the program to file.
 *
 *	In particular, this function prints the final results of the progam in a
 *	way that can be read by the python matplotlib script Full_Diagram_Plot.py. First, 
 *	the input points are printed in the format "BKGDP x y". Then, for each input
 *	point, a single timestep is written; this reflects the order in which the program
 *	determined the cells. Each timestep simply contains a sorted list of vertices of
 *	the cell in counter-clockwise direction.
 *	
 *	@param [in] cells A point object
 *	@param [in] filename
 *	@param [in] input_points
 */
void print_diagram_plot_info_to_file(std::vector<cell_info>& cells, std::string const filename, std::vector<point>& input_points){
	// open file
	std::ofstream plot_file;
	plot_file.open(filename);
	if (!plot_file){	// error checking
		std::cerr<<"Failed to open file for writing plot data for Voronoi diagram";
		return;
	}

	// write input points to file
	for (auto it = input_points.begin(); it != input_points.end(); ++it){
		plot_file<<"BKGDP " << " " << (*it).x << " " << (*it).y << "\n";		// format: BKGDP 0.487 8.001
	}

	// now for each input point, we write one timestep
	int i = 0;
	for (auto cell_it = cells.begin(); cell_it != cells.end(); ++cell_it){
		// sort the cell
		(*cell_it).sort_vertices();

		plot_file << "Timestep " << i << "\n";
		i++;
		
		// print the vertices in counterclockwise order:
		for (auto vert_it = (*cell_it).vertices.begin(); vert_it != (*cell_it).vertices.end(); ++vert_it){
			plot_file << "V " << (*vert_it).x << " " << (*vert_it).y << "\n";
		}

		plot_file << "\n";
	}

	plot_file.close();	// close file
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
 *	@return An int, 1 if the point lies in the convez region, 0 otherwise.
 */
int in_region(point pt, std::vector<line>& boundary_lines){
	// cycle through boundary planes
	for (auto it = boundary_lines.cbegin(); it != boundary_lines.cend(); ++it){
		if ((*it) * pt > (*it).c + 1e-10){
			return 0;
		}
	}
	return 1;
}


/**
 *	\brief This function reads the points from a file.
 *
 *	@param [in] points_filename Filename to read points from
 *	@return An std::vector<point> filled with the points read
 */
std::vector<point> read_input_points(char * points_filename){
	std::ifstream fp;
	fp.open(points_filename, std::ifstream::in);	// open file to read
	if(! fp.is_open()){		// error check: make sure that the file is open
		std::cerr<<"Error: Could not open "<<points_filename<<" for reading input points\n";
		exit(0);
	}
	// read the number of points
		// cases to error check for: number of points is a double ? missing y value ? what if there aren't as many points as specified ? does the code work if there is only 1 point?
	size_t numberOfPoints;
	fp>>numberOfPoints;
	if(numberOfPoints<=0){  // error checking: make sure that number of points is a positive integer
		std::cout<<"Error: number of input points in file must be a positive integer. Number of points read was "<<numberOfPoints<<"\n";
		fp.close();
		exit(0);
	}
	// read points
	std::vector<point> input_points(numberOfPoints);
		for (auto it = input_points.begin(); it != input_points.end(); ++it){
			fp >> (*it).x;
			fp >> (*it).y;
		}
	fp.close();
	return input_points;
}


/**
 *	\brief This function divides a number of points between a number of processes,
 *	returning the starting and ending index of the points assigned to the calling process.
 *
 *	@param [in] rank The id of the calling process in the MPI Communicator
 *	@param [in] size The number of processes in the MPI Communicator
 *	@param [in] number_of_points The total number of points to divide between the processes in the MPI Communicator
 *	@param [out] start The index of the first point assigned to the calling process
 *	@param [out] end The index the point after the last point assigned to the calling process
 *	@return The number of points assigned to the calling process
 */
size_t determine_start_and_end(int rank, int size, int number_of_points, int * start, int * end){
	int nlocal, s, e, deficit;
	nlocal = number_of_points / size;
	s = rank * nlocal;
	deficit = number_of_points % size;
	s =  s + ((rank < deficit) ? rank : deficit);
	if (rank < deficit) nlocal++;
	e = s + nlocal;
	if (rank >= number_of_points) {		// if there are more processes than input points
		*start = number_of_points;
		*end = number_of_points;
		return 0;
	}
	if (e >= number_of_points || rank == size-1) *end = number_of_points;
	else *end = e;
	*start = s;
	return (size_t)nlocal;
}

/**
 *	\brief This function prints the results of simple_parallel_voronoi.cc to a file.
 *	A collective function.
 *
 *	@param [in] filename The name of the file to print results to
 *	@param [in] rank The id of the calling process in the MPI Communicator
 *	@param [in] size The number of processes in the MPI Communicator
 *	@param [in] input_points The std::vector of input points assigned to the process
 *	@param [in] cells The process's computed cells for its assigned points
 *	@param [in] start The index of the first point assigned to the calling process in the original point input file
 *	@return 0 upon success, -1 if failed to open file for writing
 */
int print_cells_to_file(std::string filename, int rank, int size, std::vector<cell_info>& cells){

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

				if ( (*cell_it).vertices.empty() ) continue;

				fp << (*cell_it).p.x << " " << (*cell_it).p.y << " ";

				// loop through vertices in cell:
				for (auto vert_it = (*cell_it).vertices.begin(); vert_it != (*cell_it).vertices.end(); ++vert_it){
					fp <<(*vert_it).x << " " << (*vert_it).y << " ";
				}

				fp << "\n";
			}

			fp.close();	// close file
		}

		MPI_Barrier(MPI_COMM_WORLD);	// synchronize

	}

	return 0;
}
