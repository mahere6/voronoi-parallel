/**
 * @file sequential_voronoi.cc
 * @brief An implementation of Daniel Reem's Projector Algorithm. See
 * Daniel Reem. The projector algorithm: A simple parallel algorithm for computing voronoi diagrams and delaunay graphs.
 * Theoretical Computer Science, 970:114054, 2023. 
 * ISSN = 0304-3975
 * DOI = {https://doi.org/10.1016/j.tcs.2023.114054}
 * URL = {https://www.sciencedirect.com/science/article/pii/S0304397523003675}
 * 
 * This version can be used for timing purposes.
 * 
 * This file can be compiled with
 *   g++ -Wall -Wextra -std=c++20 -Wsign-conversion -o sequential_voronoi sequential_voronoi.cc -lm -lstdc++
 * along with the header file voronoi.h
 * @author E. Maher
 * @version 2.0
 * @date 02-07-2024
 */

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "voronoi.h"
#include <cmath>
#include <chrono>


int		parseArguments						(int argc, char **argv);
void	printUsage							(void);
void	printFileInfo						(void);
int		generateDefaultBoundary				(std::vector<point>& points);
int		readBoundaryLines					(void);
int		find_Intersection_with_Boundary		(const point p, ordered_vector& direction);
int		find_Endpoint_and_Line				(std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points);
int		in_region							(point pt);
void	print_results						(std::vector<cell_info>& cells, std::string const filename, std::vector<point>& input_points);

char * points_filename{nullptr};		// filename to read input points from
char * boundary_filename{nullptr};		// filename to read boundary lines from
double spacing{2.0};					// spacing size for default boundary lines
std::vector<line> boundary_lines;		// vector to populate with boundary halfplanes
bool output{false};

int main(int argc, char *argv[]){
	auto s_all = std::chrono::high_resolution_clock::now();		// timing
	parseArguments(argc, argv);

	// -------------------------------------------------------------------------------------
	// ----------------------------  READ INPUT POINTS  ------------------------------------
	// -------------------------------------------------------------------------------------
	auto s_points = std::chrono::high_resolution_clock::now();		// timing
	std::ifstream fp;
	fp.open(points_filename, std::ifstream::in);	// open file to read
	if(! fp.is_open()){		// error check: make sure that the file is open
		std::cerr<<"Error: Could not open "<<points_filename<<" for reading input points\n";
		exit(0);
	}
	// read the number of points
		// cases to maybe add error checking for: number of points is a double ? missing y value ? what if there aren't as many points as specified ? does the code work if there is only 1 point?
	size_t numberOfPoints;
	fp>>numberOfPoints;
	if(numberOfPoints<=0){  // error checking: make sure that number of points is a positive integer
		std::cout<<"Error: number of input points in file must be a positive integer. Number of points read was "<<numberOfPoints<<"\n";
		fp.close();
		exit(0);
	}
	if(output) std::cout<<"Number of input points: "<<numberOfPoints<<"\n";
	// read points
	std::vector<point> input_points(numberOfPoints);
	if(output) std::cout<<"Reading points...\t";
		for (auto it = input_points.begin(); it != input_points.end(); ++it){
			fp >> (*it).x;
			fp >> (*it).y;
		}
	fp.close();
	if(output) std::cout<<"Done\n";
	auto e_points = std::chrono::high_resolution_clock::now();		// timing

	// --------------------------------------------------------------------------------------
	// -------------------------------  BOUNDARY  -------------------------------------------
	// --------------------------------------------------------------------------------------
	if(boundary_filename == nullptr){	// generate default boundary
		if(output) std::cout<<"Generating default boundary\n";
		generateDefaultBoundary(input_points);
	} else {	// use specified boundary lines
		if(output) std::cout<<"Reading in boundary half-planes\n";
		readBoundaryLines();
	}

	if(output) {
		std::cout<<"Boundary lines are:\n\t";
		for(auto it = boundary_lines.begin(); it != boundary_lines.end(); ++it){
			(*it).print_halfplane();
			std::cout<<"\n\t";
		}
		std::cout<<"\n";
	}


	// -------------------------------------------------------------------------------------
	// ----------------------------- BEGIN ALGORITHM ---------------------------------------
	// -------------------------------------------------------------------------------------

	std::vector<cell_info> cells;		//	vector to hold cell data
	cells.reserve(input_points.size());
	int num_its = 0;	// count total number of times we've iterated through WedgeQueue

	// LOOP THROUGH INPUT POINTS
	auto s_compute = std::chrono::high_resolution_clock::now();		// timing

	if(output) std::cout<<"Looping through input points...\n";
	unsigned int index_through_points = 0;
	for (auto point_it = input_points.begin(); point_it != input_points.end(); ++point_it){
		cells.emplace_back(*point_it);
		
		point p{*point_it};


		std::vector<ordered_vector> rays(4);
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


		size_t edge_spot = 0;
		while(WedgeQueue.begin() + (long)edge_spot != WedgeQueue.end()){
			num_its++;

			// find endpoints and bisecting lines
			find_Endpoint_and_Line(point_it, *(WedgeQueue[edge_spot]).vector1, input_points);
			find_Endpoint_and_Line(point_it, *(WedgeQueue[edge_spot]).vector2, input_points);

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
						if (in_region(intersection)){	// if we are in the world...
							// save vertex data
							cells.back().add_vertex(intersection, (WedgeQueue[edge_spot]).vector1->index, (WedgeQueue[edge_spot]).vector1->boundary);
							// now remove edge from list
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
		}

		cells.back().sort_vertices();
		if (output){
			index_through_points++;
			if (index_through_points % 500 == 0){
				std::cout<< "\t" << 100*index_through_points / input_points.size() << "%\n";
			}
		}

	}	// end iterations through input points

	auto e_compute = std::chrono::high_resolution_clock::now();		// timing


	// print everything to file
	auto s_write = std::chrono::high_resolution_clock::now();		// timing
	print_results(cells, "results_seq_" + std::to_string(numberOfPoints) + ".txt", input_points);
	auto e_write = std::chrono::high_resolution_clock::now();		// timing

	auto e_all = std::chrono::high_resolution_clock::now();		// timing

	// timing
	auto d_all = std::chrono::duration_cast<std::chrono::milliseconds>(e_all - s_all);
	auto d_points = std::chrono::duration_cast<std::chrono::milliseconds>(e_points - s_points);
	auto d_compute = std::chrono::duration_cast<std::chrono::milliseconds>(e_compute - s_compute);
	auto d_write = std::chrono::duration_cast<std::chrono::milliseconds>(e_write - s_write);

	double time_all = ((double) d_all.count()) / 1000.0;
	double time_points = ((double) d_points.count()) / 1000.0;
	double time_compute = ((double) d_compute.count()) / 1000.0;
	double time_write = ((double) d_write.count()) / 1000.0;


	std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
	std::cout << "      TIMING RESULTS (seconds)\n";
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
	std::cout << "      Read points:        " << time_points << "\n";
	std::cout << "      Compute cells:      " << time_compute << "\n";
	std::cout << "      Write to file:      " << time_write << "\n";
	std::cout << "      Total time taken:   " << time_all << "\n";

	std::cout<<"\nProgram complete\nSee results_seq_" + std::to_string(numberOfPoints) + ".txt for diagram results and timing_results_sequential_" + std::to_string(numberOfPoints) + ".txt for timing results\n";
	std::cout<<"The file results_seq_" + std::to_string(numberOfPoints) + ".txt can be read by the python matplotlib script Results_Plot.py\n";


	std::ofstream fpo;
	fpo.open("timing_results_sequential_" + std::to_string(numberOfPoints) + ".txt");
	if(!fpo){	// error checking
		std::cerr<<"Failed to open file for writing timing data";
		return -1;
	}
	fpo << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
	fpo << "      TIMING RESULTS (seconds)\n";
	fpo << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
	fpo << "      Read points:        " << time_points << "\n";
	fpo << "      Compute cells:      " << time_compute << "\n";
	fpo << "      Write to file:      " << time_write << "\n";
	fpo << "      Total time taken:   " << time_all << "\n";

	fpo.close();

	return 0;
}


/**
 *	\brief This function reads boundary lines written in a file.
 */
int readBoundaryLines(){
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
int generateDefaultBoundary(std::vector<point>& points){
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
 *	\brief This function reads the command line arguments passed by the user and will
 *	exit if needed.
 */
int parseArguments(int argc, char * argv[]){
	int opt;
	int exit = 0;
	while ((opt = getopt(argc, argv, "p:b:id:ho")) != -1){
		switch(opt){
			case 'p':
				points_filename = optarg;
				std::cout<<"Reading points from "<<points_filename<<"\n";
				break;
			case 'b':
				boundary_filename = optarg;
				std::cout<<"Reading boundary lines from "<<boundary_filename<<"\n";
				break;
			case 'h':
				printUsage();
				exit = 1;
				break;
			case 'o':
				output = true;
				break;
			case 'd':
				spacing = atof(optarg);
				break;
			case 'i':
				printFileInfo();
				exit = 1;
				break;
			default:
				std::cout<<"Unknown option passed\n";
				printUsage();
				return -1;
		}
	}

	if(exit){
		std::exit(0);
	}
	if(points_filename == nullptr){
		std::cout<<"ERROR: User failed to specify filename for input points\n\n";
		printUsage();
		std::exit(0);
	}
	return 0;
}


/**
 *	\brief This function prints the usage statement.
 */
void printUsage(void){
	std::cout<<"Voronoi Diagram program\nby: Eleanor Maher <mahere6@tcd.ie>\nThis program will calculate the 2D Voronoi diagram for inputted points within a convex region.\nusage: ";
	std::cout<<"./sequential_voronoi [options]\n";

	// options
	std::cout<<"        -p filename     : will cause the program to read input points from the txt file named [filename]\n";
	std::cout<<"        -b filename     : will cause the program to read boundary lines from the txt file named [filename]\n";
	std::cout<<"        -d value        : will cause the program to use the default boundary of a rectangle with padding of\n";
	std::cout<<"                          size [value] around extreme input points (default padding size: 2.0)\n";
	std::cout<<"        -h              : will print this usage statement\n";
	std::cout<<"        -i              : will print information about format of input files\n";
	std::cout<<"        -o              : will cause the program to display output indicating the progress of the algorithm\n";

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
 *	@return 0
 */
int find_Endpoint_and_Line(std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points){

	point p = *p_it;

	// first check if we have already computed endpoint
	if (direction.t > 0){
		return 0;
	}


	// then find intersection of ray from p in [direction] with boundary of world
	find_Intersection_with_Boundary(p, direction);
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
 *	@return 0
 */
int find_Intersection_with_Boundary(const point p, ordered_vector& direction){
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
 *	In particular, this function prints one line of data for each cell:
 *	the first two values are the x and y values of the input point for
 *	which the cell was generated; the subsequent entries indicate the
 *	vertices of the cell in counter-clockwise direction.
 *
 *	The file output by this function can be read by the matplotlib 
 *	script Results_Plot.py.
 *	
 *	@param [in] cells A point object
 *	@param [in] filename
 *	@param [in] input_points
 */
void print_results(std::vector<cell_info>& cells, std::string const filename, std::vector<point>& input_points){
	// open file
	std::ofstream fp;
	fp.open(filename);
	if (!fp){	// error checking
		std::cerr<<"Failed to open file for writing final data for Voronoi diagram";
		return;
	}

	// now for each cell, we write one line
	auto it = input_points.begin();
	for (auto cell_it = cells.begin(); cell_it != cells.end(); ++cell_it){
		// sort the cell
		(*cell_it).sort_vertices();

		fp << (*it).x << " " << (*it).y << " ";		// print the point
		it++;
		
		// print the vertices in counterclockwise order:
		for (auto vert_it = (*cell_it).vertices.begin(); vert_it != (*cell_it).vertices.end(); ++vert_it){
			fp << (*vert_it).x << " " << (*vert_it).y << " ";
		}
		fp << "\n";
	}

	fp.close();	// close file
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
int in_region(point pt){
	// cycle through boundary planes
	for (auto it = boundary_lines.cbegin(); it != boundary_lines.cend(); ++it){
		if ((*it) * pt > (*it).c + 1e-10){
			return 0;
		}
	}
	return 1;
}
