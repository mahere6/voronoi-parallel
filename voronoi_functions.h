/**
 * @file voronoi_functions.h
 * @brief Header file for voronoi_functions.cc
 * @author E. Maher
 * @version 1.0
 * @date 02-07-2024
 */
#include "voronoi3.0.h"

int		parseArguments						(int argc, char * argv[], char ** points_filename, char ** boundary_filename, double * spacing);
int		parseArguments0						(int argc, char * argv[], char ** points_filename, char ** boundary_filename, double * spacing);
void	printUsage							(void);
void	printFileInfo						(void);
int		generateDefaultBoundary				(std::vector<point>& points, std::vector<line>& boundary_lines, double spacing);
int		readBoundaryLines					(char * boundary_filename, std::vector<line>& boundary_lines);
int		find_Intersection_with_Boundary		(const point p, ordered_vector& direction, std::vector<line>& boundary_lines);
int		find_Endpoint_and_Line				(std::vector<point>::iterator p_it, ordered_vector& direction, std::vector<point>& input_points, std::vector<line>& boundary_lines);
void	print_diagram_plot_info_to_file		(std::vector<cell_info>& cells, std::string const filename, std::vector<point>& input_points);
int		in_region							(point pt, std::vector<line>& boundary_lines);
size_t	determine_start_and_end				(int rank, int size, int number_of_points, int * start, int * end);
std::vector<point>	read_input_points		(char * points_filename);
int		print_cells_to_file					(std::string filename, int rank, int size, std::vector<cell_info>& cells);