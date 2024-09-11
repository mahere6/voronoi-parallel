/**
 * @file random_pts_gen.cc
 * @brief
 * @author E. Maher
 * @version 1.0
 * @date 
 */

#include <stdio.h>  
#include <unistd.h>  
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <chrono>

// make with
//		g++ -o random_pts_gen random_pts_gen.cc -Wall -Wextra -std=c++20


void printUsage(void);


int main(int argc, char *argv[]){

	// parse arguments
	// -n [number of points]
	// -s [width of area]
	// -f filename to print to
	
	std::string filename;
	int opt;
	int exit;
	int num_points{0};
	double size{0.0};
	while ((opt = getopt(argc, argv, "n:s:f:h")) != -1){
		switch(opt){
			case 'f':
				filename = optarg;
				std::cout<<"Printing points to "<<filename<<"\n";
				break;
			case 'h':
				printUsage();
				exit = 1;
				break;
			case 'n':
				num_points = atoi(optarg);
				if (num_points < 1){
					std::cout<<"Error: user passed flag \"-n "<<num_points<<"\" which is not an acceptable value. Integer value must be greater than 0.";
					exit = 1;
				}
				break;
			case 's':
				size = atof(optarg);
				if (size<=0){
					std::cout<<"Error: user passed flag \"-s "<<size<<"\" which is not an acceptable value. Floating-point value must be greater than 0.";
					exit = 1;
				}
				break;
			default:
				std::cout<<"Unknown option passed\n";
		}
	}

	if(exit){
		return -1;
	}
	if(filename.empty() || (size == 0.0) || (num_points == 0)){
		std::cout<<"ERROR: User failed to specify input information\n\n";
		printUsage();
		return -1;
	}

	// prepare random number generator engine
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::ranlux48 gen (seed);

	// open file
	std::ofstream fp;
	fp.open(filename);
	if (!fp){	// error checking
		std::cerr<<"Failed to open file for writing point data";
		return -1;
	}

	// print num points
	fp<<num_points<<"\n";

	// print points
	for (int i = 0; i < num_points; i++){
		fp<< ((((double) gen() ) / gen.max()) - 0.5) * 2 * size << " " << ((((double) gen() ) / gen.max()) - 0.5) * 2 * size << "\n";
	}

	fp.close();		// close file


	return 0;
}


void printUsage(void){
	std::cout<<"This program prints a specified number of randomly generated points to a file\n";
	std::cout<<"Usage:\n";
	std::cout<<"         -n [val]             will set the number of points to print to val\n";
	std::cout<<"         -s [val]             will set the range for generated points to [-val, val]\n";
	std::cout<<"         -f [filename]        will print points to the file named filename\n";
	std::cout<<"         -h                   will print this usage statement\n\n";
	std::cout<<"The program will not run unless filename, number of points, and range of points is specified.\n\n";
}