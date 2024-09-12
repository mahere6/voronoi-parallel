Voronoi Diagrams in Parallel: Investigating Ray-Shooting and Daniel Reem's Projector Algorithm
by Ellie Maher
In partial fulfillment of MSc in High-performance Computing in the School of Mathematics

The goal of this project was to explore the parallelization of Voronoi diagram generation 
through the use of Daniel Reem's ray-shooting Projector Method. A sequential implementation
was developed along with two different parallel implementations that take in a set of points
and generate the corresponding Voronoi diagram information.

-----------------------------------------------------------------
FILES INCLUDED IN THIS SUBMISSION:
For compiling the programs:
  Makefile

For displaying output from programs:
  Results_Plot.py
  Voronoi_Plot_Slider.py
  What_Is_Going_On_Plot.py

For structs:
  voronoi.h

For the sequential implementation of Reem's algorithm:
  sequential_voronoi.cc
  sequential_voronoi_output.cc

For the simple parallel implementation of Reem's algorithm:
  simple_parallel_voronoi.cc
  voronoi_functions.h
  voronoi_functions.cc

For the complex parallel implementation of Reem's algorithm:
  complex_parallel_voronoi.cc
  complex_parallel_voronoi_output.cc

For generating random points:
  random_pts_gen.cc

For testing the accuracy of results generated:
  test_complex_accuracy.cc

-----------------------------------------------------------------
RUNNING THE CODE

All of the C++ programs accept the flag -h when run to clarify 
how exactly the program can be used. 

Making Everything:
  The Makefile includes everything needed to compile the code on 
  puffin with the command 'make'. Furthermore, there are additional
  commands available:
    "make clean" removes all executables and object files;
    "make fresh" removes all output files generated except those
      containing timing results; and
    "make fresh_timing" removes all output files containing timing
      results.
      
The sequential implementation:
  Two versions of the sequential implementation are provided: one
  for timing purposes, and another that records information about 
  every single step that the sequential implementation makes when 
  calculating the diagram. The latter implementation runs extremely
  slowly, but is useful for anyone curious about Reem's ray-
  shooting algorithm because the recorded information can be read
  and visualized by an included Python matplotlib script, 
  Voronoi_Plot_Slider.py.
  
  The former is included in the file sequential_voronoi.cc and 
  compiles into the executable 
    sequential_voronoi
  This can be run with the command
    ./sequential_voronoi [options]
        -p filename     : will cause the program to read input points 
                          from the txt file named [filename]
        -b filename     : will cause the program to read boundary lines 
                          from the txt file named [filename]
        -d value        : will cause the program to use the default 
                          boundary of a rectangle with padding of
                          size [value] around extreme input points 
                          (default padding size: 2.0)
        -h              : will print the usage statement
        -i              : will print information about format of input files
        -o              : will cause the program to display output 
                          indicating the progress of the algorithm

A filename with input points MUST be passed with the 
flag -p for the program to run.
If no filename with boundary lines is specified, a default 
boundary will be used. This default boundary is a rectangle surrounding 
the input points with spacing of default size 2.0 around extreme input points.
The padding size for the default rectangle boundary can be adjusted with the flag -d.

See the section below for the format of input files.

The latter is included in the file sequential_voronoi_output.cc
and compiles into the executable
  sequential_voronoi_output
and can be run with the command
  ./sequential_voronoi_output [options]
with the exact same options as sequential_voronoi.

The only reason you would use the executable sequential_voronoi_output
would be to visualize the progress of the algorithm; it's quite slow.
And I recommend only inputting a small problem size, say 5 points; that's
really all you need to visualize the process. The information
required for visualizing the operation of the algorithm is produced by 
the program in a file "plot_info.txt" which can be read by the Python matplolib
script Voronoi_Plot_Slider.py with the command
  python ./Voronoi_Plot_Slider.py

Each program produces Voronoi diagram information in the file named
  results_seq_[num_points].txt
where [num_points] is the number of points inputted to the program. 
This file can be read and visualized by the Python matplotlib script
Results_Plot.py by adjusting the variable
  FileName
in the script to be results_seq_[num_points].txt and then using the command
  python ./Results_Plot.py

 Lastly, the executable sequential_voronoi produces a file named
   timing_results_sequential_[num_points].txt
 containing timing results.


The Simple Parallel Implementation:
  The simple parallel implementation is included with the files
    simple_parallel_voronoi.cc
    voronoi_functions.cc
    voronoi_functions.h
  and compiles into the executable
    v_parallel_simple
  which can be run using the command
    mpirun -np [number of processes] ./v_parallel_simple [options]
        -p filename     : will cause the program to read input points 
                          from the txt file named [filename]
        -b filename     : will cause the program to read boundary lines 
                          from the txt file named [filename]
        -d value        : will cause the program to use the default boundary 
                          of a rectangle with padding of size [value] around 
                          extreme input points (default padding size: 2.0)
        -h              : will print the usage statement
        -i              : will print information about format of input files

A filename with input points MUST be passed with the 
flag -p for the program to run. If no filename with boundary 
lines is specified, a default boundary will be used. This default 
boundary is a rectangle surrounding the input points with 
spacing of default size 2.0 around extreme input points.
The padding size for the default rectangle boundary can be 
adjusted with the flag -d.

See the section below for the format of input files.

The program produces two output files:
  results_simple_parallel_[num_points].txt 
contains diagram results, and
  timing_results_simple_[num_points].txt
contains timing results. num_points is the number of
points inputted to the program.

The first file can be read and visualized by the Python matplotlib 
script Results_Plot.py by adjusting the variable
  FileName
in the script to be results_simple_parallel_[num_points].txt and 
then using the command
  python ./Results_Plot.py


The Complex Parallel Implementation:
  Similarly to the sequential implementation, two versions 
  of the complex parallel implementation are provided. One is 
  for timing purposes, and the other records information about
  what each individual process gets up to throughout the execution
  of the program. The former implementation is included in the file
    complex_parallel_voronoi.cc
  and compiles into the executable 
    v_parallel_complex
  The latter is included in the file 
    complex_parallel_voronoi_output.cc
  and compiles into the executable
    v_parallel_complex_output

  v_parallel_complex can be run with the command
    mpirun -np [value] ./v_parallel_complex [options]
        -p filename     : will cause the program to read input 
                          points from the txt file named [filename]
        -d value        : will cause the program to generate the world
                          boundary as a rectangle with padding of 
                          size [value] around extreme input points 
                          (default padding size: 2.0)
        -h              : will print this usage statement
        -i              : will print information about format of input files
        -x value        : will specify the 2D processor layout by setting 
                          the number of processes in the x-dimension
                          to be [value]. Thus the processor grid will 
                          be [value] by [total_number_of_processes / value]
        -o value        : will cause the program to print information about the 
                          operation of the algorithm as running on rank [value]
        -t              : will test the accuracy of the results generated 
                          against the cell data in the file named 
                             'results_simple_parallel_[value].txt'
                          where [value] is the number of points specified
                          after the flag -p
        -u filename     : will test the accuracy of the results generated 
                          against the cell data in [filename].

A filename with input points MUST be passed with the 
flag -p for the program to run.
The default boundary is a rectangle surrounding the input 
points with spacing of default size 2.0 around extreme input points.
The padding size for the default rectangle boundary can be 
adjusted with the flag -d.
The number of processes in the x-dimension MUST be 
specified. This must be a positive integer which divides the
total number of processes specified (with the flag -np for OpenMPI).
The flag -u overrides the flag -t.

See the section below for the format of input files.

The executable v_parallel_complex_output can be run with the command
  mpirun -np [value] ./v_parallel_complex [options]
with the same options and requirements above as for the 
executable v_parallel_complex, however without any of the accuracy
testing flags (-t and -u).

Both executables write computed diagram data to the file
  results_complex_parallel_[num_points].txt
and write timing results to the file
  timing_results_complex_[num_points].txt
where num_points is the total number of points inputted to the 
program. The file results_complex_parallel_[num_points].txt
can be read and visualized by the Python matplotlib 
script Results_Plot.py by adjusting the variable
  FileName
in the script to be results_complex_parallel_[num_points].txt and 
then using the command
  python ./Results_Plot.py

The executable v_parallel_complex_output also produces a text
file for each processor used, revealing information about
the operation of that processor. For a process with rank [rank],
the corresponding output file is 
  whats_going_on_rank[n].txt
and this file can be read and visualized by the Python matplotlib 
script What_Is_Going_On_Plot.py by adjusting the variable
  FileName
in the script to be whats_going_on_rank[n].txt and 
then using the command
  python ./Results_Plot.py
This produces a timestep plot that shows in order:
  1. The initial points held by the process in light blue;
  2. The local cells formed and kept (not giveups);
  3. The received giveups in red;
  4. The received backups in dark blue;
  5. The additional cells computed for the received giveups;
  6. The final cells produced after re-adjusting initial cells.


Generating random points to input as sites to the programs:
  The file random_pts_gen.cc contains a program that 
  produces a specified number of random points in a specified
  square region to a specified filename, printed in a format
  that can be read by any of the sequential or parallel
  implementations.

  The file is compiled into the executable 
    random_pts_gen
  which can be run with the command
    ./random_pts_gen [options]
         -n [val]             will set the number of points 
                              to print to val
         -s [val]             will set the range for generated 
                              points to [-val, val]
         -f [filename]        will print points to the file 
                              named filename
         -h                   will print this usage statement

The program will not run unless filename, number of points, 
and range of points is specified.

Sometimes the program doesn't work, not sure why. Just
try the command a couple times and it'll get going.


Testing the accuracy of computed cells:
  I've included a program with the file 
    test_complex_accuracy.cc
  that compiles into the executable
    test_complex
  and can be used to check the number of miscalculated
  cells by the complex parallel implementation.
  It can be run with the command
    ./test_complex [options]
      -n value      : specifies the number of points used
      -c filename   : specifies the name of the file 
                      with accurate cell data
      -t filename   : specifies the name of the file 
                      with cell data to be tested
      -h            : will print this usage statement

The number of points must be specified.
If the flag -c is not passed, then the default filename
    results_simple_parallel_[value].txt
will be used.
If the flag -t is not passed, then the default filename
    results_complex_parallel_[value].txt
will be used.


-----------------------------------------------------------------
REQUIRED FORMAT OF INPUT FILES
  Files containing input points should first include a positive integer
   indicating the number of input points contained in the file, followed by a
   space, tab, or newline. Input point values should be separated by spaces.
   For example:
      4
    	0 1
    	-3 4.8
    	8 0.2
    	4 -3
   is suitable for a file of four input points, whereas
      4
    	0, 1
    	-3, 4.8
    	8, 0.2
    	4, -3
   is not suitable.

   Files containing boundary line information should specify the boundary
   lines as halfplanes, of the form ax + by <= c. The intersection of the
   specified halfplanes should be a closed, convex region which contains all
   input points. Files containing boundary line information should firsly
   include a positive integer indicating the number of halfplanes to read.
   Then, for each halfplane ax + by <= c, the values a, b, and c should be
   included and separated by spaces.
   For example:
      4
    	-1 0 14
    	1 0 13
    	0 -1 6.5
    	0 1 9
   would be suitable to input the following boundary halfplanes:
      x >= -14
    	x <= 13
    	y >= -6.5
    	y <= 9


-----------------------------------------------------------------
FORMAT OF DIAGRAM INFORMATION OUTPUT
  The Voronoi Diagram information produces by any of the sequential
  or parallel implementations includes one line per input point, in which
  the first two values are the x and y value of the point, followed by
  the vertices of the cell printed in clockwise order, starting at noon.
  For example, the line
    0 0 0 1 1 0 0 -1 -1 0
  represents the cell for a point (0,0) with vertices (0,1), (1,0),
  (0,-1) and (-1,0).
    
  


  
  



  


  
