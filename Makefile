#
# Makefile for Parallel Voronoi
#
# E. Maher
# V 3.0
# 9-9-2024
#


all: sequential_voronoi sequential_voronoi_output voronoi_functions.o v_parallel_simple v_parallel_complex v_parallel_complex_output random_pts_gen test_complex

sequential_voronoi: voronoi.h sequential_voronoi.cc
	g++ -Wall -Wextra -std=c++20 -Wsign-conversion -o sequential_voronoi sequential_voronoi.cc -lm -lstdc++

sequential_voronoi_output: voronoi.h sequential_voronoi_output.cc
	g++ -Wall -Wextra -std=c++20 -Wsign-conversion -o sequential_voronoi_output sequential_voronoi_output.cc -lm -lstdc++

v_parallel_simple: voronoi_functions.o voronoi_functions.cc voronoi_functions.h simple_parallel_voronoi.cc voronoi.h
	mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o v_parallel_simple simple_parallel_voronoi.cc voronoi_functions.o -lm -lstdc++

voronoi_functions.o: voronoi_functions.cc voronoi_functions.h voronoi.h
	mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o voronoi_functions.o -c voronoi_functions.cc -lm

v_parallel_complex: complex_parallel_voronoi.cc voronoi.h
	mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o v_parallel_complex complex_parallel_voronoi.cc -lm -lstdc++

v_parallel_complex_output: complex_parallel_voronoi_output.cc voronoi.h
	mpiCC -Wall -Wextra -std=c++20 -Wsign-conversion -o v_parallel_complex_output complex_parallel_voronoi_output.cc -lm -lstdc++

random_pts_gen: random_pts_gen.cc
	g++ -Wall -Wextra -std=c++20 -o random_pts_gen random_pts_gen.cc

test_complex: test_complex_accuracy.cc
	g++ -Wall -Wextra -std=c++20 -Wsign-conversion -o test_complex test_complex_accuracy.cc -lm -lstdc++

.PHONY: all clean fresh fresh_timing

clean:
	rm -f sequential_voronoi sequential_voronoi_output voronoi_functions.o v_parallel_simple v_parallel_complex v_parallel_complex_output random_pts_gen test_complex

fresh:
	rm -f "whats_going_on_rank"*
	rm -f "results_seq_"*
	rm -f "results_complex_parallel_"*
	rm -f "results_simple_parallel_"*
	rm -f plot_info.txt
	
fresh_timing:
	rm -f "timing_results_"*