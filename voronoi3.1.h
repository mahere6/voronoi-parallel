/**
 * @file voronoi.h
 * @brief Struct definitions for Voronoi project
 * @author E. Maher
 * @version 3.1
 * @date 
 */


#ifndef VORONOI_H_OMJSZLDH
#define VORONOI_H_OMJSZLDH

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>


/**
 *  \brief A struct representing a point in 2-dimensional Euclidean space.
 * 
 * 	This struct will be inherited (is-a) by other structs, such as ordered_vector and vertex.
 */
struct point {
	double x;		//!<	x coordinate of point
	double y;		//!<	y coordinate of point

	//! Default constructor.
	point()							= default;		// default constructor
	~point()						= default;		// default destructor
	point(const point&)				= default;		// copy constructor
	point(point&&)					= default;		// move constructor
	point& operator=(const point&)	= default;		// copy assignment
	point& operator=(point&&)		= default;		// move assignment

	/**
	 * \brief An overloaded constructor.
	 */
	point(double xcoord, double ycoord)				// overloaded constructor
		: x{xcoord}
		, y{ycoord}{};

	

	/**
	 * \brief A member function which calculates the Euclidean distance to another point.
	 * 
	 * Distance is given by the formula sqrt( (x1-x2)^2 + (y1-y2)^2 ) for two points (x1, y1) and (x2, y2).
	 * 
	 * @param [in] other_point Another point object
	 * @return A double, the Euclidean distance from the given point to the specified point other_point
	 */
	double distance (point& other_point) const {
		return sqrt( (x - other_point.x)*(x - other_point.x) + (y - other_point.y)*(y - other_point.y));
	}

	/**
	 * \brief A member function which calculates the Euclidean distance to another point.
	 * 
	 * Distance is given by the formula sqrt( (x1-x2)^2 + (y1-y2)^2 ) for two points (x1, y1) and (x2, y2).
	 * 
	 * @param [in] x2 The x value of the other point
	 * @param [in] y2 The y value of the other point
	 * @return A double, the Euclidean distance from the given point to the specified point (x2, y2)
	 */
	double distance (double x2, double y2) const {
		return sqrt( (x - x2)*(x - x2) + (y - y2)*(y - y2));
	}
	
};


/**
 * \brief Overloading output stream operator << for a Point struct object.
 * 
 * Stream operator overload prints coordinates of a Point in the format (x, y).
 */
inline std::ostream & operator<<(std::ostream & os, const point & pt){
	os << "("<< pt.x<<","<<pt.y<<")";
	return os;
}


/**
 * \brief Overloading equality operator == for Point struct objects.
 * 
 * Equality operator returns true only if corresponding coordinates are equal.
 */
inline bool operator==(const point & pt1, const point & pt2){
	return ( (pt1.x == pt2.x) && (pt1.y == pt2.y) );
}


/**
 * \brief Overloading inequality operator != for Point struct objects.
 * 
 * Inequality operator returns false only if corresponding coordinates are equal.
 */
inline bool operator!=(const point & pt1, const point & pt2){
	return ( (pt1.x != pt2.x) || (pt1.y != pt2.y) );
}


/**
 *  \brief A struct to hold line information for a line in 2-dimensional Euclidean space.
 *
 *	The Line struct holds information to represent a line in the form ax+by=c, with the 
 *	coefficents a and b held respectively in the member variables [xcoeff] and [ycoeff] (doubles).
 *	The value c is held in a member variable [c] (a double).
 */
struct line {
	double xcoeff;		//!<	Coefficent of x in [ax+by=c]
	double ycoeff;		//!<	Coefficent of y in [ax+by=c]
	double c;			//!<	The value c in [ax+by=c]

	line()							= default;		// default constructor
	~line()							= default;		// default destructor
	line(const line&)				= default;		// copy constructor
	line(line&&)					= default;		// move constructor
	line& operator=(const line&)	= default;		// copy assignment
	line& operator=(line&&)			= default;		// move assignment

	/**
	 * \brief Overloaded constructor.
	 * 
	 * Overloaded constructor builds a Line object out of a Point object and a specified value.
	 */
	line(point pt, double val)						// overloaded constructor
		: xcoeff{pt.x}
		, ycoeff{pt.y}
		, c{val}{};
	/**
	 * \brief Overloaded constructor.
	 * 
	 * Overloaded constructor builds a Line object out of three specified values.
	 */
	line(double x, double y, double val)			// overloaded constructor
		: xcoeff{x}
		, ycoeff{y}
		, c{val}{};

	/**
	 * \brief Member function to print a Line object to standard output in the form of a halfplane.
	 * 
	 * Member function prints Line as "[xcoeff]x + [ycoeff]y <= [c]"
	 */
	void print_halfplane(){
		std::cout<< xcoeff <<"x + " << ycoeff <<"y <= "<< c;
	}
};


/**
 * \brief Overloading output stream operator << for a Line struct object.
 * 
 * Stream operator overload prints Line information in the format "ax + by = c".
 */
inline std::ostream & operator<<(std::ostream & os, const line & in){
	os << in.xcoeff <<"x + " << in.ycoeff <<"y = "<< in.c;
	return os;
}


/**
 *	\brief A struct inheriting (is-a) from Point struct to represent a vector in an order.
 *
 *	The Ordered_Vector struct holds information to represent a vector by storing the x and
 *	y coordinate in its (as a Point) member variables x and y. Furthermore, the Ordered_Vector
 *	struct holds its order in the member variable [index] (a double), which indicates the object's
 *	place in a collection of such objects rotating counterclockwise around a point.
 *	In addition, the Ordered_Vector struct contains two other member variables, a Line
 *	object [boundary] and a double [t]. These variables represent the intersection of a ray shot
 *	from a certain point p in the direction of the vector with the boundary of p's Voronoi cell:
 *	the line [boundary] is the line containing the segment of the boundary of the Voronoi cell,
 *	and the double [t] indicates the distance from p to said segment, such that p + t * ordered_vector
 *	hits the cell wall.
 */
struct ordered_vector : public point {
	line boundary;		//!<	Line containing Voronoi cell wall segment
	double t;			//!<	Coefficient to multiply by vector to reach endpoint from p
	float index;		//!<	Indicates location of unit vector in collection of unit vectors around p
	int nbr;			//!<	Pointer to the neighbour point

	ordered_vector()									= default;	// default constructor
	~ordered_vector()									= default;	// default destructor
	ordered_vector(const ordered_vector&)				= default;	// copy constructor
	ordered_vector(ordered_vector&&)					= default;	// move constructor
	ordered_vector& operator=(const ordered_vector&)	= default;	// copy assignment
	ordered_vector& operator=(ordered_vector&&)			= default;	// move assignment

	/**
	 *	\brief Overloaded constructor.
	 *
	 *	Overloaded constructor builds an Ordered_Vector object out of specified x and y
	 *	coordinates for the Point object and a specified value for the order of the vector.
	 *	The boundary Line object is initialized to 0x + 0y = 0, and the value for t is
	 *	initialized to the value -1 (indicating that these items are unknown initially).
	 */
	ordered_vector(double xcoord, double ycoord, float ind)		// overloaded constructor
		: point{xcoord, ycoord}
		, boundary{0,0,0}
		, t{-1}
		, index{ind}
		, nbr{-1}{};

	/**
	 *	\brief Overloaded constructor.
	 *
	 *	Overloaded constructor builds an Ordered_Vector object out of specified Point object
	 *	for the Point object and a specified value for the order of the vector. The
	 *	boundary Line object is initialized to 0x + 0y = 0, and the value for t is
	 *	initialized to the value -1 (indicating that these items are unknown initially).
	 */
	ordered_vector(point pt, float ind)							// overloaded constrictor
		: point{pt}
		, boundary{0,0,0}
		, t{-1}
		, index{ind}
		, nbr{-1}{};


	/**
	 * \brief Overloaded member function that resets the values held by the ordered_vector object.
	 * @param [in] a A double that replaces x
	 * @param [in] b A double that replaces y
	 * @param [in] ind A double that replaces index
	 */
	void reset (double const a, double const b, double ind) {
		x = a;
		y = b;
		t = -1;
		index = ind;
	}
	/**
	 * \brief Overloaded member function that resets the values held by the ordered_vector object.
	 * @param [in] vec An ordered_vector whose values replaces current values.
	 */
	void reset (ordered_vector vec) {
		x = vec.x;
		y = vec.y;
		t = -1;
		index = vec.index;
	}

	/**
	 *	\brief Member function that prints all vector information to standard output.
	 * 
	 *	Function prints in the format
	 *	"VECTOR INFO:
	 *		(x, y)
	 *		index = value
	 *		boundary = ax + by = c
	 *		t = value"
	 */
	void print_all () const {
		std::cout<<"\nVECTOR INFO:\n\t( " << x << ", " << y << " )\n\tindex = "<<index<<"\n\tboundary = "<<boundary<<"\n\tt = "<<t<<"\n";
	}
};


/**
 * \brief Overloading output stream operator << for an Ordered_Vector struct object.
 * 
 * Stream operator overload prints Ordered_Vector information in the format "(x, y)".
 */
inline std::ostream & operator<<(std::ostream & os, const ordered_vector & in){
	os << "("<< in.x <<", "<< in.y <<")";
	return os;
}


/**
 *	\brief A struct representing a subwedge of space originating from a point.
 *
 *	The Subwedge struct object simply contains two pointers to Ordered_Vector objects.
 */
struct subwedge {
	ordered_vector * vector1;		//!<	Pointer to ray first in clockwise order around point
	ordered_vector * vector2;		//!<	Pointer to ray second in clockwise order around point

	subwedge()								= default;		// default constructor
	~subwedge()								= default;		// default destructor
	subwedge(const subwedge&)				= default;		// copy constructor
	subwedge(subwedge&&)					= default;		// move constructor
	subwedge& operator=(const subwedge&)	= default;		// copy assignment
	subwedge& operator=(subwedge&&)			= default;		// move assignment

	/**
	 *	\brief Overloaded constructor.
	 */
	subwedge(ordered_vector * pt1, ordered_vector * pt2)	// overloaded constructor
		: vector1{pt1}
		, vector2{pt2}{};

};


/**
 * \brief Overloading output stream operator << for an Subwedge struct object.
 * 
 * Stream operator overload prints Subwedge information in the format "[ (x1, y1), (x2, y2) ]".
 */
inline std::ostream & operator<<(std::ostream & os, const subwedge & in){
	os << "[ " << *(in.vector1) << ", " << *(in.vector2) << " ]";
	return os;
}


/**
 * \brief Overloading multiplication operator * for Ordered_Vector struct objects.
 * 
 * @return a double that is the dot product of the two input Ordered_Vectors.
 */
inline double operator*(ordered_vector const v1, ordered_vector const v2){
	return (v1.x)*(v2.x) + (v1.y)*(v2.y);
}


/**
 * \brief Overloading multiplication operator * for Line struct object and Point struct object.
 * 
 * @return a double that is the dot product between the normal of the Line and the Point.
 */
inline double operator*(line const l, point const p){
	return (l.xcoeff)*(p.x) + (l.ycoeff)*(p.y);
}



/**
 * \brief Overloading addition operator + for Point struct objects.
 * 
 * @return A Point struct object that is the sum of the components of the added Points.
 */
inline point operator+(point const p1, point const p2){
	return point{p1.x + p2.x, p1.y + p2.y};
}


/**
 * \brief Overloading subtraction operator - for Point struct objects.
 * 
 * @return A Point struct object that is the difference of the components of the added Points.
 */
inline point operator-(point const p1, point const p2){
	return point{p1.x - p2.x, p1.y - p2.y};
}


/**
 * \brief Overloading division operator - for a Point struct object and a double.
 * 
 * Note: there is no error checking for 0 divisors.
 * 
 * @return A Point struct object that is the components of the Point after division by the specified Point.
 */
inline point operator/(point const p1, double const d){
	return point{p1.x / d, p1.y / d};
}


/**
 * \brief Overloading multiplication operator * for a double and a Point struct object.
 * 
 * @return A Point struct object that is the components of the Point after multiplication with the specified double.
 */
inline point operator*(double const d, point const pt){
	return point{d*(pt.x), d*(pt.y)};
}


/**
 *	\brief A struct inheriting (is-a) from Point struct to represent a vertex in an order.
 *
 *	The Vertex struct holds information to represent a vertex of a Voronoi cell by a Point,
 *	while also including information about the order of the Vertex in a set of vertices in 
 *	a counter-clockwise fashion which make up a Voronoi cell. The double [index] holds the
 *	order of the Vertex, and the line [hfplane] reveals a line upon which the vertex sits in 
 *	a segment of the boundary of the Voronoi cell.
 */
struct vertex : public point {
	line hfplane;		//!<	Line holding (one) segment of Voronoi cell containing vertex
	float index;		//!<	Order of vertex out of vertices in boundary cell in clockwise fashion

	vertex()							= default;		// default constructor
	~vertex()							= default;		// default destructor
	vertex(const vertex&)				= default;		// copy constructor
	vertex(vertex&&)					= default;		// move constructor
	vertex& operator=(const vertex&)	= default;		// copy assignment
	vertex& operator=(vertex&&)			= default;		// move assignment

	/**
	 *	\brief Overloaded constructor.
	 */
	vertex(double xcoord, double ycoord, float ind, line ln)	// overloaded constructor
		: point{xcoord, ycoord}
		, hfplane{ln}
		, index{ind}{};
	/**
	 *	\brief Overloaded constructor.
	 */
	vertex(point pt, float ind, line ln)						// overloaded constructor
		: point{pt}
		, hfplane{ln}
		, index{ind}{};
		
};


/**
 *	\brief A struct which holds the Voronoi cell information for a particular point.
 *
 *	A Cell_Info struct holds the point for which the Voronoi cell exists in the 
 *	member variable p (a point), and holds the vertices of the cell in a std::vector
 *	of Vertex struct objects. 
 */
struct cell_info {
	std::vector<vertex> vertices;	//!<	std::vector of vertices of the cell
	point p;						//!<	The point which the Voronoi cell exists for

	cell_info()								= default;		// default constructor
	~cell_info()							= default;		// default destructor
	cell_info(const cell_info&)				= default;		// copy constructor
	cell_info(cell_info&&)					= default;		// move constructor
	cell_info& operator=(const cell_info&)	= default;		// copy assignment
	cell_info& operator=(cell_info&&)		= default;		// move assignment

	/**
	 *	\brief Overloaded constructor.
	 */
	cell_info(double xcoord, double ycoord)					// overloaded constructor
		: p{xcoord, ycoord}{
			vertices.reserve(7);
		};
	/**
	 *	\brief Overloaded constructor.
	 */
	cell_info(point pt)										// overloaded constructor
		: p{pt}{
			vertices.reserve(7);
		};

	
	/**
	 * \brief Overloaded member function that adds a vertex to the std::vector of vertices.
	 * @param [in] xcoord A double indicating the x coordinate of the vertex
	 * @param [in] ycoord A double indicating the y coordinate of the vertex
	 * @param [in] ind	A double indicating the position of the vertex in counterclockwise order around p
	 * @param [in] ln	The line on which the vertex falls in the Voronoi cell
	 */
	void add_vertex(double xcoord, double ycoord, double ind, line ln){
		vertices.emplace_back(xcoord, ycoord, ind, ln);
	}
	/**
	 * \brief Overloaded member function that adds a vertex to the std::vector of vertices.
	 * @param [in] pt A Point object representing the point on which the vector falls
	 * @param [in] ind	A double indicating the position of the vertex in counterclockwise order around p
	 * @param [in] ln	The line on which the vertex falls in the Voronoi cell
	 */
	void add_vertex(point pt, double ind, line ln){
		vertices.emplace_back(pt, ind, ln);
	}

	/**
	 * \brief A member function that sorts the vertices in counter-clockwise order around p.
	 */
	void sort_vertices(){
		std::sort(vertices.begin(), vertices.end(), [](const vertex& v1, const vertex& v2){return v1.index < v2.index;});
	}
};


/**
 * \brief Overloading output stream operator << for an Cell_Info struct object.
 * 
 * Stream operator overload prints cell_info information in a table format, with all points
 * listed and their respective indices and wall lines specified.
 */
inline std::ostream & operator<<(std::ostream & os, const cell_info & cell){
	os << "POINT: "<< cell.p <<"\n\t";
	os << "index\t      x\t\ty       wall: ax + by\t  =   c\n"<<std::setprecision(5);
	for (auto it = cell.vertices.cbegin(); it != cell.vertices.cend(); ++it){
		os<<"\t("<< (*it).index << ")\t  " << std::setw(7) << (*it).x << "    " << std::setw(7) << (*it).y << "\t" << std::setw(7) << (*it).hfplane.xcoeff << "    " << (*it).hfplane.ycoeff << "\t" << std::setw(7) << (*it).hfplane.c << "\n";
	}
	return os;
}



// -------------------------------------------------------------------------------------------
// ------------------ STRUCTS FOR GENERATING THE TIMESTEP PLOT -------------------------------
// -------------------------------------------------------------------------------------------


/**
 *	\brief A struct which represents a line segment in 2-dimensional Euclidean space.
 *
 *	A Segment struct holds the endpoints of a line segment as four doubles x1, y1, x2, y2, where
 *	the endpoints are (x1, y1) and (x2, y2).
 */
struct segment {
	double x1;		//!<	x coordinate of first point
	double y1;		//!<	y coordinate of first point
	double x2;		//!<	x coordinate of second point
	double y2;		//!<	y coordinate of second point

	segment()							= default;		// default constructor
	~segment()							= default;		// default destructor
	segment(const segment&)				= default;		// copy constructor
	segment(segment&&)					= default;		// move constructor
	segment& operator=(const segment&)	= default;		// copy assignment
	segment& operator=(segment&&)		= default;		// move assignment

	/**
	 * \brief An overloaded constructor.
	 */
	segment(point pt1, point pt2)						// overloaded constructor
		: x1{pt1.x}
		, y1{pt1.y}
		, x2{pt2.x}
		, y2{pt2.y}{};
	/**
	 * \brief An overloaded constructor.
	 */
	segment(double x1, double y1, double x2, double y2)	// overloaded constructor
		: x1{x1}
		, y1{y1}
		, x2{x2}
		, y2{y2}{};
};


/**
 *	\brief A struct which holds and prints timestep data to generate a slider timestep plot of the algorithm's process.
 *
 *	The struct prints a file that can be read by the matplotlib python script Voronoi_Plot_Slider.py.
 *	The struct holds many values to be plotted and plot information, and can be incremented to
 *	a new timestep by using one of the included member functions, which will print currently held 
 *	values to file, then adjust the currently held values.
 */
struct timestep_plot_data {
	point p;							//!<	The point whose cell the algorithm is currently working on
	std::vector<point> x_points;		//!<	A collection of points to be displayed with an 'x'
	std::vector<segment> rays;			//!<	A collection of rays to be dislpayed in red
	std::string header;					//!<	A string header to display above the plot
	std::vector<line> lines;			//!<	A collection of lines (usually bisector lines) to dislpay on the plot
	std::vector<point> green_points;	//!<	A collection of points to display in green
	std::vector<line> green_lines;		//!<	A collection of points to display in green
	std::vector<point> vertices;		//!<	A collection of vertices found
	std::vector<segment> walls;			//!<	A collection of cell lines determined
	int timestep_num;					//!<	The current timestep number that we are on
	std::string filename;				//!<	The filename to print to


	timestep_plot_data()										= delete;  // default constructor
	timestep_plot_data(size_t inital_reserve, std::string fname, std::vector<point>& input_points, std::vector<line>& boundary_lines);	// overloaded constructor
	~timestep_plot_data()										= default;  // default destructor
	timestep_plot_data(const timestep_plot_data&)				= default;  // copy constructor
	timestep_plot_data(timestep_plot_data&&)					= default;  // move constructor
	timestep_plot_data& operator=(const timestep_plot_data&)	= default;  // copy assignment
	timestep_plot_data& operator=(timestep_plot_data&&)			= default;  // move assignment

	/**
	 * \brief A member function that returns the current timestep.
	 * @return An int, the current timestep.
	 */
	int gettime() const {
		return timestep_num;
	}

	/**
	 * \brief A member function that clears the items to be diplayed in green
	 */
	void clear_green() {
		green_lines.clear();
		green_points.clear();
	}

	/**
	 * \brief A member function that adds a point to be diplayed in green.
	 * @param [in] pt A point object
	 */
	void add_green_point(point const pt){
		green_points.emplace_back(pt);
	}

	/**
	 * \brief A member function that adds a line to be diplayed in green.
	 * @param [in] ln A line object
	 */
	void add_green_line(line const ln){
		green_lines.emplace_back(ln);
	}

	/**
	 * \brief A member function that clears the rays to be displayed.
	 */
	void clear_rays() {
		rays.clear();
	}

	/**
	 * \brief A member function that adds a ray to be diplayed.
	 * @param [in] pt A point object, one endpoint of the ray
	 * @param [in] ry A point object, other endpoint of the ray
	 */
	void add_ray(point const pt, point const ry){
		rays.emplace_back(pt.x, pt.y, ry.x, ry.y);
	}

	/**
	 * \brief A member function that first removes all rays to be displayed then adds a ray to be diplayed.
	 * @param [in] pt A point object, one endpoint of the ray
	 * @param [in] ry A point object, other endpoint of the ray
	 */
	void set_ray(point const pt, point const ry){
		rays.clear();
		rays.emplace_back(pt.x, pt.y, ry.x, ry.y);
	}

	/**
	 * \brief A member function that adds a cell wall to be diplayed.
	 * @param [in] p1 A point object, one endpoint of the wall
	 * @param [in] p2 A point object, other endpoint of the wall
	 */
	void add_wall(point p1, point p2){
		walls.emplace_back(p1, p2);
	}


	/**
	 * \brief A member function that moves to the next timestep by first printing to file the data to be 
	 * displayed in this timestep, then adjusting the current data held by the struct.
	 */
	void increment();

	/**
	 * \brief A member function that adjusts the current point that the algorithm is focussed on, 
	 * then increments to the next timestep by calling the member function increment().
	 * @param [in] pt A point object
	 * @see increment()
	 */
	void select_point(point const pt){
		p = pt;
		header.assign("select point");
		increment();
		return;
	}

	/**
	 * \brief A member function that changes the header to de displayed on the plot.
	 * @param [in] str A std::string object
	 */
	void text(std::string str){
		header.assign(str);
	}

	/**
	 * \brief A member function that adds a point to be displayed with an x.
	 * @param [in] pt A point object
	 */
	void add_x(point pt){
		x_points.emplace_back(pt);
	}

	/**
	 * \brief A member function that adds a line to be displayed.
	 * @param [in] ln A line object
	 */
	void add_line(line ln){
		lines.emplace_back(ln);
	}

	/**
	 * \brief A member function that adds a vertex to be displayed on the plot.
	 * @param [in] pt A point object
	 */
	void add_vertex(point pt){
		vertices.emplace_back(pt);
	}

	/**
	 * \brief A member function similar to increment() that moves to the next timestep 
	 * by first printing to file the data to be displayed in this timestep, then 
	 * adjusting the current data held by the struct. Contrary to increment(), this
	 * function does not change the members x_points and header.
	 */
	void keep_x_header();

	/**
	 * \brief A member function similar to increment() that moves to the next timestep 
	 * by first printing to file the data to be displayed in this timestep, then 
	 * incrementing the current value held by the member variable timestep_num. 
	 * Contrary to increment(), this function does not change any other member variables.
	 */
	void keep_all();

};


inline void timestep_plot_data::increment(){
	// first write previous timestep data to file
	std::ofstream plot_file;
	plot_file.open(filename, std::ios::app);	// open file to write data to
	if (!plot_file){	// error checking
		std::cerr<<"Failed to open file for writing timestep data";
		return;
	}

	// WRITE STUFF TO FILE
	plot_file<<"Timestep "<<timestep_num<<"\n";	// indicate timestep
	int s = 1;	// will reveal whether there is a header or not
	if (header.empty()) {
		s = 0;
	}
	// indicate how many of each item we would like to plot for this timestep
	plot_file<<"COUNTS 1 "<<x_points.size()<<" "<<rays.size()<<" "<<s<<" "<<lines.size()<<" "<<green_points.size()<< " "<< green_lines.size()<< " "<<vertices.size()<< " "<< walls.size()<<"\n";
	if (s) plot_file<<"HEADER\n"<<header<<"\n";	// write header
	plot_file<<"TARGET " << p.x << " " << p.y << "\n";	// write target point
	for (auto it = x_points.cbegin(); it != x_points.cend(); ++it){	// print points marked with an x
		plot_file<<"XPOINT " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = rays.cbegin(); it != rays.cend(); ++it){	// print rays to be drawn
		plot_file<<"RAY " << (*it).x1 << " " << (*it).y1 << " "<< (*it).x2 << " " << (*it).y2 << "\n";
	}
	for (auto it = lines.cbegin(); it != lines.cend(); ++it){	// print lines to be drawn
		plot_file<<"LINE " << (*it).xcoeff << " " << (*it).ycoeff <<" "<< (*it).c << "\n";
	}
	for (auto it = green_lines.cbegin(); it != green_lines.cend(); ++it){	// print green lines to be drawn
		plot_file<<"GLINE " << (*it).xcoeff << " " << (*it).ycoeff << " "<< (*it).c << "\n";
	}
	for (auto it = green_points.cbegin(); it != green_points.cend(); ++it){	// print green points to be drawn
		plot_file<<"GPOINT " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = vertices.cbegin(); it != vertices.cend(); ++it){	// print vertices to be drawn
		plot_file<<"VERTEX " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = walls.cbegin(); it != walls.cend(); ++it){	// print cell walls to be drawn
		plot_file<<"WALL " << (*it).x1 << " " << (*it).y1 << " " << (*it).x2 << " " << (*it).y2 << "\n";
	}

	plot_file<<"\n";
	plot_file.close();




	// now we move to the next timestep by adjusting elements in struct
	timestep_num++;		// this is the timestep that we are move to
	header.clear();		// clear the header we will use
	x_points.clear();	// clear the points marked with x
	lines.clear();		// clear the grey lines drawn


	return;
}

inline timestep_plot_data::timestep_plot_data(size_t initial_reserve, std::string fname, std::vector<point>& input_points, std::vector<line>& boundary_lines)
	: p{0,0}, timestep_num{1}, filename{fname}{	// constructor
	// we start on timestep 1 because timestep 0 is just the initial data (input points and boundaries)
	// timestep 1 should just highlight the initial point so we are able now to move to timestep 2
	// allocate a reserve of memory
	x_points.reserve(initial_reserve);
	rays.reserve(initial_reserve);
	header.reserve(initial_reserve);
	lines.reserve(initial_reserve);
	green_points.reserve(initial_reserve);
	green_lines.reserve(initial_reserve);
	walls.reserve(initial_reserve);

	// first we write input points and boundary lines to file
	std::ofstream plot_file(filename);	// open file for writing
	if(!plot_file){		// error check
		std::cerr<<"Failed to open file for writing timestep plot data";
		return;
	}
	// write input points to file
	for (size_t i = 0; i < input_points.size(); i++){
		plot_file<<"BKGDP "<< i <<" "<<input_points[i].x<<" "<<input_points[i].y<<"\n";		// format: BKGDP 6 0.487 8.001
	}
	// write boundary lines to file for timestep plot vvv
	for (size_t i = 0; i < boundary_lines.size(); i++){
		plot_file<<"BKGDL "<<i<< " "<< boundary_lines[i].xcoeff <<" "<< boundary_lines[i].ycoeff<<" "<< boundary_lines[i].c << "\n";	//format: BKGDL 3 8.021 -55.3 20.9
	}

	// now write timestep 0 to file and move to timestep 1
	plot_file<<"Timestep 0\n";
	plot_file<<"COUNTS 0 0 0 1 0 0 0 0 0 \n";
	plot_file<<"HEADER\ninitial state\n";


	plot_file.close();	// close file

};

inline void timestep_plot_data::keep_x_header() {
	// first write previous timestep data to file
	std::ofstream plot_file;
	plot_file.open(filename, std::ios::app);	// open file to write data to
	if (!plot_file){	// error checking
		std::cerr<<"Failed to open file for writnig timestep data";
		return;
	}

	// WRITE STUFF TO FILE
	plot_file<<"Timestep "<<timestep_num<<"\n";	// indicate timestep
	int s = 1;	// will reveal whether there is a header or not
	if (header.empty()) {
		s = 0;
	}
	// indicate how many of each item we would like to plot for this timestep
	plot_file<<"COUNTS 1 "<<x_points.size()<<" "<<rays.size()<<" "<<s<<" "<<lines.size()<<" "<<green_points.size()<< " "<< green_lines.size()<< " "<<vertices.size()<< " "<< walls.size()<<"\n";
	if (s) plot_file<<"HEADER\n"<<header<<"\n";	// write header
	plot_file<<"TARGET " << p.x << " " << p.y << "\n";	// write target point
	for (auto it = x_points.cbegin(); it != x_points.cend(); ++it){	// print points marked with an x
		plot_file<<"XPOINT " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = rays.cbegin(); it != rays.cend(); ++it){	// print rays to be drawn
		plot_file<<"RAY " << (*it).x1 << " " << (*it).y1 << " "<< (*it).x2 << " " << (*it).y2 << "\n";
	}
	for (auto it = lines.cbegin(); it != lines.cend(); ++it){	// print lines to be drawn
		plot_file<<"LINE " << (*it).xcoeff << " " << (*it).ycoeff <<" "<< (*it).c << "\n";
	}
	for (auto it = green_lines.cbegin(); it != green_lines.cend(); ++it){	// print green lines to be drawn
		plot_file<<"GLINE " << (*it).xcoeff << " " << (*it).ycoeff << " "<< (*it).c << "\n";
	}
	for (auto it = green_points.cbegin(); it != green_points.cend(); ++it){	// print green points to be drawn
		plot_file<<"GPOINT " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = vertices.cbegin(); it != vertices.cend(); ++it){	// print vertices to be drawn
		plot_file<<"VERTEX " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = walls.cbegin(); it != walls.cend(); ++it){	// print cell walls to be drawn
		plot_file<<"WALL " << (*it).x1 << " " << (*it).y1 << " " << (*it).x2 << " " << (*it).y2 << "\n";
	}

	plot_file<<"\n";
	plot_file.close();




	// now we move to the next timestep by adjusting elements in struct
	timestep_num++;		// this is the timestep that we are move to
	lines.clear();		// clear the grey lines drawn


	return;
}

inline void timestep_plot_data::keep_all() {
	// first write previous timestep data to file
	std::ofstream plot_file;
	plot_file.open(filename, std::ios::app);	// open file to write data to
	if (!plot_file){	// error checking
		std::cerr<<"Failed to open file for writnig timestep data";
		return;
	}

	// WRITE STUFF TO FILE
	plot_file<<"Timestep "<<timestep_num<<"\n";	// indicate timestep
	int s = 1;	// will reveal whether there is a header or not
	if (header.empty()) {
		s = 0;
	}
	// indicate how many of each item we would like to plot for this timestep
	plot_file<<"COUNTS 1 "<<x_points.size()<<" "<<rays.size()<<" "<<s<<" "<<lines.size()<<" "<<green_points.size()<< " "<< green_lines.size()<< " "<<vertices.size()<< " "<< walls.size()<<"\n";
	if (s) plot_file<<"HEADER\n"<<header<<"\n";	// write header
	plot_file<<"TARGET " << p.x << " " << p.y << "\n";	// write target point
	for (auto it = x_points.cbegin(); it != x_points.cend(); ++it){	// print points marked with an x
		plot_file<<"XPOINT " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = rays.cbegin(); it != rays.cend(); ++it){	// print rays to be drawn
		plot_file<<"RAY " << (*it).x1 << " " << (*it).y1 << " "<< (*it).x2 << " " << (*it).y2 << "\n";
	}
	for (auto it = lines.cbegin(); it != lines.cend(); ++it){	// print lines to be drawn
		plot_file<<"LINE " << (*it).xcoeff << " " << (*it).ycoeff <<" "<< (*it).c << "\n";
	}
	for (auto it = green_lines.cbegin(); it != green_lines.cend(); ++it){	// print green lines to be drawn
		plot_file<<"GLINE " << (*it).xcoeff << " " << (*it).ycoeff << " "<< (*it).c << "\n";
	}
	for (auto it = green_points.cbegin(); it != green_points.cend(); ++it){	// print green points to be drawn
		plot_file<<"GPOINT " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = vertices.cbegin(); it != vertices.cend(); ++it){	// print vertices to be drawn
		plot_file<<"VERTEX " << (*it).x << " " << (*it).y << "\n";
	}
	for (auto it = walls.cbegin(); it != walls.cend(); ++it){	// print cell walls to be drawn
		plot_file<<"WALL " << (*it).x1 << " " << (*it).y1 << " " << (*it).x2 << " " << (*it).y2 << "\n";
	}

	plot_file<<"\n";
	plot_file.close();




	// now we move to the next timestep by adjusting elements in struct
	timestep_num++;		// this is the timestep that we are move to


	return;
}





#endif /* end of include guard: VORONOI_H_OMJSZLDH */