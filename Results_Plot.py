# The List of libraries that need to be installed to run this code is written at the bottom of this file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.spatial import Delaunay

# Change the FileName as needed to read the desired file
FileName = 'results_seq_1000.txt'

# State the list of input points and boundary halfplanes as an array
listOfInputPoints = []
total_num_of_points = 0
total_num_of_vertices = 0

# Array of vertices per timestep
vertices = []

# Open the file
with open(FileName) as file_object:
	file_object = open(FileName, 'r')
	print (file_object)

	# Turn the lines into an array
	lines = []
	for line in file_object:
		lines.append(line)

	# For each line in the text file
	for idx, line in enumerate(lines):
		# Split the line into an array of words
		words = line.split()
		if len(words) == 0: continue
		# Read the x and y value
		point = []
		point.append(float(words[0]))
		point.append(float(words[1]))
		listOfInputPoints.append(point)
		total_num_of_points = total_num_of_points + 1

		# Now read vertices
		array_of_vertices=[]
		for i in range(2, len(words), 2):
			total_num_of_vertices = total_num_of_vertices + 1
			vertex_to_add=[]
			vertex_to_add.append(float(words[i]))
			vertex_to_add.append(float(words[i+1]))
			array_of_vertices.append(vertex_to_add)
		vertices.append(array_of_vertices)



# Print information read
print ('Number of points:', total_num_of_points)
print ('Average number of vertices: ', total_num_of_vertices/total_num_of_points)

# Turn list of input points into a numpy array:
points = np.array(listOfInputPoints)
print("Input points:")
for p in points:
	print(p[0], p[1])

currentTimeStep = 0

# fig is the figure
# ax are the axes
fig, ax = plt.subplots()

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.15, bottom=0.25)
#fig.subplots_adjust(bottom=0.25)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')

# Draw the points
plt.plot(points[:,0], points[:,1], 'o')

# add slider
axamp = fig.add_axes([0.3, .03, 0.50, 0.02])
samp = Slider(axamp, 'Cells', 0, total_num_of_points+1, valinit=currentTimeStep, valstep=1)

# add buttons
axprev = plt.axes([0.05, 0.025, 0.1, 0.04])
axnext = plt.axes([0.85, 0.025, 0.1, 0.04])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Prev')


fig2 = plt.subplot()

print()

def update(val):
	# samp.val is the current value of the slider
	currentTimeStep = int(samp.val)

	# clear plot
	plt.cla();

	# plot input points
	plt.plot(points[:,0], points[:,1], 'o')

	if currentTimeStep==1:
		# draw everything
		for aray in vertices:
			if len(aray) !=0:
				vertices_to_plot = np.asarray(aray)
				# plot vertices
				plt.plot(vertices_to_plot[:,0], vertices_to_plot[:,1], color='purple', marker='x')

				# plot line segments elephant
				number_of_vertices = len(vertices_to_plot)
				for i in range(0, number_of_vertices, 1):
					x1 = vertices_to_plot[i, 0]
					y1 = vertices_to_plot[i, 1]
					x2 = vertices_to_plot[(i+1)%number_of_vertices, 0]
					y2 = vertices_to_plot[(i+1)%number_of_vertices, 1]
					plt.plot([x1, x2], [y1, y2], color='purple')

		
	elif currentTimeStep > 1:
		if len(vertices[currentTimeStep-2]) !=0:
			vertices_to_plot = np.asarray(vertices[currentTimeStep-2])
			# plot vertices
			plt.plot(vertices_to_plot[:,0], vertices_to_plot[:,1], color='purple', marker='x')

			# plot line segments elephant
			number_of_vertices = len(vertices_to_plot)
			for i in range(0, number_of_vertices, 1):
				x1 = vertices_to_plot[i, 0]
				y1 = vertices_to_plot[i, 1]
				x2 = vertices_to_plot[(i+1)%number_of_vertices, 0]
				y2 = vertices_to_plot[(i+1)%number_of_vertices, 1]
				plt.plot([x1, x2], [y1, y2], color='purple')




	plt.draw()
	## redraw canvas while idle
	fig.canvas.draw_idle()


# When next button is pushed
def next(event):
    currentTimeStep = samp.val
    if currentTimeStep < total_num_of_points + 1:
        samp.set_val(currentTimeStep + 1)

# When prev button is pushed
def prev(event):
    currentTimeStep = samp.val
    if currentTimeStep > 0:
        samp.set_val(currentTimeStep - 1)


# call update function on slider value change
samp.on_changed(update)


# call button function when buttons are pressed
bnext.on_clicked(next)
bprev.on_clicked(prev)

plt.show()



# List of libraries that need to be installed to run this code:
#pip install numpy
#pip install matplotlib
#pip install scipy