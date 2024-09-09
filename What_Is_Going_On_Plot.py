# The List of libraries that need to be installed to run this code is written at the bottom of this file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.spatial import Delaunay

# Adjust FileName as needed to read the desired file
FileName = 'whats_going_on_rank1.txt'

# State the list of input points and boundary halfplanes as an array
listOfInputPoints = []
numberOfInitialTimesteps = 0
numberOfMidTimesteps = 0
numberOfFinalTimesteps = 0
total_num_of_vertices = 0

# Array of vertices per timestep
initial_vertices = []
mid_vertices = []
final_vertices = []

giveups = []		# giveups recieved
backups = []		# backups received



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
		# If the first word is "BKGDP" then this is an input point
		if words[0]=="MYPOINT":
			if len(words) < 3: continue
			point = []
			point.append(float(words[1]))
			point.append(float(words[2]))
			listOfInputPoints.append(point)

		# If the first word is "Timestep" then this begins a cell's vertice in clockwise order
		if words[0]=="TimestepINITIAL":
			numberOfInitialTimesteps = numberOfInitialTimesteps + 1
			# read subsequent vertices
			array_of_vertices=[]
			keep_going=1
			while keep_going>0:
				# consider the next line
				nextLine=lines[idx+keep_going]
				wordsInNextLine = nextLine.split()
				if len(wordsInNextLine)==0:
					total_num_of_vertices = total_num_of_vertices + keep_going
					keep_going = 0
					initial_vertices.append(array_of_vertices)
					continue
				vertex_to_add=[]
				vertex_to_add.append(float(wordsInNextLine[1]))
				vertex_to_add.append(float(wordsInNextLine[2]))
				array_of_vertices.append(vertex_to_add)
				keep_going = keep_going + 1

		# If the first word is "Timestep" then this begins a cell's vertice in clockwise order
		if words[0]=="TimestepFINAL":
			numberOfFinalTimesteps = numberOfFinalTimesteps + 1
			# read subsequent vertices
			array_of_vertices=[]
			keep_going=1
			while keep_going>0:
				# consider the next line
				nextLine=lines[idx+keep_going]
				wordsInNextLine = nextLine.split()
				if len(wordsInNextLine)==0:
					total_num_of_vertices = total_num_of_vertices + keep_going
					keep_going = 0
					final_vertices.append(array_of_vertices)
					continue
				vertex_to_add=[]
				vertex_to_add.append(float(wordsInNextLine[1]))
				vertex_to_add.append(float(wordsInNextLine[2]))
				array_of_vertices.append(vertex_to_add)
				keep_going = keep_going + 1

		# If the first word is "Timestep" then this begins a cell's vertice in clockwise order
		if words[0]=="TimestepMID":
			numberOfMidTimesteps = numberOfMidTimesteps + 1
			# read subsequent vertices
			array_of_vertices=[]
			keep_going=1
			while keep_going>0:
				# consider the next line
				nextLine=lines[idx+keep_going]
				wordsInNextLine = nextLine.split()
				if len(wordsInNextLine)==0:
					total_num_of_vertices = total_num_of_vertices + keep_going
					keep_going = 0
					mid_vertices.append(array_of_vertices)
					continue
				vertex_to_add=[]
				vertex_to_add.append(float(wordsInNextLine[1]))
				vertex_to_add.append(float(wordsInNextLine[2]))
				array_of_vertices.append(vertex_to_add)
				keep_going = keep_going + 1

		if words[0]=="RG":		# ----------------------	GIVEUPS
			point = []
			point.append(float(words[1]))
			point.append(float(words[2]))
			giveups.append(point)


		if words[0]=="RB":		# ----------------------	BACKUPS
			point = []
			point.append(float(words[1]))
			point.append(float(words[2]))
			backups.append(point)



# Print information read
#print ('numberOfTimesteps:', numberOfTimesteps)
#print ('average number of vertices: ', total_num_of_vertices/numberOfTimesteps)

# Turn list of input points into a numpy array:
points = np.array(listOfInputPoints)
print("Input points:")
for p in points:
	print(p[0], p[1])

count = 0
for step in initial_vertices:
	print("Timestep ", count)
	count = count + 1

	for vertex in step:
		print("V ", vertex[0], vertex[1])

count = 0
for step in final_vertices:
	print("Timestep ", count)
	count = count + 1

	for vertex in step:
		print("V ", vertex[0], vertex[1])

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
samp = Slider(axamp, 'Timestep', 0, 5, valinit=currentTimeStep, valstep=1)

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
	if currentTimeStep == 0:
		print("my points")

# --------------------------------------------------------------------
	if currentTimeStep==1:
		# draw grid
		print("Initial diagram")
		for aray in initial_vertices:
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

# --------------------------------------------------------------------
	if currentTimeStep>=2:
		if len(giveups) == 0:
			print("\nno recieved giveups")
		else:
			print('\nrecieved giveups')
			points_to_plot = np.asarray(giveups)
			for p in points_to_plot:
				plt.plot(p[0], p[1], color='red', marker='o')

# --------------------------------------------------------------------
	if currentTimeStep>=3:
		if len(backups) == 0:
			print("\nno recieved backups")
		else:
			print('\nrecieved backups')
			points_to_plot = np.asarray(backups)
			for p in points_to_plot:
				plt.plot(p[0], p[1], color='blue', marker='o')

# --------------------------------------------------------------------
	if currentTimeStep==4:
		print("Middle diagram")
		# draw grid
		for aray in mid_vertices:
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

# --------------------------------------------------------------------
	if currentTimeStep==5:
		print("Final diagram")
		# draw grid
		for aray in final_vertices:
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






	plt.draw()
	## redraw canvas while idle
	fig.canvas.draw_idle()


# When next button is pushed
def next(event):
    currentTimeStep = samp.val
    if currentTimeStep < 5:
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