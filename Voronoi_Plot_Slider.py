# The List of libraries that need to be installed to run this code is written at the bottom of this file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.spatial import Delaunay

# State the list of input points and boundary halfplanes as an array
listOfInputPoints = []
listOfBoundaryPlanes = []
numberOfTimesteps = 0
FileName = 'plot_info.txt'


# Array of items per timestep
headers = []
targets = []
xPoints = []
raySegments = []
grayLines = []
greenLines = []
greenPoints = []
vertices = []
walls = []




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

		if len(words)>0:
			# If the first word is "BKGDP" then this is an input point
			if words[0]=="BKGDP":
				# If there are enough values to describe a point:
				if len(words)>3:
					point = []
					point.append(float(words[2]))
					point.append(float(words[3]))
					listOfInputPoints.append(point)


			# If the first word is "BKGDL" then this is an boundary line
			if words[0]=="BKGDL":
				# If there are enough values to describe a line:
				if len(words)>4:
					line = []
					line.append(float(words[2]))
					line.append(float(words[3]))
					line.append(float(words[4]))
					listOfBoundaryPlanes.append(line)

			# If the first word is "DIM" then this specifies the dimensions of the plot
			#if words[0]=="DIM":
				# If there are enough values to describe a line:
				#if len(words)>4:
					#dimensions.append(float(words[1]))
					#dimensions.append(float(words[2]))
					#dimensions.append(float(words[3]))
					#dimensions.append(float(words[4]))

			# If the first word is "Timestep"
			if words[0]=="Timestep":
				print("\n\nfound Timestep", numberOfTimesteps, "--------------------------")
				numberOfTimesteps=numberOfTimesteps+1
				lineAfter=lines[idx+1]
				wordsInLineAfter=lineAfter.split()
				if wordsInLineAfter[0]=="COUNTS":
					print("found "+lineAfter)
					# format is "COUNTS 1 1 1 1 0 0 0 0 0" which indicate the number of each item to draw in this timestep
					# If there are enough values to read counts:
					if len(wordsInLineAfter)>9:
						target=int(wordsInLineAfter[1])
						num_x=int(wordsInLineAfter[2])
						num_ray=int(wordsInLineAfter[3])
						header=int(wordsInLineAfter[4])
						num_lines=int(wordsInLineAfter[5])
						num_gr_pts=int(wordsInLineAfter[6])
						num_gr_lines=int(wordsInLineAfter[7])
						num_v=int(wordsInLineAfter[8])
						num_walls=int(wordsInLineAfter[9])
						# How many lines we need to read for this timestep:
						totalLinesToRead=(header*2)+num_x+target+num_ray+num_lines+num_gr_pts+num_gr_lines+num_v+num_walls
						print("Total lines to read is ", totalLinesToRead)
						# Below is the line that we are at now
						lineat = 2
						# Reading the header
						if header==1:
							# Move to the next line
							lineat=lineat+1
							message = lines[idx + lineat]
							lineat=lineat+1
							headers.append(message)
							print("header is " + message)
						else:
							headers.append(" ")
							print("no header")
						

						# Reading the target point
						if target==1:
							point = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							point.append(float(wordsInCurrentLine[1]))
							point.append(float(wordsInCurrentLine[2]))
							lineat=lineat+1
							targets.append(point)
							print("target is ", point[0],  point[1])
						else:
							targets.append([])
							print("no target")

						# Reading the x points
						xpts = []
						print("Reading x_points...")
						for i in range(0, num_x, 1):
							point = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							point.append(float(wordsInCurrentLine[1]))
							point.append(float(wordsInCurrentLine[2]))
							lineat=lineat+1
							xpts.append(point)
							print("    ", point[0], point[1])
						xPoints.append(xpts)

						# Reading the rays
						rays = []
						print("Reading rays...")
						for i in range(0, num_ray, 1):
							ray = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							ray.append(float(wordsInCurrentLine[1]))
							ray.append(float(wordsInCurrentLine[2]))
							ray.append(float(wordsInCurrentLine[3]))
							ray.append(float(wordsInCurrentLine[4]))
							lineat=lineat+1
							rays.append(ray)
							print("    ", ray[0], " ", ray[1], " ", ray[2], " ", ray [3])
						raySegments.append(rays)

						# Reading the lines
						gray_lines = []
						print("Reading lines...")
						for i in range(0, num_lines, 1):
							gray_line = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							gray_line.append(float(wordsInCurrentLine[1]))
							gray_line.append(float(wordsInCurrentLine[2]))
							gray_line.append(float(wordsInCurrentLine[3]))
							lineat=lineat+1
							gray_lines.append(gray_line)
							print("    ", gray_line[0], " ", gray_line[1], " ", gray_line[2])
						grayLines.append(gray_lines)

						# Reading the green lines
						green_lines = []
						print("Reading green lines...")
						for i in range(0, num_gr_lines, 1):
							green_line = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							green_line.append(float(wordsInCurrentLine[1]))
							green_line.append(float(wordsInCurrentLine[2]))
							green_line.append(float(wordsInCurrentLine[3]))
							lineat=lineat+1
							green_lines.append(green_line)
							print("    ", green_line[0], " ", green_line[1], " ", green_line[2])
						greenLines.append(green_lines)

						# Reading the green points
						green_points = []
						print("Reading green points...")
						for i in range(0, num_gr_pts, 1):
							green_point = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							green_point.append(float(wordsInCurrentLine[1]))
							green_point.append(float(wordsInCurrentLine[2]))
							lineat=lineat+1
							green_points.append(green_point)
							print("    ", green_point[0], " ", green_point[1])
						greenPoints.append(green_points)

						# Reading the vertices
						vtxs = []
						print("Reading vertices...")
						for i in range(0, num_v, 1):
							vtx = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							vtx.append(float(wordsInCurrentLine[1]))
							vtx.append(float(wordsInCurrentLine[2]))
							lineat=lineat+1
							vtxs.append(vtx)
							print("    ", vtx[0], " ", vtx[1])
						vertices.append(vtxs)

						# Reading the walls
						wls = []
						print("Reading walls...")
						for i in range(0, num_walls, 1):
							wal = []
							currentLine=lines[idx+lineat]
							wordsInCurrentLine=currentLine.split()
							wal.append(float(wordsInCurrentLine[1]))
							wal.append(float(wordsInCurrentLine[2]))
							wal.append(float(wordsInCurrentLine[3]))
							wal.append(float(wordsInCurrentLine[4]))
							lineat=lineat+1
							wls.append(wal)
							print("    ", wal[0], " ", wal[1], " ", wal[2], " ", wal[3])
						walls.append(wls)



print ('numberOfTimesteps:', numberOfTimesteps)

# Turn list of input points into a numpy array:
points = np.array(listOfInputPoints)
print("Input points:")
for p in points:
	print(p[0], p[1])
print()
for l in listOfBoundaryPlanes:
	print(l[0], l[1],l[2])
print()
# Turn list of boundary data into numpy array:

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
# Draw the background lines
x_min = -9 -2
x_max = 8 + 2
# Determine extreme y values
y_min = -1.5 - 2
y_max = 4 + 2
for bckground_lin in listOfBoundaryPlanes:
	a, b, c = bckground_lin
	x1 = x_min
	x2 = x_max
	if b != 0:	# a non-vertical line
		y1 = (c - a*x1) / b
		if (y1 > y_max):
			y_max = y1
		elif (y1 < y_min):
			y_min = y1
		y1 = (c - a*x2) / b
		if (y1 > y_max):
			y_max = y1
		elif (y1 < y_min):
			y_min = y1
	else:
		if c/a < x_min:
			x_min = c/a
		elif c/a > x_max:
			x_max = c/a

# plot lines
for bckground_lin in listOfBoundaryPlanes:
	a, b, c = bckground_lin
	x1 = x_min
	x2 = x_max
	if b != 0:	# a non-vertical line
		y1 = (c - a*x1) / b
		y2 = (c - a*x2) / b
	else:		# vertical line
		x1 = c/a
		x2 = c/a
		y1 = y_min
		y2 = y_max

	plt.plot([x1, x2], [y1, y2], color='black')


# add header
title = ax.set_title(f'initial state')

# add slider
axamp = fig.add_axes([0.3, .03, 0.50, 0.02])
samp = Slider(axamp, 'Timestep', 0, numberOfTimesteps-1, valinit=currentTimeStep, valstep=1)

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

	# update image
	plt.cla();


	# plot boundary lines
	for bckground_lin in listOfBoundaryPlanes:
		a, b, c = bckground_lin
		x1 = x_min
		x2 = x_max
		if b != 0:	# a non-vertical line
			y1 = (c - a*x1) / b
			y2 = (c - a*x2) / b
		else:		# vertical line
			x1 = c/a
			x2 = c/a
			y1 = y_min
			y2 = y_max

		plt.plot([x1, x2], [y1, y2], color='black')

	#triangles = np.asarray(listOfTriangles[currentTimeStep])
	#print ('Triangles in this Timestep (as an array):', triangles)
	t = np.asarray(targets[currentTimeStep])
	h = headers[currentTimeStep]
	x = np.asarray(xPoints[currentTimeStep])
	r = np.asarray(raySegments[currentTimeStep])
	lgray = np.asarray(grayLines[currentTimeStep])
	lgreen = np.asarray(greenLines[currentTimeStep])
	pgreen = np.asarray(greenPoints[currentTimeStep])
	v = np.asarray(vertices[currentTimeStep])
	w = np.asarray(walls[currentTimeStep])





	plt.plot(points[:,0], points[:,1], 'o')
	plt.plot(t[0], t[1], color='red', marker='x')
	for ra in r:
		x1, y1, x2, y2 = ra
		plt.plot([x1, x2], [y1, y2], color ='red')

	print("WALLS")
	for wall in w:
		x1, y1, x2, y2 = wall
		plt.plot([x1, x2], [y1, y2], linestyle = 'dotted', color ='grey')
		print(x1, " ", x2, " ", y1, " ", y2)



	for le in lgreen:
		a, b, c = le
		if b != 0:	# a non-vertical line
			x1 = x_min
			x2 = x_max
			y1 = (c - a*x1) / b
			y2 = (c - a*x2) / b
			# if y1 or y2 exceeds boundaries...
			if y1 > y_max:
				y1 = y_max
				x1 =(c-b*y1)/a
			elif y1<y_min:
				y1 = y_min
				x1 =(c-b*y1)/a
			if y2 > y_max:
				y2 = y_max
				x2 =(c-b*y2)/a
			elif y2<y_min:
				y2 =y_min
				x2 =(c-b*y2)/a
		else:		# vertical line
			x1 = c/a
			x2 = c/a
			y1 = y_min
			y2 = y_max

		# print("green: ", x1, y1, x2, y2)
		plt.plot([x1, x2], [y1, y2], linestyle = 'dotted', color ='green')

	for la in lgray:
		a, b, c = la
		# print("LA: ", a, b, c)
		if b != 0:	# a non-vertical line
			x1 = x_min
			x2 = x_max
			y1 = (c - a*x1) / b
			y2 = (c - a*x2) / b
			# if y1 or y2 exceeds boundaries...
			if y1 > y_max:
				y1 = y_max
				x1 =(c-b*y1)/a
			elif y1<y_min:
				y1 = y_min
				x1 =(c-b*y1)/a
			if y2 > y_max:
				y2 = y_max
				x2 =(c-b*y2)/a
			elif y2<y_min:
				y2 =y_min
				x2 =(c-b*y2)/a
			
		else:		# vertical line
			x1 = c/a
			x2 = c/a
			y1 = y_min
			y2 = y_max

		# print("grey: ", x1, y1, x2, y2)
		plt.plot([x1, x2], [y1, y2], linestyle = 'dashed', color ='grey')

	for gp in pgreen:
		plt.plot(gp[0], gp[1], color='green', marker='x')

	for vp in v:
		plt.plot(vp[0], vp[1], color='purple', marker='+')

	for xp in x:
		plt.plot(xp[0], xp[1], color='blue', marker='x')

	


	


	ax.set_title(h)
	plt.draw()
	## redraw canvas while idle
	fig.canvas.draw_idle()




def next(event):
    currentTimeStep = samp.val
    if currentTimeStep < numberOfTimesteps - 1:
        samp.set_val(currentTimeStep + 1)

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
