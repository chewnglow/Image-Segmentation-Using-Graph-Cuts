import numpy as np
import cv2
from matplotlib.image import imread
import networkx as nx
from numpy import linalg as LA
import math
from PIL import Image
from collections import defaultdict 

def convert(x,y,image) :
	a, b, c = image.shape
	return int(x*(b) + y) 

def convert2(n,image) :
	a, b, c = image.shape
	return n//b , n%b

def norm(x,cm) :
	r = np.matmul(x,x.T)
	# rr = np.matmul(r,x.T)
	# rrr = rr
	# print(rrr)
	return 100*math.exp(-r)

### opening image ####
image = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

### loading original and marked images ####
original = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
fg = cv2.imread('foreground.jpg', cv2.IMREAD_COLOR)
bg = cv2.imread('background.jpg', cv2.IMREAD_COLOR)
# marked = cv2.imread('marked.jpg', cv2.IMREAD_COLOR)

### dimensions of image #####
rows, cols, n = original.shape
new_image = np.zeros((rows,cols,n))
# print(rows, cols)

source = rows*cols
sink = source + 1

G = nx.DiGraph()
G.add_nodes_from(range(sink+1))

mean = np.zeros((n))

for r in range(rows) :
	for j in range(cols) :
		mean = mean + original[r][j]

mean = mean/(rows*cols)

print(mean)

covar_mat = np.zeros((n,n))

for r in range(rows) :
	for j in range(cols) :
		x = (original[r][j]-mean)
		x = np.matrix(x)
		x.reshape((n,1))
		covar_mat = covar_mat + np.matmul(x.T, x)

covar_mat = covar_mat / (rows*cols)
# covar_mat = np.linalg.inv(covar_mat) 

prob = np.zeros((255,1))
probl = np.zeros((255,1))

for r in range(rows) :
	for j in range(cols) :
		if fg[r][j][0] == 255 :
			xx = 0.299*original[r][j][0] + 0.587*original[r][j][1] + 0.114*original[r][j][2]
			xx = int(xx)
			# prob[xx] = prob[xx] + 1
		if bg[r][j][0] == 255 :
			xx = 0.299*original[r][j][0] + 0.587*original[r][j][1] + 0.114*original[r][j][2]
			xx = int(xx)
			# probl[xx] = probl[xx] + 1

# print(probl)
f = np.sum(prob)
b = np.sum(probl)

c = 0
print("Constructing graph .....")
for r in range(rows) :
	for j in range(cols) :
		c = c + 1
		if fg[r][j][0] == 255 :
			G.add_edge(source, convert(r,j,original))
		elif bg[r][j][0] == 255 :
			G.add_edge(convert(r,j,original), sink)
		else :
			xx = 0.299*original[r][j][0] + 0.587*original[r][j][1] + 0.114*original[r][j][2]
			xx = int(xx)
			# G.add_edge(source, convert(r,j,original), capacity = 10*prob[xx]/f)
			# G.add_edge(convert(r,j,original), sink, capacity = 10*probl[xx]/b)
		if r+1 < rows :
			G.add_edge(convert(r,j,original), convert(r+1,j,original), capacity = norm(original[r][j] - original[r+1][j],covar_mat))
		if r-1 > 0 :
			G.add_edge(convert(r,j,original), convert(r-1,j,original), capacity = norm(original[r][j] - original[r-1][j],covar_mat))
		if j+1 < cols :
			G.add_edge(convert(r,j,original), convert(r,j+1,original), capacity = norm(original[r][j] - original[r][j+1],covar_mat))
		if j-1 > 0 :
			G.add_edge(convert(r,j,original), convert(r,j-1,original), capacity = norm(original[r][j] - original[r][j-1],covar_mat))

print(c)
print(len(G.nodes))
print("Graph constructed.")

print("performing Max-Flow mincut ....")
cut_value, partition = nx.minimum_cut(G, source, sink)
print(cut_value)
print("Max-Flow mincut done.")
reachable, non_reachable = partition
print(len(reachable))
print(len(non_reachable))
count = 0
for r in reachable :
	if r > 0 and r < rows*cols :
		count = count + 1
		a , b = convert2(r,original)
		new_image[a][b][0] = 255
		new_image[a][b][1] = 255
		new_image[a][b][2] = 255

print(count)

for r in non_reachable :
	if r > 0 and r < rows*cols :
		count = count + 1
		a , b = convert2(r,original)
		new_image[a][b][0] = 0
		new_image[a][b][1] = 0
		new_image[a][b][2] = 0

print(count)

# cv2.imshow(new_image)

cv2.imwrite('final.jpg',new_image)
