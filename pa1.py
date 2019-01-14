# CAP5415 2014
# Assignment 1

import numpy as np
# import scipy.ndimage
import matplotlib.pyplot as plt

## Problem 1
# Write a function that convolves an image with a given convolution filter
# function [output_Image]= myImageFilter( Input_image, filter)

def convolve(img, kernel):
	""" Perform a 2D convolution with kernel on img.
	Returns the result of the convolution.
	stride = 1
	padding = None
	"""
	x, y = img.shape
	fx, fy = kernel.shape
	px, py = (x - fx + 1, y - fy + 1)
	print (px, py)

	result = np.zeros((px, py))

	for i in range(px):
		for j in range(py):
			img_piece = img[i: (i + fx), j: (j + fy)]
			prod = img_piece * kernel
			value = np.sum(prod)
			# print value.shape
			result[i, j] = value

	return result


sobel_x = np.zeros((3,3), dtype=np.int)
sobel_x[0, :] = [-1, -2, -1]
sobel_x[1, :] = [0, 0, 0]
sobel_x[2, :] = [1, 2, 1]
print sobel_x

img = plt.imread('House1.jpg')
plt.imshow(img)
plt.show()


result = convolve(img, sobel_x)

plt.imshow(result)
plt.show()


