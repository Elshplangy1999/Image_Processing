# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:27:24 2022

@author: Emad Elshplangy
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

img = cv.imread('img.png')#read Image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_RGB =  cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.title("Original IMG")
plt.show()

#Smoothing using Gaussian fillter for better output
gaussian = cv.GaussianBlur(img_gray,(3,3),0)

plt.imshow(gaussian,cmap='gray')
plt.title("Gaussian IMG")
plt.show()

# Sobel Edge Detector
sobelx = cv.Sobel(gaussian,cv.CV_8U,1,0,ksize=5)
sobely = cv.Sobel(gaussian,cv.CV_8U,0,1,ksize=5)
sobel = sobelx + sobely

plt.imshow(sobel,cmap='gray')
plt.title("Sobel IMG")
plt.show()

# Prewitt Gradient Operator
x_kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
y_kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_x = cv.filter2D(gaussian, -1, x_kernel)
prewitt_y = cv.filter2D(gaussian, -1, y_kernel)
prewitt = prewitt_x + prewitt_y

plt.imshow(prewitt,cmap='gray')
plt.title("Prewitt IMG")
plt.show()

# Laplacian of Gaussian
laplacian = cv.Laplacian(gaussian,cv.CV_64F)
plt.imshow(laplacian,cmap='binary')
plt.title("Laplacian IMG")
plt.show()
#Canny simple
canny = cv.Canny(gaussian,100,100)
plt.imshow(canny,cmap='gray')
plt.title("Canny IMG")
plt.show()

#canny with multi-step algorithm
#1.Noise reduction
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

gaussian_img = convolve(img_gray,gaussian_kernel(5))
plt.imshow(gaussian_img,cmap='gray')
plt.title("1.Noise reduction")
plt.show()
#2.Gradient calculation
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32) 
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    img=Ix+Iy
    return (img, theta)

sobel_img,theta=sobel_filters(gaussian_img)
plt.imshow(sobel_img,cmap='gray')
plt.title("2.Gradient calculation")
plt.show()
#3.Non-maximum suppression
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255        
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
            except IndexError:
                pass 
    return Z
non_max_img = non_max_suppression(sobel_img,theta)
plt.imshow(non_max_img,cmap='gray')
plt.title("3.Non-maximum suppression")
plt.show()
#4.Double threshold
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255) 
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res, weak, strong)
d_threshold_img,weak,strong = threshold(sobel_img,0.07,0.09)
plt.imshow(d_threshold_img,cmap='gray')
plt.title("4.Double threshold")
plt.show()
#5.Edge Tracking by Hysteresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError:
                    pass
    return img
hys_img = hysteresis(d_threshold_img,weak,strong)
plt.imshow(hys_img,cmap='gray')
plt.title("5.Edge Tracking by Hysteresis")
plt.show()