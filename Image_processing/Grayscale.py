import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

  

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
image = mpimg.imread('kang1000.jpg')

gray = rgb2gray(image)
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
  
cv2.waitKey(0)
cv2.destroyAllWindows()