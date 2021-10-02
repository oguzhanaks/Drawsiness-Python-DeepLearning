import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder
import dlib
from PIL import Image
import matplotlib.image as mpimg
import os
import glob,os
from glob import glob
i =1465
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray 

pngs = glob(r'C:\Users\oguzh\Desktop\tezicin')

for j in pngs:
    x = cv2.imread(j)
    detector = dlib.get_frontal_face_detector()
    faces = detector(x) 
    ind = 0
 
    tl_col = faces[ind].tl_corner().x
    tl_row = faces[ind].tl_corner().y
    br_col = faces[ind].br_corner().x
    br_row = faces[ind].br_corner().y
    tl_h = faces[ind].height()
    tl_w = faces[ind].width()
    
    x=x[tl_row:tl_row + tl_h, tl_col:tl_col + tl_w, ::-1]

    
    gray = rgb2gray(x)
    
    name = r'C:\Users\oguzh\Desktop\tezicin'
    cv2.imwrite(   name, gray)
    i = i+1
    # cv2.imwrite(   j[:-3] + 'jpg', x)