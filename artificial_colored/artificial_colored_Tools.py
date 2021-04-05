# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 01:42:53 2020

@author: MaxGr
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import mat4py
import random
import os


def gray2rgb(image):
    image = np.clip(image, 0.5, 0.9)  
    r = image*random.randint(129, 167)
    g = image*random.randint(163, 181)
    b = image*random.randint(195, 230)
    
    return np.stack([r,g,b],2)

def imdilate(image): 
    
    kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], np.uint8)

    erosion = cv2.dilate(image,kernel,iterations = 1)

    return erosion



def false_magic(skeleton, circleinfo):
    
#   skeleton2 = bwperim.bwperim(skeleton)
    skeleton3 = imdilate(skeleton)
    # plt.imshow(skeleton3)
    
    skeleton = skeleton3.astype(np.uint8)
    
    
#operation

    height = skeleton.shape[0]
    width  = skeleton.shape[1]

    xgrid,ygrid = np.meshgrid(np.arange(1,width+1), np.arange(1,height+1))


    circle = np.sqrt((circleinfo['y'] - ygrid)*(circleinfo['y'] - ygrid) + (circleinfo['x'] - xgrid)*(circleinfo['x'] - xgrid)) <= circleinfo['radius']+2
#circle[circle <= circleinfo['radius']+2] = 1
#circle[circle >  circleinfo['radius']+2] = 0
    circle = circle.astype(np.uint8)


    circle2 = np.sqrt((circleinfo['y'] - ygrid)*(circleinfo['y'] - ygrid) + (circleinfo['x'] - xgrid)*(circleinfo['x'] - xgrid)) <= circleinfo['radius']-6
#circle2[circle2 <= circleinfo['radius']-6] = 1
#circle2[circle2 >  circleinfo['radius']-6] = 0
    circle2 = circle2.astype(np.uint8)



    tempImg = np.zeros((height, width))

    index_logical = (tempImg + circle - circle2) == 1
    index_logical = index_logical.astype(np.uint8)


    temp1 = skeleton
    temp1[temp1 == 1] = 255

# white [248, 248, 255]
    temp1[index_logical == 1] = 248 # first layer white


# purple [209 95 238]
    index1d = (temp1 == 255).astype(np.uint8)



# outer layer [35-70, 65-90, 60-75]
#outer frame
    temp1_d = np.random.randint(35,70,(height,width)).astype(np.uint8)
    temp2_d = np.random.randint(65,90,(height,width)).astype(np.uint8)
    temp3_d = np.random.randint(60,75,(height,width)).astype(np.uint8)

#dendrite layer
    temp1_d[index1d > 0] = random.randint(130, 170)
    temp2_d[index1d > 0] = random.randint(75, 104)
    temp3_d[index1d > 0] = random.randint(109, 150)

#outer dual-ring circle
    temp1_d[index_logical > 0] = random.randint(242, 252)
    temp2_d[index_logical > 0] = random.randint(247, 255)
    temp3_d[index_logical > 0] = random.randint(240, 255)

#inner dual-ring cirlce
    temp1_d[circle2 > 0] = random.randint(129, 167)
    temp2_d[circle2 > 0] = random.randint(163, 181)
    temp3_d[circle2 > 0] = random.randint(195, 230)




#show RGB image
    rgb_img = np.zeros((height,width,3)).astype("uint8")
    rgb_img[:,:,0] = temp3_d
    rgb_img[:,:,1] = temp2_d
    rgb_img[:,:,2] = temp1_d

    # plt.imshow(rgb_img)

    return rgb_img










