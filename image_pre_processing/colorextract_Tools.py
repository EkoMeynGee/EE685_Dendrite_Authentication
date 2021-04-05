# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:35:11 2020

@author: MaxGr
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import mat4py
import random









def imsegkmeans(image,K):
    
    Z = image.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #K = 10
    ret,label,center=cv2.kmeans(Z,K,None,criteria,3,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    
    #plt.imshow(res2[:,:,0])

    return res2
    
    
    
    
    
def erosion(image, kernelsize, iterations):
    
    kernel = np.ones((kernelsize,kernelsize),np.uint8)  
    
    #kernel = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]], np.uint8)

    erosion = cv2.erode(image,kernel,iterations = iterations)
    
    return erosion





def dilate(image, kernelsize, iterations): 
    
    kernel = np.ones((kernelsize,kernelsize),np.uint8)  

    #kernel = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]], np.uint8)

    dilation  = cv2.dilate(image,kernel,iterations = iterations)

    return dilation 





def imageProcess(image, numerosion, numdilate):
    
    for i in range(numerosion+1):
        image = erosion(image)
        
        
    for i in range(numdilate+1):
        image = dilate(image)
        
    return image





def openimage(image, kernelsize, iterations):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelsize,kernelsize))

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = iterations)

    return image








def houghCircle(image_RGB):
    [H , W, Z] = image_RGB.shape
    minFrame = min(H,W)
    

    if minFrame < 500:
        dst = cv2.pyrMeanShiftFiltering(image_RGB, 20, 100) # plt.imshow(dst)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) # plt.imshow(gray)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=100, param2=15, minRadius=0, maxRadius=70)
        # cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=100, param2=15, minRadius=10, maxRadius=70)
        
    elif minFrame > 4000:
        dst = cv2.pyrMeanShiftFiltering(image_RGB, 30, 100) # plt.imshow(dst)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) # plt.imshow(gray)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 5000, param1=200, param2=10, minRadius=50, maxRadius=500)
        
    else:
        dst = cv2.pyrMeanShiftFiltering(image_RGB, 10, 100) # plt.imshow(dst)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) # plt.imshow(gray)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=20, minRadius=0, maxRadius=100)
        
    circles = np.uint16(np.around(circles))
    
    # for i in (circles[0,:]):
    #     # draw the outer circle
    #     cv2.circle(image_RGB,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv2.circle(image_RGB,(i[0],i[1]),2,(0,0,255),3)
    # plt.imshow("circle image", image_RGB)
    
    return circles


def houghCircle_750x750(image_RGB):

    dst = cv2.pyrMeanShiftFiltering(image_RGB, 10, 100) 
    gray = dst[:,:,1]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=10, minRadius=0, maxRadius=100)
    circles = np.uint16(np.around(circles))
    
    # for i in (circles[0,:]):
    #     # draw the outer circle
    #     cv2.circle(image_RGB,(i[0],i[1]),i[2],(0,0,255),2)
    #     # draw the center of the circle
    #     #cv2.circle(image_RGB,(i[0],i[1]),2,(255,0,0),2)
    # plt.imshow(image_RGB)

    return circles



def houghCircle_skeleton(skeleton):
    #dst = cv2.pyrMeanShiftFiltering(image, 10, 100) 
    #gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(skeleton, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=20, minRadius=0, maxRadius=100)
    circles = np.uint16(np.around(circles)) 
    
    #for i in circles[0, :]:
        #cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        #cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    #plt.imshow("circle image", image)
    
    return circles




def findConnectedComponents(num_objects,labels):
    Components = {'labels':[], 'pixels':[]}
    for i in range(num_objects):
        Components['labels'].append(i)
        Components['pixels'].append(np.sum(labels == i))
    
    return Components




def findMaxComponent(Components):
    pixels = Components['pixels']
    list = sorted(pixels,reverse=True)
    index = pixels.index(list[1])
    
    return index





def crop(image_RGB, centerCircle):
    [H,W,Z] = image_RGB.shape
    [X,Y,Z] = centerCircle.astype(np.int)
    
    factor = Z/37
    scale = int(750*factor/2) 
    
    c1 = Y-scale
    c2 = Y+scale
    c3 = X-scale
    c4 = X+scale
    
    m1 = X
    m2 = Y
    m3 = W-X
    m4 = H-Y
    
    margin = [m1,m2,m3,m4]
    mm = min(margin)

    if (c1 < 0) or (c2 > H) or (c3 < 0) or (c4 > W):
        image_crop = image_RGB[Y-mm:Y+mm,X-mm:X+mm,0:3]
    
    else:
        image_crop = image_RGB[c1:c2,c3:c4,0:3]

    return image_crop

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








