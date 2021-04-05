# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:36:44 2020

@author: MaxGr
"""



import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import math
import mat4py
import random
import skimage.io as skio
import skimage.color as skcolor
import colorextract_Tools as cT
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi

workdir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/image_pre_processing"

# testFile = 'b5.png'

# imageFile = testFile

# imageFile = testFile_rotate


def colorextract(imageFile):
        

    # Read image
    image_RGB = skio.imread(imageFile)[:,:,0:3]
    plt.imshow(image_RGB)
    [H,W,z] = image_RGB.shape
    
    if min(H,W) >= 1000:
        factor = min(H,W)/1000
        image_RGB = cv2.resize(image_RGB, (int(W/factor),int(H/factor)), interpolation = cv2.INTER_AREA)
    
    
    # Size detection
    circles = cT.houghCircle(image_RGB)[0]
    centerCircle = circles[0]
    [X,Y,Z] = centerCircle.astype(np.int)

    # Crop to universal size
    image_RGB = cT.crop(image_RGB, centerCircle)
    image_RGB = cv2.resize(image_RGB, (750,750), interpolation = cv2.INTER_AREA)
    plt.imshow(image_RGB)


    # Detect center circle
    circles = cT.houghCircle_750x750(image_RGB)[0]
    centerCircle = circles[0]
    
    if centerCircle[2] < 35:
        image_RGB = cT.crop(image_RGB, centerCircle)
        image_RGB = cv2.resize(image_RGB, (750,750), interpolation = cv2.INTER_AREA)
        plt.imshow(image_RGB)
        circles = cT.houghCircle_750x750(image_RGB)[0]
        centerCircle = circles[0]
    
    
    
    R = 5
    
    circle1 = np.zeros(image_RGB.shape, np.uint8)
    circle2 = np.zeros(image_RGB.shape, np.uint8)
    
    circleMask1 = cv2.circle(circle1, (centerCircle[0],centerCircle[1]), centerCircle[2]+R, (255, 255, 255), -1)
    circleMask2 = cv2.circle(circle2, (centerCircle[0],centerCircle[1]), centerCircle[2]+R-1, (255, 255, 255), -1)
    
    #plt.imshow(circleMask1)
    #plt.imshow(circleMask2)
    
    circleMask1 = circleMask1[:,:,2].astype(bool).astype(np.uint8)
    circleMask2 = circleMask2[:,:,2].astype(bool).astype(np.uint8)
    
    #plt.imshow(circleMask2)
    #plt.imshow(circleMask1-circleMask2)
    
    
    circleInfo = centerCircle
    circleInfo[2] = circleInfo[2]+R-1
    
    
    
    #convert frame from RGB to YCBCR colorspace
    image_LAB = skcolor.rgb2lab(image_RGB)
    #plt.imshow(image_LAB)
    #cv2.imwrite('image_LAB.png',image_LAB)
    #image_LAB.save('image_LAB.png')
    #skio.imsave('image_LAB.png', image_LAB)
    #plt.axis('off')
    #plt.savefig('image_YCbCr.png')
    
    
    
    
    
    
    #pixel_labels = imsegkmeans(ab, 3, 'NumAttempts', 3);
    kmeansImage = cT.imsegkmeans(image_LAB,3)
    image_kmeans = kmeansImage[:,:,2]
    # figure, imshow(pixel_labels,[])
    
    #plt.imshow(image_kmeans)
    #plt.axis('off')
    np.unique(image_kmeans)
    #cv2.imwrite('image_kmeans.png',image_kmeans)
    
    
    #
    ret,image_binary = cv2.threshold(image_kmeans,127,255,cv2.THRESH_BINARY)
    image_bw = image_binary.astype(bool).astype(np.uint8)
    #plt.imshow(image_bw)
    np.unique(image_bw)
    #cv2.imwrite('image_bw.png',image_bw)
    
    
    
    #
    image_bw = image_bw | circleMask2
    #plt.imshow(image_bw)
    
    
    #
    image_bw = cT.erosion(image_bw, 5, 1)
    #plt.imshow(image_bw)
    
    image_bw = cT.openimage(image_bw, 3, 1)
    #plt.imshow(image_bw)
    
    
    image_bw = image_bw.astype(bool)

    
    
    
    
    #
    image_new = skmorph.remove_small_objects(image_bw, min_size=700, connectivity=2, in_place=False)
    #plt.imshow(image_new)
    
    image_new = image_new.astype(np.uint8)
    num_objects,labels = cv2.connectedComponents(image_new)
    
    
    ComponentList = cT.findConnectedComponents(num_objects,labels)
    
    labels_dendrite = ComponentList['labels'][cT.findMaxComponent(ComponentList)]
    
    labels[labels!=labels_dendrite]=0
    
    
    image_segmented = labels.astype(np.uint8)
    image_segmented[image_segmented >= 1] = 255
    
    
    #plt.imshow(image_segmented)
    #cv2.imwrite('image_segmented.png',image_segmented)
    
    #image_segmented = cv2.blur(image_segmented, (15,15))
    #plt.imshow(image_segmented)    
    
    
    
    image_segmented[image_segmented >  0] = 1
    image_skeleton = skmorph.skeletonize(image_segmented)
    plt.imshow(image_skeleton)
    
    
    
    image_skeleton = (image_skeleton | circleMask1) - circleMask2
    
    image_skeleton[image_skeleton > 0] = 255
    plt.imshow(image_skeleton)
    #cv2.imwrite('image_skeleton.png',image_skeleton)
    
    
    return image_skeleton, circleInfo
    
    
    
    
#skeleton = colorextract(testFile)
    
    
    
    






