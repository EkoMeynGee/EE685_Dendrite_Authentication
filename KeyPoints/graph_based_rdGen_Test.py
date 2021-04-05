# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 02:33:40 2020

@author: MaxGr
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import math
import mat4py
import random
import datetime

import skimage.io as skio
import skimage.color as skcolor
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi

import colorextract as cE
import colorextract_Tools as cT

import graph_based_rdGen_Tools as gT



workdir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Test Sample/"



#
start = datetime.datetime.now()


for i in range(1,11):
    
    testFile = workdir+'b'+str(i)+'.png'
    
    print('Image '+str(i)+' is processing...')
    
    skeleton = cE.colorextract(testFile)
    
    
    image = skeleton[0]
    circleInfo = skeleton[1]
    
    plt.imshow(image)
    
    
    
    #
    [height, width] = image.shape
    centerx = circleInfo[0]
    centery = circleInfo[1]
    radius = circleInfo[2]
    
    
    
    #ADD 1st mask
    initial_image = image
    
    rootinfo = {'x':centerx, 'y':centery, 'radius':radius}
    
    
    
    TrueDotsSet = gT.findInitialDots(image, 1, rootinfo, [])
    
    
    
    
    
    #Check initialDots existance and store
    # index = 1;
    
    points = {'P':[], 'x':[], 'y':[]}
    
    for k in range(len(TrueDotsSet)):
        #     if (initial_image(TrueDotsSet(k,1), TrueDotsSet(k,1)) ~= 1)
        #         continue
        #     end
        points['P'].append(k)
        points['x'].append(TrueDotsSet[k][0])
        points['y'].append(TrueDotsSet[k][1])
        
        #eval(['points.p' num2str(k) '.x = TrueDotsSet(k,2);']);
        #eval(['points.p' num2str(k) '.y = TrueDotsSet(k,1);']);
        #     index = index + 1;
    
    
    # imshow(image)
    # viscircles([TrueDotsSet(:,2) TrueDotsSet(:,1)],ones(1,size(TrueDotsSet,1)) * 0.3);
    parent = {'level':1, 'x':rootinfo['x'], 'y':rootinfo['y']}
    
    
    
    # The input is subnode for central node, parent and image,...
    # the output is overthrough node based on the input nodes.
    
    result = gT.newNode_Search(TrueDotsSet, initial_image, parent,[], [], 1, circleInfo)
    
    
    
    a = np.array(result).T
    b = np.vstack((a[0],a[1])).T.tolist()
    
    circleDraft = gT.drawCircles(initial_image,b)
    cv2.imwrite('KeyPoints_'+str(i)+'.png',circleDraft)




#
end = datetime.datetime.now()
print ('Running time is ', end-start)




