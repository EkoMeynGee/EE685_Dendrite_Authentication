# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:12:04 2020

@author: MaxGr
"""



import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import math
import mat4py
import random
import copy

import skimage.io as skio
import skimage.color as skcolor
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi

from func_timeout import func_set_timeout



import sys
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/image_pre_processing')
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/KeyPoints')

import colorextract as cE
import colorextract_Tools as cT

import graph_based_rdGen_Tools as gT



# testFile = 'C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Test Sample/b5.png'

# skeleton = cE.colorextract(testFile)

# image = skeleton[0]
# circleInfo = skeleton[1]

# plt.imshow(image)

# varargin  = []


# circleInfo = circleRef

# detail_5x(initial_image, 435, 425)


# image = image_180
# circleInfo = circleTest


@func_set_timeout(5)
def graph_based_rdGen(image, circleInfo, varargin):

    
    #
    [height, width] = image.shape
    centerx = circleInfo[0]
    centery = circleInfo[1]
    radius = circleInfo[2]
    
    
    
    #ADD 1st mask
    initial_image = image
    
    rootinfo = {'x':centerx, 'y':centery, 'radius':radius}
    
    
    
    TrueDotsSet = gT.findInitialDots(initial_image, 1, rootinfo, [])
    
    
    
    
    
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
    
    result = gT.killRepeat(result)
    
    # result = gT.killInitial(result)
    
    a = np.array(result).T
    b = np.vstack((a[0],a[1])).T.tolist()
    
    circleDraft = gT.drawCircles(initial_image,b)
    #cv2.imwrite('KeyPoints_'+str(i)+'.png',circleDraft)
    
    
    
    
    
    # hold all;
    # plot(result(:,2),result(:,1),'.','MarkerSize',20);
    # prepare desired graph
    
    
    new_result = np.zeros((len(result), 14), object)
    
    for i in range(len(result)):
        # node[Rx(the distance between parent and child), Ry (same as
        # before), angle, type ,level, parentx, parenty, x, y, middle radius]
        # 11/19/18 modified angle to atan2 and make it as degree  Zaoyi
        new_result[i,0] = result[i][0] - result[i][4]
        new_result[i,1] = result[i][1] - result[i][5]
        
        new_result[i, 2] = gT.azimuthAngle((result[i][0],result[i][1]),(result[i][4],result[i][5]),(centerx,centery))
        
        # new_result[i,13] = gT.azimuthAngle(centerx,centery,result[i][4],result[i][5])
        # new_result[i, 2] = gT.azimuthAngle(result[i][0],result[i][1],result[i][4],result[i][5]) - new_result[i,13]
            
        new_result[i,3] = result[i][2]
        new_result[i,4] = result[i][3]
        new_result[i,5] = result[i][4]
        new_result[i,6] = result[i][5]
        new_result[i,7] = result[i][0]
        new_result[i,8] = result[i][1]
        new_result[i,9] = radius
        
    
    
    length_level = int(max(new_result[:,4])) # The highest depth
    
    
    
    
    # 
    # figure, imshow(image)
    # viscircles([new_result(:,8) new_result(:,9)],ones(1,size(new_result,1)) * 0.5);
    
    
    
    
    # Make the node set to have an id, which is based on angle
    # Modified Zaoyi 11/25/2018
    
    new_result = new_result[new_result[:,2].argsort()]
    
    for z in range(1, length_level+1):      # To get the nodes with in the same level
        samelevely = np.where(new_result[:,4]==z)[0]
        
        for i in range(len(samelevely)):
            new_result[samelevely[i]][9] = i+1
            
    
    
    # Mark the node set to have ability to handle the children     ###
    # Modified Zaoyi 11/18/2018
    
    new_result[:,11] = range(1,len(new_result)+1)
    
    cellnodes = copy.deepcopy(new_result)
    
    
    # Link children method, see specfic in the linkchildern method ## modified
    # Zaoyi Chi
    cellnodes = gT.linkchildren(cellnodes, new_result)
    
    for param in range(len(cellnodes)):
        if (cellnodes[param,3] == 1):
            cellnodes[param,3] = 'bifurcation'
            
        elif (cellnodes[param,3] == 2):
            cellnodes[param,3] = 'end'
            
        elif (cellnodes[param,3] == 0):
            cellnodes[param,3] = 'initial'
    
    # Link siblings method, only for same parent and same level are siblings #
    # modified Zaoyi Chi
    
    cellnodes = gT.linksibling(cellnodes, new_result)
    
    
    
    
    
    
    if (varargin  == []) or (varargin  == 0):
        # Build TreeStructure ###Modified Zaoyi
        [TreeStruct, distinfo, angleinfo] = gT.buildTreeStrcut(cellnodes, new_result, rootinfo, [])
        
    else:
        varargin.append(distinfo)
        varargin.append(angleinfo)
        # distinfo = varargin[0]
        # angleinfo = varargin[1]
        TreeStruct = gT.buildTreeStrcut(cellnodes, new_result, rootinfo, varargin)
        ###


    return [TreeStruct, new_result, cellnodes, distinfo, angleinfo, circleDraft] 







# testFile = 'C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Test Sample/b5.png'

# skeleton = cE.colorextract(testFile)

# image = skeleton[0]
# circleInfo = skeleton[1]

# plt.imshow(image)

# varargin  = []


# TEMP = graph_based_rdGen(image, circleInfo, varargin)
















