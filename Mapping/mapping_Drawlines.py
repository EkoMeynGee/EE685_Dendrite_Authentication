# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 04:17:40 2021

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
import time


import skimage.io as skio
import skimage.color as skcolor
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi


import sys
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/image_pre_processing')
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/KeyPoints')
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Mapping')

import colorextract as cE
import colorextract_Tools as cT

import graph_based_rdGen as gG
import graph_based_rdGen_Tools as gT

import mappingTest_Tools as mTT
import mappingTest as mT


workdir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Mapping/"
imagedir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Test Sample/"
KeyPoints_List = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/KeyPoints_List/"


testFile_rotate = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/IMG_20210225_192252.jpg"



#======================================================================
image_index = 1

image_name        = 'b'+str(image_index)
#image_name_rotate = 'b'+str(image_index)+'rotat20'
testFile        = imagedir + image_name +'.png'
# testFile_rotate = imagedir + image_name_rotate +'.png'

image_testFile = skio.imread(testFile)[:,:,0:3]
plt.imshow(image_testFile)


#testFile_rotate = skio.imread(testFile_rotate)[:,:,0:3]
#plt.imshow(testFile_rotate)


#==============================================================
skeleton1 = cE.colorextract(testFile)

image = skeleton1[0]
circleInfo = skeleton1[1]
varargin  = []


#Ref
plt.imshow(image)

circleRef = cT.houghCircle_skeleton(image)[0][0]

image_ref = gG.graph_based_rdGen(image, circleRef, varargin)

tree_ref   = image_ref[0]
#new_result = image_ref[1]
cell_ref   = image_ref[2]
# distinfo   = image_ref[3]
# angleinfo  = image_ref[4]




#
# tree_ref = np.load(KeyPoints_List+image_name+".npy", allow_pickle=True)


#==============================================================
start = time.time()

skeleton2 = cE.colorextract(testFile_rotate)


image_180 = skeleton2[0]
circleInfo = skeleton2[1]
varargin  = []

#Test
# image_180 = np.rot90(image, 2)
# image_180 =  np.rot90(image_180, 2)

plt.imshow(image_180)

circleTest = cT.houghCircle_skeleton(image_180)[0][0]

image_test = gG.graph_based_rdGen(image_180, circleTest, varargin)

tree_test   = image_test[0]
# new_result = image_test[1]
cell_test   = image_test[2]
# distinfo   = image_test[3]
# angleinfo  = image_test[4]


# end = time.time()
# print('Timecost: ', (end-start))



# start = time.time()
#
[CMTree1, Rate, CMTree2, i_mat] = mT.mappingTest_Fast(tree_test, tree_ref, 0.5, [], [], 0, tree_test, tree_ref, 0.4, 0.6, [], [])



#
image_linked = mTT.matchedNodeDrawLine(image_180, image, CMTree1, CMTree2)
cv2.imwrite(workdir+'image_linked_'+image_name+'.png',image_linked)

end = time.time()
print('Timecost: ', (end-start))















# #
# raw_image1 = skio.imread(testFile_rotate)[:,:,0:3]
# plt.imshow(raw_image1)
    
# raw_image2 = skio.imread(testFile)[:,:,0:3]
# plt.imshow(raw_image2)


# image_linked_RAW = mTT.matchedNodeDrawLine_RAW(raw_image1, raw_image2, CMTree1, CMTree2)
# cv2.imwrite(workdir+'image_linked_RAW_'+image_name+'.png',image_linked_RAW[:,:,::-1])

  







