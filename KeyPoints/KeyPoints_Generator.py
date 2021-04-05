# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:34:39 2021

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
import func_timeout


from glob import glob
from func_timeout import func_set_timeout


import skimage.io as skio
import skimage.color as skcolor
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi

import os
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


TestFile_path = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Testing_image/"
FileList = os.listdir(TestFile_path)

KeyPoints_List = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/KeyPoints_List/"


def KeyPoints_Auto(File, iteration):
    start = time.time()
    
    keypoints_list = []
    keypoints_Best = []
    image_keypoints = []
    
    for i in range(iteration):
        
        try:
            skeleton = cE.colorextract(File)
        except:
            continue
            
        image = skeleton[0]
        circleInfo = skeleton[1]
        varargin  = []
        #
        plt.imshow(image)
        
        circleRef = cT.houghCircle_skeleton(image)[0][0]
        
        try:
            image_ref = gG.graph_based_rdGen(image, circleRef, varargin)
        except:
            continue
        
        tree_ref   = image_ref[0]
        cell_ref   = image_ref[2]
        image_keypoints = image_ref[5]
        
        keypoints_list = tree_ref
        
        if len(keypoints_list) > len(keypoints_Best):
            
            keypoints_Best = keypoints_list
        
    end = time.time()
    print('Timecost: ', (end-start))
    
    return [keypoints_Best, image_keypoints]







for i in range(len(FileList)):
    #==============================================================
    testFile = TestFile_path + FileList[i]
    iteration = 100
    
    FileName = FileList[i].replace(".png", "")
    print(FileName)
    
    [keypoints_Best , image_keypoints] = KeyPoints_Auto(testFile , iteration)
    
    try:
        np.save(KeyPoints_List+FileName+".npy" , keypoints_Best)
        cv2.imwrite(KeyPoints_List+FileName+"_keypoints.png",image_keypoints)
    except:
        continue
    
    #==============================================================


print('Done...')









