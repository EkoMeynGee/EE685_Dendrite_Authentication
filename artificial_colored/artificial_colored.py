# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:36:17 2020

@author: Hao Wang
"""

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import artificial_colored_Tools as UT
import cv2
import datetime
import mat4py
import random
import math
import os
import skimage.io as skio






#
start = datetime.datetime.now()



filePath = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Artificial_Dendrites_Validation/"
FileList = os.listdir(filePath)

newFilePath = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Artificial_Dendrites_Validation_RGB/"


image_size = 800

center = {'x':image_size/2, 'y':image_size/2, 'radius':image_size/25}



for i in range(len(FileList)):
    
    
    
    
    image_224 = skio.imread(filePath+FileList[i])
    
    
    image_RGB = UT.false_magic(image_224, center)
    
    
    
    cv2.imwrite( (newFilePath+FileList[i]).replace("png", "jpg"), image_RGB)
    
    print("image_"+str(i))


#
end = datetime.datetime.now()
print ('Running time is ', end-start)











