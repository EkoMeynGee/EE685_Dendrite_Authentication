# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:36:24 2021

@author: MaxGr
"""


import matplotlib.pyplot as plt
import numpy as np
import Dendrite_Generator_Fast_Function as dF
import cv2
import random
import math
import datetime
import time
import os
import skimage.io as skio



from func_timeout import func_set_timeout


#This part put a circle into the center of image


workdir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Dendrite_Model"

#numBranch = 10

@func_set_timeout(5)
def dendriteModelGenerator(numBranch):
    
    image_size = 800
    
    image = np.zeros((image_size,image_size)).astype("uint8")
    height = image.shape[0]
    width  = image.shape[1]
    circleCenter = {'x':width/2, 'y':height/2, 'radius':image_size/25}
    xgrid,ygrid = np.meshgrid(np.arange(1,width+1), np.arange(1,height+1))
    
    circle  = np.sqrt((circleCenter['y'] - ygrid)*(circleCenter['y'] - ygrid) + (circleCenter['x'] - xgrid)*(circleCenter['x'] - xgrid)) <= circleCenter['radius']
    circle2 = np.sqrt((circleCenter['y'] - ygrid)*(circleCenter['y'] - ygrid) + (circleCenter['x'] - xgrid)*(circleCenter['x'] - xgrid)) <= circleCenter['radius']-1
    circle3 = np.sqrt((circleCenter['y'] - ygrid)*(circleCenter['y'] - ygrid) + (circleCenter['x'] - xgrid)*(circleCenter['x'] - xgrid)) <= circleCenter['radius']-10
    
    circle  = circle.astype(np.uint8)
    circle2 = circle2.astype(np.uint8)
    circle3 = circle3.astype(np.uint8)
    
    outerCircle = image + circle - circle2
    image = image + circle - circle2 + circle3
    # plt.imshow(image)
    
    #This part randomly find the initial starts points on the circle
    yCoordin,xCoordin = np.where (outerCircle == 1)
    numDots = len(yCoordin)
    
    randIndexSet = []
    randInitalSet = np.zeros((numBranch,2)).astype(int)
    index = 0
    
    while (index != numBranch):
        randindexTemp = random.randint(0, numDots-1)
        tempDot = {'x':xCoordin[randindexTemp], 'y':yCoordin[randindexTemp]}
     
        if ((randindexTemp in randIndexSet) or dF.checkClosePoints(tempDot, randInitalSet, 5)):
            continue
    
        randIndexSet.append(randindexTemp)
        
        randInitalSet[index,0] = tempDot['x']
        randInitalSet[index,1] = tempDot['y']
        index = index + 1
    
    #Inital Probability Matrix
    InitalProbMatrix1 = np.array([0, 0, 0.05, 0.9, 0.05, 0, 0, 0])
    InitalProbMatrix2 = np.array([0, 0.05, 0.9, 0.05, 0, 0, 0, 0])
    InitalProbMatrix3 = np.array([0.05, 0.9, 0.05, 0, 0, 0, 0, 0])
    InitalProbMatrix4 = np.array([0.9, 0.05, 0, 0, 0, 0, 0, 0.05])
    InitalProbMatrix5 = np.array([0.05, 0, 0, 0, 0, 0, 0.05, 0.9])
    InitalProbMatrix6 = np.array([0, 0, 0, 0, 0, 0.05, 0.9, 0.05])
    InitalProbMatrix7 = np.array([0, 0, 0, 0, 0.05, 0.9, 0.05, 0])
    InitalProbMatrix8 = np.array([0, 0, 0, 0.05, 0.9, 0.05, 0, 0])
    
    PointsNode = {'x':[], 'y':[], 'degree':[], 'Prob_Matrix':[], 'nodeType':[], }
    #PointsNode = np.zeros((5, numBranch), dtype=object)
    
    #Calculate degree and assign Probility matrix
    for index in range(numBranch):
        PointsNode['x'].append(randInitalSet[index,0])
        PointsNode['y'].append(randInitalSet[index,1])
        PointsNode['degree'].append(math.degrees(math.atan2((circleCenter['y']-randInitalSet[index,1]),(randInitalSet[index,0]-circleCenter['x']))))
        
        degreeTemp = PointsNode['degree'][index]
 
        if (degreeTemp < 45 and degreeTemp >=0):
            percent = degreeTemp / 45        
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix1 + percent * InitalProbMatrix2) )
        elif (degreeTemp <90 and degreeTemp >= 45):
            percent = (degreeTemp - 45) / 45
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix2 + percent * InitalProbMatrix3) )
        elif (degreeTemp <135 and degreeTemp >= 90):
            percent = (degreeTemp - 90) / 45
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix3 + percent * InitalProbMatrix4) )
        elif (degreeTemp <=180 and degreeTemp >= 135):
            percent = (degreeTemp - 135) / 45;
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix4 + percent * InitalProbMatrix5) )
        elif (degreeTemp < 0 and degreeTemp >= -45):
            percent = (degreeTemp) / -45
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix1 + percent * InitalProbMatrix8) )
        elif (degreeTemp < -45 and degreeTemp >= -90):
            percent = (degreeTemp + 45) / -45
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix8 + percent * InitalProbMatrix7) )
        elif (degreeTemp < -90 and degreeTemp >= -135):
            percent = (degreeTemp + 90) / -45
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix7 + percent * InitalProbMatrix6) )
        else:
            percent = (degreeTemp + 135) / -45
            PointsNode['Prob_Matrix'].append( ((1-percent) * InitalProbMatrix6 + percent * InitalProbMatrix5) )

    DendriteImage = dF.subTreeGen(PointsNode, image, circleCenter)
    DendriteImage = dF.imdilate(DendriteImage)
        
    #plt.imshow(DendriteImage)
    DendriteImage[DendriteImage == 1] = 255 # first layer white
    cv2.imwrite('RndDH_Temp.png',DendriteImage)

    return DendriteImage





def Check_Total():
    filePath = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Artificial_Dendrites/"
    oldFileList = os.listdir(filePath)
    
    for i in range(len(oldFileList)):
        #==============================================================
    
        oldFileList[i] = (oldFileList[i].replace(".png", ""))
        oldFileList[i] = int(oldFileList[i])
        print(oldFileList[i])
    
    FileList = list(range(1,20000+1))
    newFileList = list(set(FileList).difference(set(oldFileList)))
    
    for i in range(len(newFileList)):
        numBranch = np.random.choice([3,4,5])
        try:
            DendriteSample = dendriteModelGenerator(numBranch)
        except:
            continue

        if DendriteSample.sum() >= 1000000 and DendriteSample.sum() <= 2000000:
            cv2.imwrite("Val_"+str(newFileList[i]) + '.png',DendriteSample)

        print("image_"+str(i)+ " | Total nodes: " + str(DendriteSample.sum()))
        
        
    
    






#Test
def Dendrite_20000():
    start_total = datetime.datetime.now()
    
    filePath = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Temp/"

    numImages = 20000
    index = 1
        
    while index < numImages+1:
        start = time.time()
        
        numBranch = np.random.choice([2,3,4,5,6,7])
        
        try:
            DendriteSample = dendriteModelGenerator(numBranch)
        except:
            continue
        
        plt.imshow(DendriteSample)
        image_bit = cv2.resize(DendriteSample, (128,128), interpolation = cv2.INTER_AREA) 
        plt.imshow(image_bit)
    
        if DendriteSample.sum() >= 1000000 and DendriteSample.sum() <= 2000000:
            cv2.imwrite(filePath+str(index) + '.png',DendriteSample)
            index = index+1
        
        end = time.time()
            
        print("image_"+str(index-1)+" | Timecost: "+ str(end-start) + " | Total nodes: " + str(DendriteSample.sum()))

    #
    end_total = datetime.datetime.now()
    print ('Running time is ', end_total-start_total)
    
    
    


     
     
     
     
     

     

 
def image_resize():
    filePath = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Artificial_Dendrites_Validation_RGB/"
    FileList = os.listdir(filePath)
    
    
    newFilePath = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Artificial_Dendrites_Validation_224_RGB/"
    
    start = time.time()
    
    for i in range(len(FileList)):
        image_800 = skio.imread(filePath+FileList[i])
        image_256 = cv2.resize(image_800, (224,224), interpolation = cv2.INTER_AREA)    
        cv2.imwrite(newFilePath+FileList[i],image_256)
           
        print("image_"+str(i))
    
    end = time.time()
    print("Total Timecost: "+ str(end-start))
 

 
 
 
 
 
 
 
# image_resize()
 
Dendrite_20000()
 
 
 















