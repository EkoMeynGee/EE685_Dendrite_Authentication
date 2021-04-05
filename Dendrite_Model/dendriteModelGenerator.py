# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:38:51 2020

@author: MaxGr
"""

import matplotlib.pyplot as plt
import numpy as np
import dendriteModelGenerator_Tools as dT
import cv2
import random
import math
import datetime
import time

#This part put a circle into the center of image



#numBranch = 10

def dendriteModelGenerator(numBranch):
    
    
    image = np.zeros((800,800)).astype("uint8")
    
    height = image.shape[0]
    width  = image.shape[1]
        
    
    circleCenter = {'x':width/2, 'y':height/2, 'radius':35}
    
    
    
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
     
        if ((randindexTemp in randIndexSet) or dT.checkClosePoints(tempDot, randInitalSet, 30)):
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
    
    
        
    DendriteImage = dT.subTreeGen(PointsNode, image, circleCenter)
    DendriteImage = dT.imdilate(DendriteImage)
        
    #plt.imshow(DendriteImage)
    DendriteImage[DendriteImage == 1] = 255 # first layer white
    cv2.imwrite('RndDH_Temp.png',DendriteImage)
        

    return DendriteImage



    
#Test

#
start_total = datetime.datetime.now()

workdir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/artificial_colored"


#
Group = 1
numBranch = 4
numImages = 1000


for i in range(1,numImages):
    start = time.time()
    
    DendriteSample = dendriteModelGenerator(numBranch)
    cv2.imwrite('Group_'+str(Group)+'_DHSample_'+str(numBranch)+'_Branch_'+str(i)+'.png',DendriteSample)
    
    end = time.time()
    print("image_"+str(i)+" Timecost: ", (end-start))






#
end_total = datetime.datetime.now()
print ('Running time is ', end_total-start_total)





     
     
 
 
 
#plt.imshow(image)
#image[image == 1] = 255 # first layer white
#cv2.imwrite('Test_Temp.png',image)

 
 
 
 
 
 
 
 
 
 
 





