# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:42:51 2021

@author: MaxGr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:29:39 2020

@author: MaxGr
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random
import copy


def imdilate(image): 
    
    kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], np.uint8)

    erosion = cv2.dilate(image,kernel,iterations = 1)

    return erosion




def checkClosePoints(PointA, InitalSet, minDist):
#Check if two points are too close, return true if too close

    numPoints = InitalSet.shape[0]
    logicalResult = False

    for index in range(numPoints):
        tempPoint = {'x':InitalSet[index,0], 'y':InitalSet[index,1]}

        dist = np.sqrt((tempPoint['x']-PointA['x'])*(tempPoint['x']-PointA['x']) + (tempPoint['y']-PointA['y'])*(tempPoint['y']-PointA['y']))
        
        if (dist < minDist):
            logicalResult = True
            break

    return logicalResult




def randNodeType():
#'1' end nodes, '2' bifuration nodes, '3' usalpoints 
#   Detailed explanation goes here
    
    Prob_Matrix = np.array([1, 10, 100000])
    Prob_Matrix = Prob_Matrix / sum(Prob_Matrix)
    
    randomVector = []
    
    for index in range(1,3+1):
        tempProb = Prob_Matrix[index-1]
        num = round(tempProb * 10000)
        if num == 0:
            continue
    
        tempVector = (index) * np.ones((1, num),dtype=int)
        randomVector = randomVector + tempVector.tolist()[0]
        
    
    random.shuffle(randomVector)
    
    randIndex = random.randint(2,len(randomVector)-1)
    
    nodeType = randomVector[randIndex]

    return nodeType


# for i in range(1000):
#     t = randNodeType_Fast()
#     print(t)

def randNodeType_Fast():
    
    ProbMatrix = np.array([0, 15, 1000])
    ProbMatrix = ProbMatrix / sum(ProbMatrix)
    
    nodeType = np.random.choice([1, 2, 3], p=ProbMatrix)
    
    return nodeType




def randPathChoose(ProbMatrix):
#   Based on the Probility matrix, we randomly choose a path

    randomVector = []
    for index in range(8):
        tempProb = ProbMatrix[index]
        num = round(tempProb * 10000)
        if num == 0:
            continue
        
        tempVector = index * np.ones((1, num),dtype=int)
        randomVector = randomVector + tempVector.tolist()[0]
        
    
    random.shuffle(randomVector)

    randIndex = random.randint(1,len(randomVector)-1)
    
    pathType = randomVector[randIndex]+1
    
    return pathType
    




def randPathChoose_Fast(ProbMatrix):
    
    pathType = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], p=ProbMatrix)
    
    return pathType
    



def nextCoords(xTemp, yTemp, pathType):
    newXnext = 0
    newYnext = 0
    if pathType == 1:
        newXnext = xTemp - 1
        newYnext = yTemp - 1
    elif pathType == 2:
        newXnext = xTemp
        newYnext = yTemp - 1
    elif pathType == 3:
        newXnext = xTemp + 1
        newYnext = yTemp - 1
    elif pathType == 4:
        newXnext = xTemp + 1
        newYnext = yTemp
    elif pathType == 5:
        newXnext = xTemp + 1
        newYnext = yTemp + 1
    elif pathType == 6:
        newXnext = xTemp
        newYnext = yTemp + 1
    elif pathType == 7:
        newXnext = xTemp - 1
        newYnext = yTemp + 1
    elif pathType == 8:
        newXnext = xTemp - 1
        newYnext = yTemp
    
    return [newXnext,newYnext]
    




def subOverlapCheaker(x, y, pathType, image):
    
    [ruleA, ruleB] = image.shape
    logicalOut = False
    
    upPathType = (pathType+1) % 8
    downPathType = (pathType-1) % 8
    
    if (upPathType == 0):
        upPathType = 8
    
    
    if (downPathType == 0):
        downPathType = 8
    
    
    for index in range(1,2+1):
        if (index == 1):
            pathTypeTemp = upPathType
        else:
            pathTypeTemp = downPathType
        
        [xnext , ynext] = nextCoords(x,     y,     pathTypeTemp)
        [xnext2,ynext2] = nextCoords(xnext, ynext, pathTypeTemp)
        
        if ((ynext2 >= ruleA) or (ynext2 <= 0) or (xnext2 >= ruleB) or (xnext2 <= 0) or (image[ynext2, xnext2] == 1)):
            logicalOut = True
            break

        
        if ((ynext >= ruleA) or (ynext <= 0) or (xnext >= ruleB) or (xnext <= 0) or (image[ynext, xnext] == 1)):
            logicalOut = True
            break
    
    return logicalOut
    

#subOverlapCheaker(393, 395, 4, image)

# xnext1 = xnext
# ynext1 = ynext

def checkFutureOverlap(xnext1, ynext1, pathType, image):
#   Main check function for avoiding overlap or future overlap
    
    [ruleA,ruleB] = image.shape
    circleCenter = {'x':ruleB / 2, 'y':ruleA / 2, 'radius': (ruleB / 2) - 20}

    [xgrid,ygrid] = np.meshgrid(np.arange(1,ruleB+1), np.arange(1,ruleA+1))
    
    circle = np.sqrt((circleCenter['y']-ygrid)*(circleCenter['y']-ygrid) + (circleCenter['x']-xgrid)*(circleCenter['x']-xgrid)) <= circleCenter['radius']    
    circle2 = np.sqrt((circleCenter['y']-ygrid)*(circleCenter['y']-ygrid) + (circleCenter['x']-xgrid)*(circleCenter['x']-xgrid)) <= circleCenter['radius'] - 1
    
    circle = circle.astype(np.uint8)
    circle2 = circle2.astype(np.uint8)

    image = image + circle - circle2;
    
    logicalOut = False
    
    for index in range(1, 6+1):
        if (index != 1):
            [x,y] = nextCoords(eval('xnext'+str(index-1)), eval('ynext'+str(index-1)), pathType)
            vars()["xnext" + str(index)] = x
            vars()["ynext" + str(index)] = y
            
        xtemp = 0
        ytemp = 0
        subChecker = 0
        xtemp = eval('xnext' +str(index))
        ytemp = eval('ynext' +str(index))
        subChecker = subOverlapCheaker(eval('xnext'+str(index)),eval('ynext'+str(index)), pathType, image)
        
        if ((ytemp >= ruleA) or (ytemp <= 0) or (xtemp >= ruleB) or (xtemp <= 0) or (image[ytemp, xtemp] == 1)):
            logicalOut = True
            break
        
        if (subChecker):
            logicalOut = True
            break
        
    return logicalOut
    
    
    

    
def checkFutureOverlap_Fast(xnext, ynext, xtemp, ytemp, image):
    
    threshold = 2
    
    dist = 10
    
    vector = [ynext-ytemp, xnext-xtemp]
    
    initpoint   = [0,0]
    futurepoint = [0,0]
    
    initpoint[0] = ytemp
    initpoint[1] = xtemp
    
    futurepoint[0] = ytemp+dist*vector[0]
    futurepoint[1] = xtemp+dist*vector[1]
    
    
    if vector[0] < 0:
        initpoint[0] = futurepoint[0]
        futurepoint[0] = ytemp
   
    if vector[1] < 0:
        initpoint[1] = futurepoint[1]
        futurepoint[1] = xtemp
        
    if vector[0] == 0:
        futurepoint[0] = ytemp+1
    
    if vector[1] == 0:
        futurepoint[1] = xtemp+1
        
    if vector == [0,0]:
        return False


    sumpoints = np.sum(image[initpoint[0]:futurepoint[0], initpoint[1]:futurepoint[1]])

    if sumpoints >= threshold:
        return True
    else:
        return False
    



def reAssignProbMatrix(pathType, ProbMatrix):
#   Based on the direction it went, re-assign probility matrix
    dirType = pathType-1
    newProbMat = [0, 0, 0, 0, 0, 0, 0, 0]
    newProbMat[dirType] = ProbMatrix[dirType] * 0.99
    diff = (ProbMatrix[dirType] - newProbMat[dirType]) / 7
    
    for index in range(8):
        if (index == dirType):
            continue

        newProbMat[index] = ProbMatrix[index] + diff

    
    return newProbMat






def exponential_decay(t, start, lamda, end):
    alpha = np.log(start / end) / lamda
    l = - np.log(start) / alpha
    decay = np.exp(-alpha * (t + l))
    return decay






def subTreeGen(PointsNode , image, circleCenter):
#****** PathType detail goes here *********
#------------1   2    3--------------------
#------------8  null  4--------------------
#------------7   6    5--------------------

    image_size = image.shape[0]
    #print(image.sum())

    numNodes = len(PointsNode['x'])

    SubProbMatrix1 = np.array([0, 0, 0.05, 0.9, 0.05, 0, 0, 0])
    SubProbMatrix2 = np.array([0, 0.05, 0.9, 0.05, 0, 0, 0, 0])
    SubProbMatrix3 = np.array([0.05, 0.9, 0.05, 0, 0, 0, 0, 0])
    SubProbMatrix4 = np.array([0.9, 0.05, 0, 0, 0, 0, 0, 0.05])
    SubProbMatrix5 = np.array([0.05, 0, 0, 0, 0, 0, 0.05, 0.9])
    SubProbMatrix6 = np.array([0, 0, 0, 0, 0, 0.05, 0.9, 0.05])
    SubProbMatrix7 = np.array([0, 0, 0, 0, 0.05, 0.9, 0.05, 0])
    SubProbMatrix8 = np.array([0, 0, 0, 0.05, 0.9, 0.05, 0, 0])
        
    
    for index in range(numNodes):
        
        Type = randNodeType_Fast()
        
        xTemp = PointsNode['x'][index]
        yTemp = PointsNode['y'][index]
        ProbMatrix = PointsNode['Prob_Matrix'][index]

        PointsNode['nodeType'].append(Type)

        image[yTemp,xTemp] = 1
        
        #
        root_distance = np.sqrt((circleCenter['y']-yTemp)**2 + (circleCenter['x']-xTemp)**2)
        killbranch = exponential_decay(root_distance, 1, image_size, 0.999)
        killer = np.random.choice([0,1], p=[killbranch,1-killbranch])

        if root_distance >= image_size*0.95 or killer == 1:
            # print("Killed by lambda! Rootdist: "+str(root_distance)+" killbranch: "+str(killbranch)+" killer: "+str(killer))
            continue

#Test
        
        # print(image.sum())
        
        if Type == 1:
            print("End: " + str([xTemp,yTemp]))
            Type == 2
        
        
        
        if Type == 2:
            
            SubPointsB = {'x':[], 'y':[], 'degree':[], 'Prob_Matrix':[], 'nodeType':[], }
            
            #SubPoints = np.zeros((10, 5), dtype=object)

            numSubBranches = np.random.choice([1,2], p=[0.999,0.001])
            pathHistories = []
            degreeHistories = []
            junction = 1
            trialCount = 0
            trialCount2 = 0
            
            while junction <= (numSubBranches + 1):
                rndDegreeChange = random.randint(0, 60)
                degreeTemp = math.degrees(math.atan2((circleCenter['y']-yTemp),(xTemp-circleCenter['x'])))
                
                degreeChangeRdIndex = np.random.choice([1,2,3])
                
                if (degreeChangeRdIndex in degreeHistories):
                    if (trialCount2 >= image_size/40):
                        # print("Tail: "+str(trialCount2))
                        break
    
                    trialCount2 = trialCount2 + 1
                    continue
                      
                degreeHistories.append(degreeChangeRdIndex)
                
                
                
                if degreeChangeRdIndex == 1:
                    degree = degreeTemp + rndDegreeChange
                elif degreeChangeRdIndex == 2:
                    degree = degreeTemp - rndDegreeChange
                elif degreeChangeRdIndex == 3:
                    degree = degreeTemp
                
                
                
                if (degree > 180):
                    degree = degree - 360
                elif (degree < -180):
                    degree = 360 + degree
            
                    
                
                if (degree < 45 and degree >=0):
                    percent = degree / 45
                    Prob_Matrix = (1-percent) * SubProbMatrix1 + percent * SubProbMatrix2
                elif (degree < 90 and degree >= 45):
                    percent = (degree - 45) / 45
                    Prob_Matrix = (1-percent) * SubProbMatrix2 + percent * SubProbMatrix3
                elif (degree < 135 and degree >= 90):
                    percent = (degree - 90) / 45
                    Prob_Matrix = (1-percent) * SubProbMatrix3 + percent * SubProbMatrix4
                elif (degree <=180 and degree >= 135):
                    percent = (degree - 135) / 45
                    Prob_Matrix = (1-percent) * SubProbMatrix4 + percent * SubProbMatrix5
                elif (degree < 0 and degree >= -45):
                    percent = degree / -45
                    Prob_Matrix = (1-percent) * SubProbMatrix1 + percent * SubProbMatrix8
                elif (degree < -45 and degree >= -90):
                    percent = (degree + 45) / -45
                    Prob_Matrix = (1-percent) * SubProbMatrix8 + percent * SubProbMatrix7
                elif (degree < -90 and degree >= -135):
                    percent = (degree + 90) / -45
                    Prob_Matrix = (1-percent) * SubProbMatrix7 + percent * SubProbMatrix6
                else:
                    percent = (degree + 135) / -45
                    Prob_Matrix = (1-percent) * SubProbMatrix6 + percent * SubProbMatrix5
                
                
                pathType = randPathChoose_Fast(Prob_Matrix)
                
                if (pathType in pathHistories):
                    if (trialCount >= image_size/40):
                        # print("Tail: "+str(trialCount))
                        break
                    
                    trialCount = trialCount + 1
                    continue
                
                pathHistories.append(pathType)
                [xnext,ynext] = nextCoords(xTemp, yTemp, pathType)
                
                
                
                
                if checkFutureOverlap(xnext, ynext, pathType, image):
                    numSubBranches = numSubBranches - 1
                    continue
                
                if checkFutureOverlap_Fast(xnext, ynext, xTemp, yTemp, image):
                    continue
                
                
                SubPointsB['x'].append(xnext)
                SubPointsB['y'].append(ynext)
                SubPointsB['Prob_Matrix'].append(Prob_Matrix)

                junction = junction + 1
        
            
            if SubPointsB['x'] == []:
                # print("Fool!")
                continue
            else:
                image = subTreeGen(SubPointsB, image, circleCenter)
                
            #print(image.sum())
            
            
        elif Type == 3:

            SubPointsC = {'x':[], 'y':[], 'degree':[], 'Prob_Matrix':[], 'nodeType':[], }
            
            pathType = randPathChoose_Fast(ProbMatrix)
            [xnext,ynext] = nextCoords(xTemp, yTemp, pathType)
            
            # if checkFutureOverlap(xnext, ynext, pathType, image):
                # print("Overlap: " + str([xTemp,yTemp]))
                # continue
            
            if checkFutureOverlap_Fast(xnext, ynext, xTemp, yTemp, image):
                continue

            SubPointsC['x'].append(xnext)
            SubPointsC['y'].append(ynext)
            SubPointsC['Prob_Matrix'].append(reAssignProbMatrix(pathType, ProbMatrix))
            
            image = subTreeGen(SubPointsC, image, circleCenter)
            
            #print(image.sum())
        

    newimage = image
    # print(image.sum())
    # plt.imshow(image)

    return newimage




#reAssignProbMatrix(pathType, ProbMatrix)




#checkFutureOverlap(4, 4, 4, image)








#SubPoints = np.zeros((1, 5), dtype=object)

#SubPoints[0][0] = 200
#SubPoints[0][1] = 600
#SubPoints[0][3] = [0.01771711, 0, 0, 0, 0,0.03228289, 0.59880919, 0.35119081]


#subTreeGen(SubPoints, image, circleCenter)


#plt.imshow(image)





















































