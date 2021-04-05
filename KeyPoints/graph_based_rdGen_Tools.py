# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:44:05 2020

@author: MaxGr
"""



import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import copy
import math
import mat4py
import random
import skimage.io as skio
import skimage.color as skcolor
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi

import sys
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/image_pre_processing')
 
import colorextract as cE
import colorextract_Tools as cT


# skleton = image
# iterTimes = 1
# rootinfo = rootinfo
# exdotsSet = []



# def setdiff(list_X, list_Y):
    
#     list_new = list_X + list_Y
#     list_out = list_new[:]
    
#     for i in range(len(list_new)):
#         if list_new.count( list_new[i] ) > 1:
#             list_out.remove(list_new[i])
        
#     return list_out




# Point 1 = self
# Point 2 = parent
# Point 3 = root
def azimuthAngle(point_1, point_2, point_3):
    
    root = point_3
    
    point_1 = [ point_1[0]-root[0] , -(point_1[1]-root[1]) ]
    point_2 = [ point_2[0]-root[0] , -(point_2[1]-root[1]) ]
    point_3 = [ point_3[0]-root[0] , -(point_3[1]-root[1]) ]
    
    angle_parent = math.degrees(math.atan2( point_2[1] , point_2[0] ))
    angle_self   = math.degrees(math.atan2( point_1[1]-point_2[1] , point_1[0]-point_2[0] ))
    
    if angle_parent*angle_self <0:
        if angle_parent <0:
            angle_parent = 360-abs(angle_parent)
        if angle_self <0:
            angle_self = 360-abs(angle_self)
    
    angle_absolute = angle_self - angle_parent
    
    return angle_absolute






def angle_triplepoints(point_1, point_2, point_3):

    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1] - point_2[1]))
    
    if a==0 or b==0 or c==0:
        return 0
    
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))

    return 180-B

# cal_ang((261,246),(288,239),(376,374))





def setdiff(list_X, list_Y):
    
    list_new = []
    
    for i in range(len(list_X)):
        if list_X[i] not in list_Y:
            list_new.append(list_X[i])
        
    return list_new




def setunion(list_X, list_Y):
    
    list_new = list_X + list_Y
    
    for i in range(len(list_new)):
        if list_new.count(list_new[i]) > 1:
            list_new[i] = 'REPEAT'
        
    while list_new.count('REPEAT') > 0:
        list_new.remove('REPEAT')
    
    return list_new



def setintersec(list_X, list_Y):
    
    if list_X==[] or list_Y==[]:
        return []
    
    list_new = list_X + list_Y
    list_out = []
    
    for i in range(len(list_new)):
        if list_new.count( list_new[i] ) > 1:
            list_out.append(list_new[i])
          
    for i in range(len(list_out)):
        if list_out.count(list_out[i]) > 1:
            list_out[i] = 'REPEAT'
        
    while list_out.count('REPEAT') > 0:
        list_out.remove('REPEAT')
        
    return list_out



# list_X = [[1, 2], [398, 388], [397, 387], [397, 388], [398, 386]]
# list_Y = [[1, 2], [3, 4], [5, 6], [398, 388], []]

# setdiff(list_X,list_Y)
# setunion(list_X,list_Y)
# setintersec(list_X,list_Y)





def imageMat(image, list_X, list_Y):
    mat = []
    for i in range(len(list_X)):
        for j in range(len(list_Y)):
            mat.append(image[list_X[i] , list_Y[j]])
    mat = np.array(mat)
    mat = mat.reshape((len(list_Y) , len(list_X)))   
    return mat








# skleton = initial_image
# iterTimes = 1
# rootinfo
# exdotsSet = []






def findInitialDots(skleton, iterTimes, rootinfo, exdotsSet):
    #Find the initial points of each branch of the skleton
    #The recursive function to find the points, iterTimes starts with 0
    #skleton = skleton.astype(bool).astype(np.uint8)
    
    [height,length] = skleton.shape  

    circle1 = np.zeros((height, length, 3), np.uint8)
    firstMask = cv2.circle(circle1, (rootinfo['x'],rootinfo['y']), rootinfo['radius'] + 5, (255, 255, 255), -1)
    firstMask = firstMask[:,:,0]#.astype(bool).astype(np.uint8)
    #plt.imshow(firstMask)

    skletonTemp = skleton - firstMask
    skletonTemp[skletonTemp!=255] = 0
    # plt.imshow(skletonTemp)
    # cv2.imwrite('skletonTemp.png',skletonTemp)



    circle2 = np.zeros((height, length, 3), np.uint8)
    secondMask = cv2.circle(circle2, (rootinfo['x'],rootinfo['y']), rootinfo['radius'] + 5, (255, 255, 255), 2)
    secondMask = secondMask[:,:,0]#.astype(bool)#.astype(np.uint8)
    # plt.imshow(secondMask)
    
    initialDotsImage = skletonTemp & secondMask
    # plt.imshow(initialDotsImage)
    # cv2.imwrite('initialDotsImage.png',initialDotsImage)



    [dotsSety,dotsSetx] = np.where(initialDotsImage > 0 )
    dotsSet = np.vstack((dotsSetx,dotsSety)).T.tolist()
    dotsSet = killToClose(dotsSet)
    
    
    #Solve the new dotsSet have two adjacent dots, only have the smaller one
    numDots = len(dotsSet)
#    dotsSet = dotsSet.tolist()
    tempdotsOg = dotsSet[:]
    
    for index in range(numDots):
        tempDot = tempdotsOg[index]        
        tempDotsSet = setdiff(dotsSet, [tempDot])
        
        for index2 in range(len(tempDotsSet)):
            tempDot2 = tempDotsSet[index2]
            
            if ((tempDot[0]==tempDot2[0] or tempDot[0]==tempDot2[0]+1 or tempDot[0]==tempDot2[0]-1) and (tempDot[1]==tempDot2[1] or tempDot[1]==tempDot2[1]+1 or tempDot[1]==tempDot2[1]-1)):      
                
                r1 = ((tempDot[0]-rootinfo['x'])**2 + (tempDot[1]-rootinfo['y'])**2)**0.5
                r2 = ((tempDot2[0]-rootinfo['x'])**2 + (tempDot2[1]-rootinfo['y'])**2)**0.5
                
                if (r1 <= r2):
                    dotsSet = setdiff(dotsSet, [tempDot2])
                else:
                    dotsSet = setdiff(dotsSet, [tempDot])
    
                break

    #Do comparison between the new dotsSet and previous one
    numDots = len(dotsSet)
    numexDots = len(exdotsSet)
    tempSet = dotsSet[:]
    
    for index in range(numDots):
        X = tempSet[index][0]
        Y = tempSet[index][1]
        
        for index2 in range(numexDots):
            exX = exdotsSet[index2][0]
            exY = exdotsSet[index2][1]
            
            if ((exX==X or exX==(X+1) or exX==(X-1)) and (exY==Y or exY==(Y+1) or exY==(Y-1))):
                
                dotsSet = setdiff(dotsSet,[[X, Y]])
                
                break

    if (iterTimes == 1):
        TrueDotsSet = dotsSet
    else:
        if exdotsSet != []:
            TrueDotsSet = dotsSet + exdotsSet
        else:
            TrueDotsSet = dotsSet
    
    if (iterTimes == 2):
        return TrueDotsSet
    else:
        TrueDotsSet = findInitialDots(skleton, iterTimes + 1, rootinfo, TrueDotsSet)

    
    
    return TrueDotsSet
    






    # function type_counter = pointDetection(littleFrame)
    #
    # #littleFrame = imageWoCircle(boxFramex, boxFramey);
    # littleFrame_list = [littleFrame(1,1), littleFrame(1,2), littleFrame(1,3), littleFrame(2,3),...
    #     littleFrame(3,3), littleFrame(3,2), littleFrame(3,1), littleFrame(2,1)];
    # type_counter = 0;
    #
    # for index = 1:8
    #     if (index == 1)
    #         temp_a = littleFrame_list(8);
    #     else
    #         temp_a = littleFrame_list(index-1);
    #     end
    #
    #     type_counter = type_counter + abs(littleFrame_list(index) - temp_a);
    # end
    #
    # end
    #
    
    
    
    
def middleOfThree(newdots):
    check = np.array(newdots).T
    #len(np.unique(check[0]))
    #len(np.unique(check[1]))

    if (len(np.unique(check[0])) == 1) or (len(np.unique(check[1])) == 1):
        
        indexM = 1
        
    else:
        for indexM in range(3):
            if (check[0][indexM] in check[0]) and (check[1][indexM] in check[1]):
                break

    index =  setdiff([0,1,2], [indexM])
    index_sort = [indexM, index[0], index[1]]
    
    return index_sort
    
    
    
    
    
# inputFrame=matFrame
# dotsSet = setdiff(dotsSet,newdots)

    
    
def go_old_way(inputFrame, tempX, tempY, dotsSet, newdots, nodesSet):
    
    boxFrame = copy.deepcopy(inputFrame)
    boxFrame[1,1] = 0
    #[Ys, Xs] = find((boxFrame == 1))
    [Ys,Xs] = np.where(boxFrame > 0)
    exactXs = Xs + tempX - 1
    exactYs = Ys + tempY - 1
    
    exact = np.vstack((exactXs,exactYs)).T.tolist()

    old_dot = setintersec(exact, dotsSet)
    
    check = np.array(nodesSet).T.tolist()
    check_T = np.vstack(check[0:2]).T.tolist()
    
    if (len(old_dot) != 1 ):
        old_dot2 = setintersec(old_dot, check_T)
        
        if old_dot2 == []:
            old_dot2 = old_dot
        
        old_dot = old_dot2
    
    diffX = -(old_dot[0][0] - tempX)
    diffY = -(old_dot[0][1] - tempY)
    
    firstdot = [tempX+diffX, tempY+diffY]
    
    if firstdot in newdots:
        dot = setdiff(newdots,[firstdot])
        dot.append(firstdot)
    else:
        dot = newdots

    return dot
    
    
    
    
    
    
    
# inputFrame = imageMat(imageWoCircle, boxFramey, boxFramex)

# plt.imshow(imageWoCircle[380:400,370:400])
    
# tempX = tempX
# tempY = tempY
# inputFrame = imageMat(imageWoCircle, boxFramey, boxFramex)
# circleInfo = circleInfo
    
def initialRemoval(tempX, tempY, inputFrame, circleInfo):
    deg = math.degrees(math.atan2((circleInfo[1]-tempY),(tempX-circleInfo[0])))
    
    if deg == 180:
        deg = -180

    inputList = copy.deepcopy(inputFrame)
    inputList = inputList.T.reshape((1,9))[0]
    
    choice = np.zeros((8,3), np.uint8)
    choice[4] = [1, 2, 3]
    choice[5] = [2, 3, 6]
    choice[6] = [3, 6, 9]
    choice[7] = [6, 8, 9]
    choice[0] = [7, 8, 9]
    choice[1] = [4, 7, 8]
    choice[2] = [1, 4, 7]
    choice[3] = [1, 2, 4]
    
    for index in range(1,9):
        if ((deg < (index-4)*45) and (deg >= (index-5)*45)):
            ratio = (deg - (index-5)*45)/45
            if ratio > 0.5:
                if (index == 8):
                    out = choice[0]
                else:
                    out = choice[index]
                
            else:
                out = choice[index-1]
            
            break
        
    for i in range(len(out)): inputList[out[i]-1]=0
        
    matFrame = inputList.reshape((3,3)).T
    
    return matFrame
    



# tempX = 622
# tempY = 370
# c = np.array([[  0,   0,   0],[  0, 255, 255],[255,   0,   0]])
# circleInfo = np.array([372, 414,  35])
# initialRemoval(tempX, tempY, c, circleInfo)



# [538, 393, 1, 4, 506, 393]
# dotsSet = [[505, 395],[504,396]]


# points = [[505, 395]]

# points = TrueDotsSet
# imageWoCircle = initial_image
# parent = parent
# dotsSet = []
# nodesSet = []
# init_flg = 1
# circleInfo = circleInfo




def newNode_Search(points, imageWoCircle, parent, dotsSet, nodesSet, init_flg, circleInfo):
    #This is newNode Search function to handle the always error inside of old
    #node_search function
    #The output will be a matrix with bunch of info
    #1. node.x | 2. node.y | 3. nodeType | 4. parentLevel | 5. parent.x | 6.
    #parent.y
    #circleMask = np.zeros(imageWoCircle.shape, np.uint8)

    numpoints = len(points)
    
    
    for index in range(numpoints):
        #eval(['tempX = points.p' num2str(index) '.x;']);
        #eval(['tempY = points.p' num2str(index) '.y;']);
        
        tempX = points[index][0]
        tempY = points[index][1]
        
        # plt.imshow(imageWoCircle[tempY-5:tempY+6,tempX-5:tempX+6])
        # imageWoCircle[tempY,tempX]=128
        # plt.imshow(imageWoCircle[tempY-5:tempY+6,tempX-5:tempX+6])

        #---------------------Debug code----------------------------
        #         imshow(imageWoCircle)
        #      viscircles([tempX tempY], 0.3);
        # circleMask = cv2.circle(imageWoCircle, (tempX, tempY), 5, (255, 0, 0), -1)
        # plt.imshow(imageWoCircle)
        # cv2.imwrite('circleTest.png',imageWoCircle)
        #-----------------------------------------------------------
        
        #eval(['boxFramex = (points.p' num2str(index) '.x - 1) : (points.p' num2str(index) '.x + 1);']);
        #eval(['boxFramey = (points.p' num2str(index) '.y - 1) : (points.p' num2str(index) '.y + 1);']);
        
        boxFramex = []
        boxFramey = []
        for i in range(tempX-1, tempX+1 +1): boxFramex.append(i)
        for i in range(tempY-1, tempY+1 +1): boxFramey.append(i)        
        
        dotsSet = setunion(dotsSet, [[tempX,tempY]])
        
        
        if init_flg == 1:
            matFrame = initialRemoval(tempX, tempY, imageMat(imageWoCircle, boxFramey, boxFramex), circleInfo)
            
        else:
            matFrame = imageMat(imageWoCircle, boxFramey, boxFramex)
        
        
        [Ys,Xs] = np.where(matFrame > 0)
        exactXs = Xs + tempX - 1
        exactYs = Ys + tempY - 1
        
        exact = np.vstack((exactXs,exactYs)).T.tolist()

        newdots = setdiff(exact, dotsSet)
        
        numNewDots = len(newdots)
        #     type_counter = pointDetection(imageWoCircle(boxFramey, boxFramex));
        #     bifuration = [4, 6, 8];
        #     if (ismember(type_counter, bifuration) && numNewDots ~= 0 && numNewDots ~= 1)
        #         type_defined = numNewDots;
        #     end
        #
        #     if (type_counter == 4 && numNewDots == 1)
        #         type_defined = 1;
        #     end
        #
        #     if (type_counter == 2)
        #         type_defined = numNewDots;
        #     end
        
        dotsSet = setunion(dotsSet, newdots)
    
    
        if numNewDots == 0:
            if (init_flg):
                continue
            
            nodeType = 2
            #--------DEBUG--------------
            #             viscircles([tempX tempY], 1);
            #---------------------------
            
            nodesSet.append([tempX, tempY, nodeType, parent['level'], parent['x'], parent['y']])
            #print("Case 0")
            
        elif numNewDots == 1:
            #newpoints = struct
            newpoints = []

            newpoints.append(newdots[0])
            #newpoints[1].append(newdots[0][1])
            
            #newpoints.p1.x = newdots(1);
            #newpoints.p1.y = newdots(2);
            
            if init_flg == 1:
                nodesSet.append([tempX, tempY, 0, 0, circleInfo[0], circleInfo[1]])
                parent = {'level':parent['level']+1, 'x':tempX, 'y':tempY}
            
            nodesSet = newNode_Search(newpoints, imageWoCircle, parent, dotsSet, nodesSet, 0, circleInfo)
            
            #print("Case 1")
            
        else:
            #print("Case 2")
            nodeType = 1
            newpoints = []

            
            #--------DEBUG--------------
            #             viscircles([tempX tempY], 1);
            #---------------------------
            
            nodesSet.append([tempX, tempY, nodeType, parent['level'], parent['x'], parent['y']])
            
            if numNewDots == 3:
                indexS = middleOfThree(newdots)
                
                for index2 in range(3):
                    #newpoints['x'].append(newdots[indexS[index2]][0])
                    #newpoints['y'].append(newdots[indexS[index2]][1])
                    newpoints.append(newdots[indexS[index2]])
                    #eval(['newpoints.p' num2str(index2) '.x = newdots(indexS(index2),1);']);
                    #eval(['newpoints.p' num2str(index2) '.y = newdots(indexS(index2),2);']);
                
            elif (numNewDots == 2) and (init_flg != 1):
                #newpoints = struct;
                newdots = go_old_way(matFrame, tempX, tempY, setdiff(dotsSet,newdots), newdots, nodesSet)
                
                for index2 in range(2):
                    #newpoints['x'].append(newdots[index2][0])
                    #newpoints['y'].append(newdots[index2][1])
                    newpoints.append(newdots[index2])
                    #eval(['newpoints.p' num2str(index2) '.x = newdots(index2,1);']);
                    #eval(['newpoints.p' num2str(index2) '.y = newdots(index2,2);']);
                
            else:
                for index2 in range(numNewDots):
                    #newpoints['x'].append(newdots[index2][0])
                    #newpoints['y'].append(newdots[index2][1])
                    newpoints.append(newdots[index2])
                    #eval(['newpoints.p' num2str(index2) '.x = newdots(index2,1);']);
                    #eval(['newpoints.p' num2str(index2) '.y = newdots(index2,2);']);
                
            subParent = {'level':parent['level']+1, 'x':tempX, 'y':tempY}

            #subParent.x = tempX;
            #subParent.y = tempY;
            #subParent.level = parent.level + 1;
            
            
            # file = open('ErrorLog.txt', mode='a')
            # file.write('nodesSet: '+str(nodesSet) + 'dotsSet: '+str(dotsSet) + '\n')
            # file.close()
            
            nodesSet = newNode_Search(newpoints, imageWoCircle, subParent, dotsSet, nodesSet, 0, circleInfo)

    
    return nodesSet
    
    
#newNode_Search([[341,390]], imageWoCircle, [parent], dotsSet, nodesSet, init_flg, circleInfo)
#newNode_Search(points, imageWoCircle, parent, dotsSet, nodesSet, init_flg, circleInfo)








# orcellnodes = cellnodes
# maxindex = row
# matrixnodes = new_result

def linkchildren(orcellnodes, matrixnodes):

    # load cellnodestest cellnodes
    
    # if (isequal(orcellnodes, newcellnodes))
    
    maxindex = len(matrixnodes)
    
    for index in range(maxindex):
        tempnodes = matrixnodes[index,:]
        
        find = (matrixnodes[:,5:7] == tempnodes[7:9])[:,0] & (matrixnodes[:,5:7] == tempnodes[7:9])[:,1]
        if True in find:
            childernset = matrixnodes[np.where(find==1)]
            
            #
            #     for t = 1:num
            #         tempset(1,t) = childernset(t, 12);
            #     end
            
            tempset = childernset[:,11].astype(int)
            
            orcellnodes[index, 10] = tempset.tolist()
        
        else:
            orcellnodes[index, 10] = []
    
    out = orcellnodes
    
    return out
    
    
    
    
    
# cellnodes = cellnodes 
# matrixnodes = new_result



def linksibling(cellnodes, matrixnodes):
    #This function is appied to link the sibling index method
    #Only for the parent is the same
    iterationTimes = len(cellnodes)
    
    temp = np.zeros((iterationTimes, 1), object)
    
    for index in range(iterationTimes):
        parent = matrixnodes[index, 5:7]
        level = matrixnodes[index, 4]
                
        find1 = (matrixnodes[:,5:7]==parent)[:,0] & (matrixnodes[:,5:7]==parent)[:,1]
        find2 = (matrixnodes[:,4]==level)
        find = np.where(find1 & find2)[0]

        if True in (find1 & find2):
            siblingset = matrixnodes[find, 11]
            
            difference = set(siblingset) - set([matrixnodes[index, 11]])
            temp[index, 0] = list(difference)
    
    cellnodes[:,12] = temp.T
    
    return cellnodes
    
    









 
# inputcell = cellnodes
# matrix = new_result
# rootinfo = rootinfo
# varargin  = []




def buildTreeStrcut(inputcell, matrix, rootinfo, varargin):
    # Build a clear strcut of the tree
   
    # 0  = 'index'
    # 1  = 'x'
    # 2  = 'y'
    # 3  = 'level'
    # 4  = 'angle'
    # 5  = 'relativeLen'
    # 6  = 'type'
    # 7  = 'childIndex'
    # 8  = 'levelIndex'
    # 9  = 'DistanceSelf2ParentX'
    # 10 = 'DistanceSelf2ParentY'
    # 11 = 'parentIndex'
    # 12 = 'siblingIndex'
    # 13 = 'distRoot'
    # 14 = 'linkTo'
    # 15 = 'consisResult'
    # 16 = 'consistencyScore'
    
    nargin = 3 + len(varargin)
        
    a = len(matrix)
    # Link parent
    newcol = np.zeros((a, 1), object)
    
    node = np.zeros((a, 14), object)
    #node[a,:] = ['index','x','y','level','angle','relativeLen','type','childIndex','levelIndex','DistanceSelf2ParentX','DistanceSelf2ParentY','parentIndex','siblingIndex','distRoot']
    
    for index in range(a):
        find = (matrix[:,7:9] == matrix[index,5:7])[:,0] & (matrix[:,7:9] == matrix[index,5:7])[:,1]
        
        newcol[index,0] = np.where(find)[0] + 1
        
        #Checker
        if (len(newcol[index,0]) > 1):
            print('somthing goes wrong')


    if (nargin == 5):
        angle_scaled_info = varargin[1]
        dist_scaled_info = varargin[0]
        dist_mean = dist_scaled_info[0]
        dist_var = dist_scaled_info[1]
        angle_mean = angle_scaled_info[0]
        angle_var = angle_scaled_info[1]
        
    else:
        angle_vec = inputcell[:,2]
        dist_vec = np.sqrt( ((inputcell[:,0])**2 + (inputcell[:,1])**2).astype(int) )
        angle_mean = np.mean(angle_vec)
        dist_mean = np.mean(dist_vec)
        angle_var = np.var(angle_vec)
        dist_var = np.var(dist_vec)
        dist_scaled_info = [dist_mean, dist_var]
        angle_scaled_info = [angle_mean, angle_var]
    
    
    for i in range(a):
        
        node[i,0] = inputcell[i,11]
        node[i,1] = inputcell[i, 7]
        node[i,2] = inputcell[i, 8]
        node[i,3] = inputcell[i, 4]
        # eval(['n' num2str(index) '.index = input{index,12};' ]);
        # eval(['n' num2str(index) '.x = input{index,8};']);
        # eval(['n' num2str(index) '.y = input{index,9};']);
        # eval(['n' num2str(index) '.level = input{index,5};']);
        
        if len(varargin) > 0:
            node[i,4] = (inputcell[i,2] - angle_mean)/angle_var
            node[i,5] = ( np.sqrt( (inputcell[i,0])**2 + (inputcell[i,1])**2) - dist_mean)/dist_var
            # eval(['n' num2str(index) '.angle = (input{index,3} - angle_mean)/angle_var;']);
            # eval(['n' num2str(index) '.relativeLen = (sqrt((input{index,1})^2 + (input{index,2})^2) - dist_mean)/dist_var;']);
        else:
            node[i,4] = (inputcell[i,2] - 0)/1
            node[i,5] = ( np.sqrt( (inputcell[i,0])**2 + (inputcell[i,1])**2) - 0)/1
            # eval(['n' num2str(index) '.angle = (input{index,3} - 0)/1;']);
            # eval(['n' num2str(index) '.relativeLen = (sqrt((input{index,1})^2 + (input{index,2})^2) - 0)/1;']);
            
        node[i, 6] = inputcell[i,3]
        node[i, 7] = inputcell[i,10]
        node[i, 8] = inputcell[i,9]
        node[i, 9] = inputcell[i,0]
        node[i,10] = inputcell[i,1] 
        node[i,11] = newcol[i,0]
        node[i,12] = inputcell[i,12]
        node[i,13] = ((inputcell[i,7] - rootinfo['x'])**2 + (inputcell[i,8] - rootinfo['y'])**2)**0.5
    #     eval(['n' num2str(index) '.type = input{index,4};']);
    #     eval(['n' num2str(index) '.childIndex = [input{index,11}];']);
    #     eval(['n' num2str(index) '.levelIndex = input{index,10};']);
    #     eval(['n' num2str(index) '.DistanceSelf2ParentX = input{index,1};']);
    #     eval(['n' num2str(index) '.DistanceSelf2ParentY = input{index,2};']);   
    #     eval(['n' num2str(index) '.parentIndex = newcol{index,1};']);
    #     eval(['n' num2str(index) '.siblingIndex = input{index,13};']);
    #     eval(['n' num2str(index) '.distRoot = sqrt((input{index,8} - rootinfo.y)^2 + (input{index,9} - rootinfo.x)^2);']);
    
    # for i in range(a):
    #     Tre
        
    #     eval(['Tree.n' num2str(index) ' = n' num2str(index) ';' ]);
    
    
    out = node
    
    
    return [out, dist_scaled_info, angle_scaled_info]
    












#=====================================DEBUG

# nodeSets = result

def killInitial(nodeSets):
    nodeSets_Real = []
    
    for i in range(len(nodeSets)):
        if nodeSets[i][2]!=0:
            nodeSets_Real.append(nodeSets[i])
        
    return nodeSets_Real



def killRepeat(nodeSets):
    nodeSets = np.array((nodeSets),int)
    
    #checkrepeat = np.column_stack((nodeSets[:,0:2], nodeSets[:,4:]))
    checkrepeat = nodeSets[:,0:2]
    index = nodeSets[:,2:4]
    
    fakelist = []
    
    for i in range(len(nodeSets)):
        repeat = checkrepeat == checkrepeat[i]
        repeat = repeat[:,0] & repeat[:,1] #& repeat[:,2] & repeat[:,3]
        repeatindex = np.where(repeat==1)[0]
        repeatlist = index[repeatindex]

        newlist = repeatlist[repeatlist[:,0] == np.unique(repeatlist[:,0])[0]]

        truelist = newlist[repeatlist[:,1] == np.unique(repeatlist[:,1])[0]]
        trueindex = (repeatlist == truelist)
        trueindex = trueindex[:,0] & trueindex[:,1]
        trueindex = repeatindex[trueindex]
        
        fakeindex = repeatindex[np.where(repeatindex != trueindex)]
        
        if len(trueindex)>1 and len(fakeindex)==0:
            fakelist.append(trueindex[-1])
            
        if len(fakeindex)>0:
            fakelist.append(fakeindex[0])
    
        fakelist = list(set(fakelist))
                
        
    nodeSets_Real = np.delete(nodeSets, fakelist, axis=0).tolist()

    return nodeSets_Real
    
# a = np.array(nodesSet).T
# b = np.vstack((a[0],a[1])).T.tolist()


def drawCircles(image, list_circles):
    draft = copy.deepcopy(image)
    
    for i in range(len(list_circles)):
        circle = cv2.circle(draft, tuple(list_circles[i]), 5, (255, 0, 0), -1)
        
    plt.imshow(draft)
    cv2.imwrite('draft.png',draft)

    return draft


# drawCircles(image, TrueDotsSet)
# drawCircles(image, b)



def detail_10x(image, tempX, tempY):
    #plt.imshow(image[tempY-5:tempY+6,tempX-5:tempX+6])
    image[tempY,tempX]=128
    plt.imshow(image[tempY-5:tempY+6,tempX-5:tempX+6])
    
    return tempX,tempY
        




def detail_5x(image, tempX, tempY):
    #plt.imshow(image[tempY-5:tempY+6,tempX-5:tempX+6])
    image[tempY,tempX]=128
    plt.imshow(image[tempY-30:tempY+31,tempX-30:tempX+31])
    
    return tempX,tempY
        






def killToClose(dotsSet):
    dotsSet_fake = []
    
    for i in range(len(dotsSet)):
        p1 = dotsSet[i]
        
        for j in range(i+1,len(dotsSet)):
            p2 = dotsSet[j]
            dist = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

            if dist <= 3:
                dotsSet[i] = ['fake']
                break
    
    while ['fake'] in dotsSet:
        dotsSet.remove(['fake'])
    
    return dotsSet



# dotsSet = killToClose(dotsSet)


























