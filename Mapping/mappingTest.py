# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:57:04 2021

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


import sys
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/image_pre_processing')
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/KeyPoints')
sys.path.append('C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Mapping')

import colorextract as cE
import colorextract_Tools as cT

import graph_based_rdGen as gG
import graph_based_rdGen_Tools as gT

import mappingTest_Tools as mT




workdir = "C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Mapping"



# testFile = 'C://Users/MaxGr/Desktop/DH/Dendrite Authentication/Test Sample/b5.png'
# skeleton = cE.colorextract(testFile)

# image = skeleton[0]
# circleInfo = skeleton[1]

# varargin  = []







# #Ref
# plt.imshow(image)

# circleRef = cT.houghCircle_skeleton(image)[0][0]

# image_ref = gG.graph_based_rdGen(image, circleRef, varargin)

# tree_ref   = image_ref[0]
# #new_result = image_ref[1]
# cell_ref   = image_ref[2]
# # distinfo   = image_ref[3]
# # angleinfo  = image_ref[4]




# #Test
# image_180 = np.rot90(image, 2)
# plt.imshow(image_180)

# circleTest = cT.houghCircle_skeleton(image_180)[0][0]

# image_test = gG.graph_based_rdGen(image_180, circleTest, varargin)

# tree_test   = image_test[0]
# # new_result = image_test[1]
# cell_test   = image_test[2]
# # distinfo   = image_test[3]
# # angleinfo  = image_test[4]


#
# ((173-372)**2 + (351-414)**2)**0.5
# ((411-400)**2 + (420-510)**2)**0.5




#mappingTest(tree_test,tree_ref,.5,struct,struct,0,tree_test,tree_ref,.4,.6,[]);

#function [ConsistentMatchTree, matchingRate, ConsistentMatchTree2, iter_mat] = 
#mappingTest_Fast(Tree1,Tree2,param,ConsistentMatchTree,ConsistentMatchTree2,iterTimes,FullTree1,FullTree2,matchingRate,factor,iter_mat,dtmat_big)


# Tree1                = tree_test
# Tree2                = tree_ref
# param                = 0.5
# ConsistentMatchTree1 = []
# ConsistentMatchTree2 = []
# iterTimes            = 0
# FullTree1            = tree_test
# FullTree2            = tree_ref
# matchingRate         = 0.4
# factor               = 0.6
# iter_mat             = []
# dtmat_big            = []



# [CMTree1, Rate, CMTree2, i_mat] = mappingTest_Fast(tree_test, tree_ref, 0.5, [], [], 0, tree_test, tree_ref, 0.4, 0.6, [], [])

# [CMTree1, Rate, CMTree2, i_mat] = mappingTest_Fast(Tree1,Tree2,param,ConsistentMatchTree1,ConsistentMatchTree2,iterTimes,FullTree1,FullTree2,matchingRate,factor,iter_mat,dtmat_big)







# #2
# Tree1                = InconsistentTree1
# Tree2                = InconsistentTree2
# param                = 2*param/3
# ConsistentMatchTree1 = ConsistentMatchTree1
# ConsistentMatchTree2 = ConsistentMatchTree2
# iterTimes            = iterTimes+1
# FullTree1            = FullTree1
# FullTree2            = FullTree2
# matchingRate         = matchingRate
# factor               = factor
# iter_mat             = iter_mat
# dtmat_big            = dtmat_big












def mappingTest_Fast(Tree1,Tree2,param,ConsistentMatchTree1,ConsistentMatchTree2,iterTimes,FullTree1,FullTree2,matchingRate,factor,iter_mat,dtmat_big):
    #This function is made for testing mapping algrothim
    #Tree1 always is the testing tree
    
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
    
    alfa =  0.4
    beta =  0.9
    gamma = 0.9
    consisCriterion1 = (1/3)*(alfa + alfa**2 + beta + gamma)
    consisCriterion2 = (1/6)*(alfa + alfa**2 + beta + gamma)
    
    
    if iterTimes == 0:
        [LinkedTree1, LinkedTree2, dtmat_big] = mT.mappingAndLink_Faster(Tree1,Tree2,param,FullTree1,FullTree2,factor,iterTimes,[])
    else:
        [LinkedTree1, LinkedTree2, temp] = mT.mappingAndLink_Faster(Tree1,Tree2,param,FullTree1,FullTree2,factor,iterTimes, dtmat_big)
    
    
    
    testingNum = len(LinkedTree1) #-1
    testingFields = LinkedTree1[:,0]
    
    dataNum = len(LinkedTree2) #-1
    dataFields = LinkedTree2[:,0]
    
    numOfNodes = min(testingNum,dataNum)
    
    #Initialize the LinkedTree2, and LinkedTree1------------------------------
    #-------------------------------------------------------------------------
    # LinkedTree1[testingNum, 15] = 'consisResult'
    # LinkedTree2[dataNum,    15] = 'consisResult'
    
    # LinkedTree1[testingNum, 16] = 'consistencyScore'
    # LinkedTree2[dataNum,    16] = 'consistencyScore'
    
    
    
    InconsistentTree1 = [] #np.zeros((LinkedTree1.shape),object)
    InconsistentTree2 = [] #np.zeros((LinkedTree2.shape),object)
    
    # ConsistentMatchTree1 = [] #np.zeros((LinkedTree1.shape),object)
    # ConsistentMatchTree2 = [] #np.zeros((LinkedTree2.shape),object)
    
    # for index in range(testingNum):
    #     eval(['LinkedTree1.' testingFields{index} '.consisResult = "";']);
    
    
    # for index = 1:dataNum
    #     eval(['LinkedTree2.' dataFields{index} '.consisResult = "";']);
    
    
    
    for index in range(numOfNodes):
        LinkedNodeIndex1 = LinkedTree1[index, 14]
        # eval(['LinkedNodeIndex1 = LinkedTree1.' testingFields{index} '.linkTo;']);
      
        if (LinkedNodeIndex1 not in LinkedTree2[:,0]):
              LinkedNodeIndex2 = []
        else: 
              LinkedNodeIndex2 = np.where(LinkedTree2[:,0]==LinkedNodeIndex1)[0][0]
      
        
        if (LinkedNodeIndex1==[] or LinkedNodeIndex1==0):
            LinkedTree1[index, 16] = 0
            LinkedTree1[index, 15] = 'inconsistent'
            
            InconsistentTree1.append( LinkedTree1[index] )
            
            # eval(['LinkedTree1.' testingFields{index} '.consistencyScore = 0;']);
            # eval(['LinkedTree1.' testingFields{index} '.consisResult = "inconsistent";']);
            # eval(['InconsistentTree1.' testingFields{index}' = LinkedTree1.' testingFields{index} ';']);
            continue
        
        else:
            #cat two structs into 1 struct
            exMatchedNum1 = len(ConsistentMatchTree1)
            #exMatchedFields1 = ConsistentMatchTree1[:,0]
            CatTree1 = copy.deepcopy(LinkedTree1)
            
            exMatchedNum2 = len(ConsistentMatchTree2)
            #exMatchedFields2 = ConsistentMatchTree2[:,0]
            CatTree2 = copy.deepcopy(LinkedTree2)
            
            if exMatchedNum1 > 0: 
                CatTree1 = np.row_stack((CatTree1, np.array((ConsistentMatchTree1))))
            if exMatchedNum2 > 0: 
                CatTree2 = np.row_stack((CatTree2, np.array((ConsistentMatchTree2))))

            
            # for indextt in range(exMatchedNum1):
            #     CatTree1[indextt] = ConsistentMatchTree1[indextt]
                
                # eval(['CatTree1.' exMatchedFields{indextt}' = ConsistentMatchTree.' exMatchedFields{indextt} ';']);
            
            # for indexdd in range(exMatchedNum2):
            #     CatTree2[indexdd] = ConsistentMatchTree2[indexdd]
                
                # eval(['CatTree2.' exMatchedFields2{indexdd}' = ConsistentMatchTree2.' exMatchedFields2{indexdd} ';']);
            
            #-----------------------------------------------------------------
            if (LinkedNodeIndex1 not in LinkedTree2[:,0]):
                score = 0
            else: 
                score = mT.cosistencyScore(LinkedTree1[index], LinkedTree2[LinkedNodeIndex2], CatTree1, CatTree2, alfa, beta, gamma)


            LinkedTree1[index,            16] = score
            LinkedTree2[LinkedNodeIndex2, 16] = score
            
            # eval(['score = cosistencyScore(LinkedTree1.' testingFields{index}', LinkedTree2.n' num2str(LinkedNodeIndex1)...
            #     ', CatTree1, CatTree2, alfa, beta, gamma);']);
            # eval(['LinkedTree1.' testingFields{index}'.consistencyScore = score;']);
            # eval(['LinkedTree2.n' num2str(LinkedNodeIndex1)'.consistencyScore = score;']);
    
        
        if (score <= consisCriterion1 and score > consisCriterion2):
            LinkedTree1[index,            15] = 'consistent'
            LinkedTree2[LinkedNodeIndex2, 15] = 'consistent'
            
            ConsistentMatchTree1.append( LinkedTree1[index].tolist() )
            ConsistentMatchTree2.append( LinkedTree2[LinkedNodeIndex2].tolist() )
            
            # eval(['LinkedTree1.' testingFields{index}'.consisResult = "consistent";']);
            # eval(['LinkedTree2.n' num2str(LinkedNodeIndex1)'.consisResult = "consistent";']);
            # eval(['ConsistentMatchTree.' testingFields{index}' = LinkedTree1.' testingFields{index} ';']);
            # eval(['ConsistentMatchTree2.n' num2str(LinkedNodeIndex1)' = LinkedTree2.n' num2str(LinkedNodeIndex1) ';']);
            
        else:
            LinkedTree1[index,            15] = 'inconsistent'
            LinkedTree2[LinkedNodeIndex2, 15] = 'inconsistent'
            
            InconsistentTree1.append( LinkedTree1[index] )
            InconsistentTree2.append( LinkedTree2[LinkedNodeIndex2] )
            
            # eval(['LinkedTree1.' testingFields{index}'.consisResult = "inconsistent";']);
            # eval(['LinkedTree2.n' num2str(LinkedNodeIndex1)'.consisResult = "inconsistent";']);
            # eval(['InconsistentTree1.' testingFields{index}'= LinkedTree1.' testingFields{index} ';']);
            # eval(['InconsistentTree2.n' num2str(LinkedNodeIndex1)'= LinkedTree2.n' num2str(LinkedNodeIndex1) ';']);
            

    for indexd in range(dataNum):
        judgestr = LinkedTree2[indexd, 15]
        # eval(['judgestr = LinkedTree2.' dataFields{indexd} '.consisResult;']);
        
        if (judgestr==[] or judgestr==0):
            InconsistentTree2.append( LinkedTree2[indexd] )
            
            # eval(['InconsistentTree2.' dataFields{indexd}'= LinkedTree2.' dataFields{indexd} ';']);
    
               
    InconsistentTree1 = killEmpty(InconsistentTree1)
    InconsistentTree2 = killEmpty(InconsistentTree2)
        
    
    
    #So far,Get two inconsistent tree, and recursion do the remapping by changing
    #param for DISTANCESOCRE function
    #-------------------------------------------------------------------------
    
    inconsisTestingNum = len(InconsistentTree1)
    inconsisDataNum    = len(InconsistentTree2)
    matchedNum         = len(ConsistentMatchTree1)
    ogDataNum          = len(FullTree2)-1
    testDataNum        = len(FullTree1)-1
    
    
    matchingRateDiff = matchedNum/min(testDataNum,ogDataNum) - matchingRate
    # matchingRateDiff = 2*matchedNum/(testDataNum+ogDataNum) - matchingRate;
    # matchingRateDiff = matchedNum/((ogDataNum + testDataNum)/2) - matchingRate;
    print('The last matchingRate Difference is: ' + str(matchingRateDiff*100) )
    
    matchingRate = matchingRate + (0.9**iterTimes)*matchingRateDiff
    
    iter_mat.append( matchingRate )
    
    
    
    # InconsistentMat1 = np.array((InconsistentTree1)) #np.zeros((LinkedTree1.shape),object)
    # InconsistentMat2 = np.array((InconsistentTree2)) #np.zeros((LinkedTree2.shape),object)
    
    # ConsistentMatchMat1 = np.array((ConsistentMatchTree1)) #np.zeros((LinkedTree1.shape),object)
    # ConsistentMatchMat2 = np.array((ConsistentMatchTree2)) #np.zeros((LinkedTree2.shape),object)
    
    
    
    
    if (inconsisTestingNum== 0 or inconsisDataNum==0 or iterTimes==10 or matchingRate>0.99 or matchingRate==1):
        return [ConsistentMatchTree1, matchingRate, ConsistentMatchTree2, iter_mat]
    else:
        [ConsistentMatchTree1,matchingRate,ConsistentMatchTree2,iter_mat] = mappingTest_Fast(InconsistentTree1,InconsistentTree2,2*param/3,ConsistentMatchTree1,ConsistentMatchTree2,iterTimes+1,FullTree1,FullTree2,matchingRate,factor,iter_mat,dtmat_big)             
    
    return [ConsistentMatchTree1, matchingRate, ConsistentMatchTree2, iter_mat]
    










#==================================DEBUG

def killEmpty(list_empty):
    list_true = []
    
    for i in range(len(list_empty)):
        if len(list_empty[i]) > 0:
            list_true.append(list_empty[i])
            
    return list_true
    

# killEmpty(InconsistentTree2)

















