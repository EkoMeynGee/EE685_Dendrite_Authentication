# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 18:08:44 2021

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
import munkres as mk

import skimage.io as skio
import skimage.color as skcolor
import PIL.Image as PImg
import skimage.morphology as skmorph
import skimage.feature as skfeature
import scipy.ndimage as ndi






# tree = FullTree1

def struct2mat(tree):
    # in order to make it program efficient, it is the function to convert the
    # Tree struct to matrix
    
    tree_mat = np.zeros((len(tree), 5), object)
    
    tree_mat[:,0] = tree[:,0]
    tree_mat[:,1] = tree[:,5]
    tree_mat[:,2] = tree[:,4]
    tree_mat[:,3] = tree[:,13]
    tree_mat[:,4] = tree[:,11]
    
    # for i = 1:num
    #     node = eval(['tree.n' num2str(iter) ';']);
    #     pIndex = node.parentIndex;
    #     if (isempty(pIndex))
    #         pIndex = 0;
    #     end
    #     tree_mat(iter,:) = [node.index, node.relativeLen, node.angle, node.distRoot, pIndex];
    # end
        
    
    return tree_mat
    
    
    
    
# index = 0
# index2 = 30
    

# index_tree1 = index
# index_tree2 = index2
# tree_mat1   = tree1_mat
# tree_mat2   = tree2_mat
# param       = param
# factor      = factor


def DistanceScore_Fast(index_tree1, index_tree2, tree_mat1, tree_mat2, param, factor, sub_mat):
    # This function is made to determine the distance similarity scores
    # Two inputs are two node structures.
    # This is a recursive function.
    
    # [0: node.index,
    #  1: node.relativeLen,
    #  2: node.angle,
    #  3: distRoot,
    #  4: node.parentIndex,
    #  5: node.siblingIndex];
    
    relative_relativeLen  = np.abs(tree_mat1[index_tree1, 1] - tree_mat2[index_tree2, 1])
    relative_angle = np.abs(tree_mat1[index_tree1, 2] - tree_mat2[index_tree2, 2])
    relative_distRoot = np.abs(tree_mat1[index_tree1, 3] - tree_mat2[index_tree2, 3])
    
    # 1*relative_relativeLen
    relative_d = np.linalg.norm([1*relative_relativeLen, 1*relative_angle, 2*relative_distRoot])
    
    index_p1 = tree_mat1[index_tree1, 4]
    index_p2 = tree_mat2[index_tree2, 4]
    
    iterTimes = 0
    
    while(sub_mat == 0):
        iterTimes = iterTimes + 1
        if len(index_p1)==0 or len(index_p2) == 0:
            break
        
        index_p1 = index_p1-1
        index_p2 = index_p2-1
        
        sub_dist = np.abs(tree_mat1[index_p1, 1] - tree_mat2[index_p2, 1])
        sub_angle = np.abs(tree_mat1[index_p1, 2] - tree_mat2[index_p2, 2])
        sub_d = np.linalg.norm([np.sqrt(factor)*sub_dist, np.sqrt(1-factor)*sub_angle])
        index_p1 = tree_mat1[index_p1, 4][0]
        index_p2 = tree_mat2[index_p2, 4][0]
        
        relative_d = relative_d + (param**iterTimes) * sub_d
    
    
    return relative_d
    
    
# matrix = Distancematrix

def munkers(matrix):
    x,y = matrix.shape
    mappinglist1 = []
    mappinglist2 = []

    if x > y:
        loop = y
    elif y > x:
        loop = x
    else:
        loop = len(matrix)
        
    for i in range(loop):
        list_i = matrix[i].tolist()
        list_i_min = min(list_i) 
        
        if list_i_min == 999:
            break
        
        min_index = list_i.index(list_i_min)
        

        if min_index in mappinglist2:
            ai = mappinglist2.index(min_index)
            bi = i
            a = matrix[mappinglist1[ai], mappinglist2[ai]]
            b = list_i_min
            
            if b < a:
                mappinglist1[ai] = bi
                mappinglist2[ai] = min_index
                
            else:
                continue
                
        else: 
            mappinglist1.append(i)
            mappinglist2.append(min_index)
        #mappinglist.append([i, min_index])
        
        
    mappinglist = np.array(([mappinglist1,mappinglist2])).T
    
    return mappinglist

# A = munkers(matrix)



# Tree1 = Tree1
# Tree2 = Tree2
# param = param
# FullTree1 = FullTree1
# FullTree2 = FullTree2
# factor = factor
# iterTimes = iterTimes
# Distancematrix_big = dtmat_big


def mappingAndLink_Faster(Tree1, Tree2, param, FullTree1, FullTree2, factor, iterTimes, Distancematrix_big):
    #First part according to tthe Distance Matrix to mapping and set up link
    #relationship with both two Tree structure each node
    
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
    
    LinkedTree1 = copy.deepcopy(Tree1)
    LinkedTree2 = copy.deepcopy(Tree2)
    
    if len(LinkedTree1[0]) < 17:
        LinkedTree1 = np.column_stack((Tree1, np.zeros((len(Tree1),3),int)))
    else:
        LinkedTree1 = np.array((LinkedTree1))
    
    if len(LinkedTree2[0]) < 17:  
        LinkedTree2 = np.column_stack((Tree2, np.zeros((len(Tree2),3),int)))
    else:
        LinkedTree2 = np.array((LinkedTree2))
    
        
    numT1 = len(Tree1) #-1 # Test nodes
    numT2 = len(Tree2) #-1 # Data nodes
    
    # LinkedTree1[numT1, 14] = 'linkTo'
    # LinkedTree2[numT2, 14] = 'linkTo'

    
    testField = LinkedTree1[:,0]
    dataField = LinkedTree2[:,0]
    
    if (numT1 > numT2):
        largesize = numT1
        
    elif (numT1 < numT2):
        largesize = numT2
        
    else :
        largesize = numT1

    
    Distancematrix = np.zeros((largesize,largesize))
    Distancematrix[:] = 999
    
    if (iterTimes == 0):
        #convert tree struct to tree_mat
        tree1_mat = struct2mat(FullTree1)
        tree2_mat = struct2mat(FullTree2)
    
        # [0: node.index,
        #  1: node.relativeLen,
        #  2: node.angle,
        #  3: distRoot,
        #  4: node.parentIndex,
        #  5: node.siblingIndex];
    
        for index in range(numT1):
            for index2 in range(numT2):
                t = 0.6
                D = DistanceScore_Fast(index, index2, tree1_mat, tree2_mat, param, factor, 0)
                
                Distancematrix[index,index2] = t*D + (1-t)*(np.abs(tree1_mat[index, 3] - tree2_mat[index2, 3]))
            
    else:
        tree1_mat = struct2mat(LinkedTree1)
        tree2_mat = struct2mat(LinkedTree2)
    
        for index in range(numT1):
            for index2 in range(numT2):
                t = 0.6
                D = DistanceScore_Fast(index, index2, tree1_mat, tree2_mat, param, factor, 1)
                
                Distancematrix[index,index2] = t*D + (1-t)*(np.abs(tree1_mat[index, 3] - tree2_mat[index2, 3]))
            
 
    
    mappinglist = munkers(Distancematrix)
    
    testsetIndex = mappinglist[:,0]+1
    datasetIndex = mappinglist[:,1]+1
    
    iterationTimes = len(testsetIndex)
    
    #Initialize two tree's linkTo
    # for index in range(numT1):
    #     Tree1[index, 14] = []
        #eval(['Tree1.' testField{index} '.linkTo = [];']);
    
    
    # for index in range(numT2):
    #     Tree2[index, 14] = []
        #eval(['Tree2.' dataField{index} '.linkTo = [];']);
    
    
    #Start linking------------------------------------------------------------
    for index in range(iterationTimes):
        LinkedTree1[testsetIndex[index]-1, 14] = LinkedTree2[datasetIndex[index]-1, 0]
        LinkedTree2[datasetIndex[index]-1, 14] = LinkedTree1[testsetIndex[index]-1, 0]
        #eval(['Tree1.' testField{testsetIndex(index)} '.linkTo = ' strip(dataField{datasetIndex(index)},'n') ';']);
        #eval(['Tree2.' dataField{datasetIndex(index)} '.linkTo = ' strip(testField{testsetIndex(index)}, 'n') ';']);
    
    
    
    return [LinkedTree1, LinkedTree2, Distancematrix]
    
    
    






# node1 = LinkedTree1[index]
# node2 = LinkedTree2[LinkedNodeIndex2]
# LinkedTree1 = CatTree1
# LinkedTree2 = CatTree2
# alfa =  0.4
# beta =  0.9
# gamma = 0.9

# score = 99

def cosistencyScore(node1, node2, LinkedTree1, LinkedTree2, alfa, beta, gamma):
    #Obtain the cosistency of value of tw..................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................o nodes.
    #------------------------------------------------------------------------
    #Important%%%%
    #Precondition: node1 and node2 must be 2 linked nodes.
    #------------------------------------------------------------------------
    #We have Three paramters here, alfa, beta, gamma
    #0 < alfa < 1/2, 0 < beta < 1, 0 < gamma < 1
    
    
    if len(node1[11])==0 and len(node2[11])==0:
        score = alfa + alfa**2
        
    elif (len(node1[11])==0 and len(node2[11])!=0) or (len(node1[11])!=0 and len(node2[11])==0):
        score = 0
        
    else:
        if len(LinkedTree1[LinkedTree1[:,0]==node1[11]]) != 0 and len(LinkedTree2[LinkedTree2[:,0]==node2[11]]) != 0:

            node1parent = LinkedTree1[LinkedTree1[:,0]==node1[11]] [0]
            node2parent = LinkedTree2[LinkedTree2[:,0]==node2[11]] [0]
            # eval(['node1parent = LinkedTree1.n' num2str(node1.parentIndex) ';']);
            # eval(['node2parent = LinkedTree2.n' num2str(node2.parentIndex) ';']);
            
            score = alfa*(node1parent[14] == node2parent[0]) 
            # eval(['score = alfa*(isequal(LinkedTree1.n' num2str(node1.parentIndex)...
            #     '.linkTo, LinkedTree2.n' num2str(node2.parentIndex) '.index));']);
            
        else: 
            node1parent = []
            node2parent = []
            score = 0
            
        
        
        if len(node1parent) != 0 and len(node2parent) != 0:
            
            if (len(node1parent[11])==0 and len(node2parent[11])==0):
                score = score + alfa**2
                
            elif (len(node1parent[11])==0 and len(node2parent[11])!=0) or (len(node1parent[11])!=0 and len(node2parent[11])==0):
                score = score
                
            else:
                if len(LinkedTree1[LinkedTree1[:,0]==node1parent[11]]) != 0 and len(LinkedTree2[LinkedTree2[:,0]==node2parent[11]]) != 0:
                    
                    node1parent_parent = LinkedTree1[LinkedTree1[:,0]==node1parent[11]][0]
                    node2parent_parent = LinkedTree2[LinkedTree2[:,0]==node2parent[11]][0]
                
                    score = score + (alfa**2)*(node1parent_parent[14] == node2parent_parent[0])            
                    # eval(['score = score + (alfa^2)*(isequal(LinkedTree1.n'num2str(node1parent.parentIndex) '.linkTo, LinkedTree2.n'num2str(node2parent.parentIndex) '.index));']);
    
                else:
                    node1parent_parent = 0
                    node2parent_parent = 0
                    score = score
    
    





    
    #childern mapped percentage
    count = 0
    childIndexSet1 = node1[7]
    childIndexSet2 = node2[7]
    nL1 = len(childIndexSet1)
    nL2 = len(childIndexSet2)
    denominator = max(nL1,nL2)
    
    #Since the node1 is one of test node, so start with node1
    for parIndex in range(nL1):
        if len(LinkedTree1[LinkedTree1[:,0]==childIndexSet1[parIndex]]) != 0:
            result = (LinkedTree1[LinkedTree1[:,0]==childIndexSet1[parIndex]][0,14] in childIndexSet2)
        else:
            result = 0
        
        # eval(['result = ismember(LinkedTree1.n' num2str(childIndexSet1(parIndex))...
        #     '.linkTo, childIndexSet2);']);
        
        if (result):
            count = count + 1

    if (denominator == 0):
        score = score + beta
    else:
        score = score + beta*(count/denominator)

    






    #Sibling mapped percentage
    count = 0
    siblingIndexSet1 = node1[12]
    siblingIndexSet2 = node2[12]
    num1 = len(siblingIndexSet1)
    num2 = len(siblingIndexSet2)
    denominator2 = max(num1,num2)
    
    #Since the node1 is one of test node, so start with node1
    for paramIndex in range(num1):
        if len(LinkedTree1[LinkedTree1[:,0]==siblingIndexSet1[paramIndex]]) != 0:
            show = (LinkedTree1[LinkedTree1[:,0]==siblingIndexSet1[paramIndex]][0,14] in siblingIndexSet2)
        else:
            show = False
        
        # eval(['show = ismember(LinkedTree1.n' num2str(siblingIndexSet1(paramIndex))...
        #     '.linkTo, siblingIndexSet2);']);
        
        if (show):
            count = count + 1
    
    if (denominator2 == 0):
        score = score + gamma
    else:
        score = score + gamma*(count/denominator2)
    
    score = 1/3*score
    
    
    return score
    
    
    



# image1 = image_180
# image2 = image

# consistentMatchedTree1 = CMTree1
# consistentMatchedTree2 = CMTree2

# plt.imshow(image1)
# plt.imshow(image2)



def matchedNodeDrawLine(image1, image2, consistentMatchedTree1, consistentMatchedTree2):
    
    #This function is designed for debug and visulize the matching dots
    #between 2 dendrite PUF
        
    [h1,w1] = image1.shape
    [h2,w2] = image2.shape
    
    
    consistentMatchedTree1 = np.array((consistentMatchedTree1))
    consistentMatchedTree2 = np.array((consistentMatchedTree2))
    
    list_dots_1 = copy.deepcopy(consistentMatchedTree1[:,1:3])
    list_dots_2 = copy.deepcopy(consistentMatchedTree2[:,1:3])

    image_append = np.column_stack((image1, image2))
    list_dots_2[:,0] = list_dots_2[:,0] + w1
    
    plt.imshow(image_append)
 
    for i in range(len(list_dots_1)):
        pointA = list_dots_1[i]
        pointB = list_dots_2[i]
        
        cv2.circle(image_append, tuple(pointA), 5, (255, 0, 0), -1)
        cv2.circle(image_append, tuple(pointB), 5, (255, 0, 0), -1)
        
        cv2.line(image_append, tuple(pointA), tuple(pointB), (255, 0, 0), 1, 4)

        
    plt.imshow(image_append)
    cv2.imwrite('image_mapping.png',image_append)

    return image_append





# raw_image1 = skio.imread('b5rotat.png')[:,:,0:3]
# plt.imshow(raw_image1)
    
# raw_image2 = skio.imread('b5.png')[:,:,0:3]
# plt.imshow(raw_image2)


# consistentMatchedTree1 = CMTree1
# consistentMatchedTree2 = CMTree2




def matchedNodeDrawLine_RAW(raw_image1, raw_image2, consistentMatchedTree1, consistentMatchedTree2):
    
    #This function is designed for debug and visulize the matching dots
    #between 2 dendrite PUF
        
    [h1,w1] = raw_image1.shape[0:2]
    [h2,w2] = raw_image2.shape[0:2]
    
    consistentMatchedTree1 = np.array((consistentMatchedTree1))
    consistentMatchedTree2 = np.array((consistentMatchedTree2))
    
    list_dots_1 = copy.deepcopy(consistentMatchedTree1[:,1:3])
    list_dots_2 = copy.deepcopy(consistentMatchedTree2[:,1:3])

    image_append = np.column_stack((raw_image1, raw_image2))
    list_dots_2[:,0] = list_dots_2[:,0] + w1
    
    plt.imshow(image_append)
 
    for i in range(len(list_dots_1)):
        pointA = list_dots_1[i]
        pointB = list_dots_2[i]
        
        cv2.circle(image_append, tuple(pointA), 5, (255, 255, 0), -1)
        cv2.circle(image_append, tuple(pointB), 5, (255, 255, 0), -1)
        
        cv2.line(image_append, tuple(pointA), tuple(pointB), (0, 255, 255), 1, 4)

        
    plt.imshow(image_append)
    cv2.imwrite('image_mapping.png',image_append[:,:,::-1])

    return image_append






def drawDots(image, list_dots):
    draft = copy.deepcopy(image)
    
    for i in range(len(list_dots)):
        cv2.circle(draft, tuple(list_dots[i]), 5, (255, 0, 0), -1)
        
    plt.imshow(draft)
    cv2.imwrite('image_dots.png',draft)

    return draft



# drawDots(image1, list_dots_1)
# drawDots(image2, list_dots_2)

# drawDots(image_append, list_dots_1)
# drawDots(image_append, list_dots_2)

































