'''
Created on 19.05.2019

This file handles the processing of the maxpooled vector in order to see which
ones where the most contributing vector entries. This is then used tu generate
a heatmap similar to the Grad-CAM approach described here: https://arxiv.org/pdf/1610.02391.pdf

@author: Dennis Struhs
'''
import operator
import numpy as np
from open3d import *

def _return_workArr(inputArr):
    return inputArr[0][0]

def list_contrib_vectors(inputArr):
    workArr = _return_workArr(inputArr)
    testArr = set(workArr)
    print(testArr)
    
def count_occurance(inputArr):
    workArr = _return_workArr(inputArr)
    unique, counts = np.unique(workArr, return_counts=True)
    result = dict(zip(unique, counts))
    print(result)
    return result
    
def draw_heatcloud(inpCloud, hitCheckArr):
    pColors = np.zeros((1024,3),dtype=float)
#     maxColVal = hitCheckArr[max(hitCheckArr.items(), key=operator.itemgetter(1))[0]]
    maxColVal = max(hitCheckArr)
    print('maxColVal: %s' % maxColVal)
    for index in range(len(inpCloud[0])):
        try:
            curVal = hitCheckArr[index]
            red = curVal / maxColVal
            green = 1 - (curVal / maxColVal)
            pColors[index] = [red, green, 0]
        except:
            pColors[index] = [0, 0, 0]
    
    pcd = PointCloud()
    pcd.points = Vector3dVector(inpCloud[0])
    pcd.colors = Vector3dVector(pColors)
    draw_geometries([pcd])
    
# arr = np.ndarray(shape=(1,1,6), dtype=int)
# pColors = np.zeros((1024,3),dtype=int)
# pColors[0] = [1, 255, 2]
# print(pColors)