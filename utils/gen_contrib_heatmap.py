'''
Created on 19.05.2019

This file handles the processing of the maxpooled vector in order to see which
ones where the most contributing vector entries. This is then used tu generate
a heatmap similar to the Grad-CAM approach described here: https://arxiv.org/pdf/1610.02391.pdf

@author: Dennis Struhs
'''
import numpy as np
import random
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

def get_average(inputArr):
    valSum = 0
    count = 0
    for index in range(len(inputArr)):
        curVal = inputArr[index]
        if curVal > 0:
            valSum += curVal
            count += 1
    return (valSum / count)

def get_median(inputArr):
    locArr = inputArr.copy()
    locArr = locArr[locArr > 0]
    locArr.sort()
    median = locArr[int(len(locArr) / 2)]
    return median

def get_midrange(inputArr):
    locArr = inputArr.copy()
    locArr = locArr[locArr > 0]
    minVal = min(locArr)
    maxVal = max(locArr)
    result = (minVal + maxVal) / 2
    return result

def delete_top_n_points(inputArr, numPoints):
    locArr = inputArr.copy()
    locArr.sort()
    locArr.reverse()
    for _ in range(numPoints):
        np.delete(locArr, [0])
    return locArr

def delete_all_nonzeros(inputheatMap, inputArr):
    locArr = inputArr.copy()
    candArr = []
    count = 0
    for index, eachItem in enumerate(inputheatMap):
        if eachItem > 0:
            candArr.append(index)
            count += 1
    locArr = np.delete(locArr, candArr, 1)
    return locArr, count

def delete_all_zeros(inputheatMap, inputArr):
    locArr = inputArr.copy()
    candArr = []
    count = 0 
    for index, eachItem in enumerate(inputheatMap):
        if eachItem == 0:
            candArr.append(index)
            count += 1
    locArr = np.delete(locArr, candArr, 1)
    return locArr, count

def delete_all_above_average(inputheatMap, inputArr):
    locArr = inputArr.copy()
    candArr = []
    count = 0
    avg = get_average(inputheatMap)
    for index, eachItem in enumerate(inputheatMap):
        if eachItem > avg:
            candArr.append(index)
            count += 1
    locArr = np.delete(locArr, candArr, 1)
    return locArr, count

def delete_randon_points(numPoints, inputArr):
    locArr = inputArr.copy()
    randomArr = random.sample(range(inputArr.shape[1]-1), numPoints)
    locArr = np.delete(locArr, randomArr, 1)
    return locArr

def delete_above_threshold(inputheatMap, inputArr, mode):
    locArr = inputArr.copy()
    candArr = []
    threshold = None
    count = 0
    if mode == "average":
        threshold = get_average(inputheatMap)
    elif mode == "median":
        threshold = get_median(inputheatMap)
    elif mode =="midrange":
        threshold = get_midrange(inputheatMap)
        
    for index, eachItem in enumerate(inputheatMap):
        if eachItem > threshold:
            candArr.append(index)
            count += 1
    locArr = np.delete(locArr, candArr, 1)
    
    return locArr, count

def truncate_to_threshold(inputArr, threshold):
    newArr = []
    counter = 0
    for index in range(len(inputArr)):
        curVal = inputArr[index]
        if curVal > threshold:
            newArr.append(threshold)
            counter += 1
        else:
            newArr.append(inputArr[index])
    print("BEYOND THRESHOLD VALUES: ", counter)
    return newArr
    
def draw_heatcloud(inpCloud, hitCheckArr):
    pColors = np.zeros((len(hitCheckArr),3),dtype=float)
    maxColVal = max(hitCheckArr)
#     print('maxColVal: %s' % maxColVal)
    for index in range(len(inpCloud[0])):
        try:
            curVal = hitCheckArr[index]
            if curVal == 0:
                pColors[index] = [0, 0, 0]
            else:
                red = curVal / maxColVal
                green = 1 - (curVal / maxColVal)
                pColors[index] = [red, green, 0]
        except:
            print("INVALID VALUE FOR INDEX: ", index)
            pColors[index] = [0, 0, 0]
    
    pcd = PointCloud()
    pcd.points = Vector3dVector(inpCloud[0])
    pcd.colors = Vector3dVector(pColors)
    draw_geometries([pcd])
    
def draw_pointcloud(inputPointCloudArr):
    pcd = PointCloud()
    pcd.points = Vector3dVector(inputPointCloudArr[0])
    pcd.colors = Vector3dVector(np.zeros((len(inputPointCloudArr[0]),3),dtype=float))
    draw_geometries([pcd])
    
# arr = np.ndarray(shape=(1,1,6), dtype=int)
# pColors = np.zeros((1024,3),dtype=int)
# pColors[0] = [1, 255, 2]
# print(pColors)