'''
Created on 19.05.2019

This file handles the processing of the maxpooled vector in order to see which
ones where the most contributing vector entries. This is then used tu generate
a heatmap similar to the Grad-CAM approach described here: https://arxiv.org/pdf/1610.02391.pdf

@author: Dennis Struhs
'''
import numpy as np
import random
import copy
from open3d import *

def _return_workArr( inputArr ):
    return inputArr[0][0]

def list_contrib_vectors( inputArr ):
    workArr = _return_workArr( inputArr )
    testArr = set( workArr )
    print( testArr )

def count_occurance( inputArr ):
    workArr = _return_workArr( inputArr )
    unique, counts = np.unique( workArr, return_counts = True )
    result = dict( zip( unique, counts ) )
    print( result )
    return result

def get_average( inputArr ):
    valSum = 0
    count = 0
    for index in range( len( inputArr ) ):
        curVal = inputArr[index]
        if curVal > 0:
            valSum += curVal
            count += 1
    return ( valSum / count )

def get_median( inputArr ):
    locArr = copy.deepcopy( inputArr )
    locArr = locArr[locArr > 0]
    locArr.sort()
    median = locArr[int( len( locArr ) / 2 )]
    return median

def get_midrange( inputArr ):
    locArr = copy.deepcopy( inputArr )
    locArr = locArr[locArr > 0]
    minVal = min( locArr )
    maxVal = max( locArr )
    result = ( minVal + maxVal ) / 2
    return result

def delete_top_n_points( inputArr, numPoints ):
    locArr = copy.deepcopy( inputArr )
    locArr.sort()
    locArr.reverse()
    for _ in range( numPoints ):
        np.delete( locArr, [0] )
    return locArr

def delete_all_nonzeros( inputheatMap, inputArr ):
    locArr = copy.deepcopy( inputArr )
    resArr = []
    candArr = []
    count = 0
    for index, eachItem in enumerate( inputheatMap ):
        if eachItem > 0:
            candArr.append( index )
            resArr.append([locArr[0][index],eachItem])
            count += 1

    if len( candArr ) > locArr.shape[1] or 10 > locArr.shape[1]:
        return locArr, resArr, 0

    locArr = np.delete( locArr, candArr, 1 )
    return locArr, resArr, count

def delete_all_zeros( inputheatMap, inputArr ):
    locArr = copy.deepcopy( inputArr )
    resArr = []
    candArr = []
    count = 0
    for index, eachItem in enumerate( inputheatMap ):
        if eachItem == 0:
            candArr.append( index )
            resArr.append([locArr[0][index],eachItem])
            count += 1
    locArr = np.delete( locArr, candArr, 1 )
    return locArr, resArr, count

def delete_randon_points( inputArr, numPoints ):
    locArr = copy.deepcopy( inputArr )
    randomArr = random.sample( range( inputArr.shape[1] ), numPoints )
    locArr = np.delete( locArr, randomArr, 1 )
    return locArr, randomArr

def delete_above_threshold( inputheatMap, inputArr, mode ):
    locArr = copy.deepcopy( inputArr )
    resArr = []
    candArr = []
    threshold = None
    count = 0
    if mode == "+average":
        threshold = get_average( inputheatMap )
    elif mode == "+median":
        threshold = get_median( inputheatMap )
    elif mode == "+midrange":
        threshold = get_midrange( inputheatMap )

    for index, eachItem in enumerate( inputheatMap ):
        if eachItem > threshold:
            candArr.append( index )
            resArr.append([locArr[0][index],eachItem])
            count += 1
    locArr = np.delete( locArr, candArr, axis = 1 )

    return locArr, resArr, count

def delete_below_threshold( inputheatMap, inputArr, mode ):
    locArr = copy.deepcopy( inputArr )
    candArr = []
    resArr = []
    threshold = None
    count = 0

    if mode == "-average":
        threshold = get_average( inputheatMap )
    elif mode == "-median":
        threshold = get_median( inputheatMap )
    elif mode == "-midrange":
        threshold = get_midrange( inputheatMap )

    for index, eachItem in enumerate( inputheatMap ):
        if eachItem < threshold:
            candArr.append( index )
            resArr.append([locArr[0][index],eachItem])
            count += 1

    if len( candArr ) > locArr.shape[1] or 10 > locArr.shape[1]:
        print( "SIZE IS TOO SMALL!!! RETURNING UNCHANGED ARRAY!" )
        return locArr, resArr, 0

    locArr = np.delete( locArr, candArr, 1 )

    return locArr, resArr, count

def truncate_to_threshold( inputArr, mode ):
    newArr = []
    counter = 0

    if mode == "+average" or mode == "-average":
        threshold = get_average( inputArr )
    elif mode == "+median" or mode == "-median":
        threshold = get_median( inputArr )
    elif mode == "+midrange" or mode == "-midrange":
        threshold = get_midrange( inputArr )

    for index in range( len( inputArr ) ):
        curVal = inputArr[index]
        if curVal > threshold:
            newArr.append( threshold )
            counter += 1
        else:
            newArr.append( inputArr[index] )
    print( "BEYOND THRESHOLD VALUES: ", counter )
    return newArr

def draw_heatcloud( inpCloud, hitCheckArr, mode ):
    hitCheckArr = truncate_to_threshold( hitCheckArr, mode )
    pColors = np.zeros( ( len( hitCheckArr ), 3 ), dtype = float )
    maxColVal = max( hitCheckArr )
    for index in range( len( inpCloud[0] ) ):
        try:
            curVal = hitCheckArr[index]
            if curVal == 0:
                pColors[index] = [0, 0, 0]
            else:
                red = curVal / maxColVal
                green = 1 - ( curVal / maxColVal )
                pColors[index] = [red, green, 0]
        except:
            print( "INVALID VALUE FOR INDEX: ", index )
            pColors[index] = [0, 0, 0]

    pcd = PointCloud()
    pcd.points = Vector3dVector( inpCloud[0] )
    pcd.colors = Vector3dVector( pColors )
    draw_geometries( [pcd] )

def draw_pointcloud( inputPointCloudArr ):
    pcd = PointCloud()
    pcd.points = Vector3dVector( inputPointCloudArr[0] )
    pcd.colors = Vector3dVector( np.zeros( ( len( inputPointCloudArr[0] ), 3 ), dtype = float ) )
    draw_geometries( [pcd] )

# arr = np.ndarray(shape=(1,1,6), dtype=int)
# pColors = np.zeros((1024,3),dtype=int)
# pColors[0] = [1, 255, 2]
# print(pColors)
