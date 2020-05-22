def getShapeName( index ):
    '''
    This function returns the label index as a string,
    using the shape_names.txt file for reference.
    '''
    shapeTXT = open( 'data\\modelnet40_ply_hdf5_2048\\shape_names.txt', 'r' )
    entry = ''
    for _ in range( index + 1 ):
        entry = shapeTXT.readline().rstrip()
    shapeTXT.close()
    return entry

def findCorrectLabel( inputLabelArray, desiredClassLabel ):
    '''
    This function retrieves the correct shape from the
    test batch that matches the target feature vector.
    '''
    result = None
    compLabel = getShapeName( desiredClassLabel )
    for currentIndex in range( len( inputLabelArray ) ):
        curLabel = getShapeName( inputLabelArray[currentIndex:currentIndex + 1][0] )
        if curLabel.lower() == compLabel.lower():
            result = currentIndex
            break
    return result

def getPrediction( predIndex ):
    return getShapeName( predIndex[0] )
