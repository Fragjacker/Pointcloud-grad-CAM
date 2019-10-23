'''
Created on 11.09.2019

This module handles the writing and reading of test data
into a structured binary format.

@author: Dennis Struhs
'''

import struct

def WriteFloat( BinStream , value ):
    """
    This function writes a floating point number as a 4 byte binary number.
    
    @param Binstream: The input file.
    @param value: Writes the input float value as 4 bytes.
    """
    BinStream.write( struct.pack( 'f', value ) )

def ReadFloat( BinStream ):
    """
    Read 4 byte binary and convert it to float.
    
    @param Binstream: The input file.
    @return: The float value of 4 read bytes.
    """
    return struct.unpack( 'f', BinStream.read( 4 ) )[0]

def readTestFile( filePath ):
    """
    This reads a previously stored test data file
    and stores the values into an array of floats.
    
    @param filePath: The absolute path to the file in the system.
    @return: Array of stored floats.
    """
    f = open( filePath, 'rb' )
    curVal = ReadFloat( f )
    resultArr = []
    while True:
        try:
            resultArr.append( curVal )
            curVal = ReadFloat( f )
        except:
            break
    f.close()
    return resultArr

def writeResult( filePath, item ):
    """
    This writes a single float value to file as 4 byte float values.
    
    @param filePath: The absolute path to the file in the system.
    @param resultArray: The array that contains the values to be written.
    @return: Array of stored floats.
    """
    f = open( filePath, 'ab' )
    WriteFloat( f, item )
    f.close()

def writeAllResults( filePath, resultArray ):
    """
    This writes an array of float values to file as 4 byte float values.
    
    @param filePath: The absolute path to the file in the system.
    @param resultArray: The array that contains the values to be written.
    @return: Array of stored floats.
    """
    f = open( filePath, 'wb' )
    for eachEntry in resultArray:
        WriteFloat( f, eachEntry )
    f.close()
