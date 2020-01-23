'''
Created on 07.12.2019

@author: Dennis Struhs
'''

import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import copy
import os
import sys
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
sys.path.append( BASE_DIR )
sys.path.append( os.path.join( BASE_DIR, 'models' ) )
sys.path.append( os.path.join( BASE_DIR, 'utils' ) )
import provider
import tf_util
from matplotlib import pyplot as plt
import gen_contrib_heatmap as gch
import test_data_handler as tdh

#===============================================================================
# Global variables to control program behavior
#===============================================================================
usePreviousSession = True    # --Set this to true to use a previously trained model.
performTraining = False    # --Set this to true to train the model. Set to false to only test the pretrained model.
desiredLabel = 1    # --The index of the class label the object should be tested against.
numTestRuns = 1    # --Amount of tests for the current test label object.
maxNumPoints = 2048    # --How many points should be considered? [256/512/1024/2048] [default: 1024]

#===============================================================================
# Help Functions
#===============================================================================

'''
This function computes a the necessary weight gradient for the point clouds
in order to color the points according to their assigned weights.

:param class_activation_vector: The class activation vector (Is of dimension [<batch_size>,40]).
:param feature_vector: The feature vector before the max pooling operation.
:param index: The desired class that the heatmap should be computed for
'''
def computeHeatGradient( class_activation_vector, feature_vector, classIndex, mode ):
    # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
    class_activation_vector = tf.tensordot( class_activation_vector, tf.one_hot( indices = classIndex, depth = 40 ), axes = [[1], [0]] )

    # Compute gradient of the class prediction vector w.r.t. the feature vector. Use class_activation_vector[classIndex] to set which class shall be probed.
    maxgradients = tf.gradients( ys = class_activation_vector, xs = feature_vector )
    maxgradients = tf.squeeze( maxgradients, axis = [0, 1, 3] )

    # Average pooling of the weights over all batches
    if mode == "avgpooling":
        maxgradients = tf.reduce_mean( maxgradients, axis = 1 )    # Average pooling
    elif mode == "maxpooling":
        maxgradients = tf.reduce_max( maxgradients, axis = 1 )    # Max pooling
#     maxgradients = tf.squeeze(tf.map_fn(lambda x: x[100:101], maxgradients)) # Stride pooling

    # Multiply with original pre maxpool feature vector to get weights
    feature_vector = tf.squeeze( feature_vector, axis = [0, 2] )    # Remove empty dimensions of the feature vector so we get [batch_size,1024]
    multiply = tf.constant( feature_vector[1].get_shape().as_list() )    # Feature vector matrix
    multMatrix = tf.reshape( tf.tile( maxgradients, multiply ), [ multiply[0], maxgradients.get_shape().as_list()[0]] )    # Reshape [batch_size,] to [1024, batch_size] by copying the row n times
    maxgradients = tf.matmul( feature_vector, multMatrix )    # Multiply [batch_size, 1024] x [1024, batch_size]
    maxgradients = tf.diag_part( maxgradients )    # Due to Matmul the interesting values are on the diagonal part of the matrix.

    # ReLU out the negative values
    maxgradients = tf.maximum( maxgradients, 0 )

    return maxgradients

def getOps( pointclouds_pl, labels_pl, is_training_pl, batch, pred, maxpool_out, feature_vec, loss, train_op, merged, maxgradients ):
    if maxgradients is None:
        ops = {'pointclouds_pl':pointclouds_pl,
            'labels_pl':labels_pl,
            'is_training_pl':is_training_pl,
            'pred':pred,
            'loss':loss,
            'train_op':train_op,
            'merged':merged,
            'step':batch,
            'maxpool_out':maxpool_out,
            'feature_vec':feature_vec}
    else:
        ops = {'pointclouds_pl':pointclouds_pl,
            'labels_pl':labels_pl,
            'is_training_pl':is_training_pl,
            'pred':pred,
            'loss':loss,
            'train_op':train_op,
            'merged':merged,
            'step':batch,
            'maxpool_out':maxpool_out,
            'feature_vec':feature_vec,
            'maxgradients':maxgradients}
    return ops

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

def findCorrectLabel( inputLabelArray ):
    '''
    This function retrieves the correct shape from the
    test batch that matches the target feature vector.
    '''
    result = None
    compLabel = getShapeName( testLabel )
    for currentIndex in range( len( inputLabelArray ) ):
        curLabel = getShapeName( inputLabelArray[currentIndex:currentIndex + 1][0] )
        if curLabel.lower() == compLabel.lower():
            result = currentIndex
            break
    return result

def getPrediction( predIndex ):
    return getShapeName( predIndex[0] )

def storeTestResults( mode, total_correct, total_seen, loss_sum, pred_val ):
    '''
    This function stores the test data into seperate files for later retrieval.
    '''
    curShape = getShapeName( testLabel )
    savePath = os.path.join( os.path.split( __file__ )[0], "testdata" , curShape )
    if not os.path.exists( savePath ):
        os.makedirs( savePath )
    mean_loss = loss_sum / float( total_seen )
    accuracy = total_correct / float( total_seen )
    filePath = os.path.join( savePath, curShape + "_" + str( numTestRuns ) + "_" + str( mode ) + "_meanloss" )
    print( "STORING FILES TO: ", filePath )
    tdh.writeResult( filePath, mean_loss )
    filePath = os.path.join( savePath, curShape + "_" + str( numTestRuns ) + "_" + str( mode ) + "_accuracy" )
    print( "STORING FILES TO: ", filePath )
    tdh.writeResult( filePath, accuracy )
    filePath = os.path.join( savePath, curShape + "_" + str( numTestRuns ) + "_" + str( mode ) + "_prediction" )
    print( "STORING FILES TO: ", filePath )
    tdh.writeResult( filePath, pred_val )
    log_string( 'eval mean loss: %f' % mean_loss )
    log_string( 'eval accuracy: %f' % accuracy )

testLabel = desiredLabel - 1    # -- Subtract 1 to make the label match Python array enumeration, which starts from 0.
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument( '--gpu', type = int, default = 0, help = 'GPU to use [default: GPU 0]' )
parser.add_argument( '--model', default = 'pointnet_cls', help = 'Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]' )
parser.add_argument( '--log_dir', default = 'log', help = 'Log dir [default: log]' )
parser.add_argument( '--num_point', type = int, default = maxNumPoints, help = 'Point Number [256/512/1024/2048] [default: 1024]' )
parser.add_argument( '--max_epoch', type = int, default = 250, help = 'Epoch to run [default: 250]' )
parser.add_argument( '--batch_size', type = int, default = 1, help = 'Batch Size during evaluation [default: 1]' )
parser.add_argument( '--learning_rate', type = float, default = 0.001, help = 'Initial learning rate [default: 0.001]' )
parser.add_argument( '--momentum', type = float, default = 0.9, help = 'Initial learning rate [default: 0.9]' )
parser.add_argument( '--optimizer', default = 'adam', help = 'adam or momentum [default: adam]' )
parser.add_argument( '--decay_step', type = int, default = 200000, help = 'Decay step for lr decay [default: 200000]' )
parser.add_argument( '--decay_rate', type = float, default = 0.7, help = 'Decay rate for lr decay [default: 0.8]' )
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module( FLAGS.model )    # import network module
MODEL_FILE = os.path.join( BASE_DIR, 'models', FLAGS.model + '.py' )
LOG_DIR = FLAGS.log_dir
if not os.path.exists( LOG_DIR ): os.mkdir( LOG_DIR )
if sys.platform == 'linux' or sys.platform == 'linux2':
    os.system( 'cp %s %s' % ( MODEL_FILE, LOG_DIR ) )    # bkp of model def
    os.system( 'cp train.py %s' % ( LOG_DIR ) )    # bkp of train procedure
if sys.platform == 'win32':
    import shutil
    shutil.copy( MODEL_FILE, LOG_DIR )
    shutil.copy( 'train.py', LOG_DIR )
LOG_FOUT = open( os.path.join( LOG_DIR, 'log_train.txt' ), 'w' )
LOG_FOUT.write( str( FLAGS ) + '\n' )

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float( DECAY_STEP )
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join( BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt' ) )
TEST_FILES = provider.getDataFiles( \
    os.path.join( BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt' ) )

def log_string( out_str ):
    LOG_FOUT.write( out_str + '\n' )
    LOG_FOUT.flush()
    print( out_str )

def get_learning_rate( batch ):
    learning_rate = tf.train.exponential_decay( 
                        BASE_LEARNING_RATE,    # Base learning rate.
                        batch * BATCH_SIZE,    # Current index into the dataset.
                        DECAY_STEP,    # Decay step.
                        DECAY_RATE,    # Decay rate.
                        staircase = True )
    learning_rate = tf.maximum( learning_rate, 0.00001 )    # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay( batch ):
    bn_momentum = tf.train.exponential_decay( 
                      BN_INIT_DECAY,
                      batch * BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase = True )
    bn_decay = tf.minimum( BN_DECAY_CLIP, 1 - bn_momentum )
    return bn_decay

class AdversialPointCloud():
    def __init__( self, numPoints = BATCH_SIZE ):
        self.is_training = False
        self.count = np.zeros( ( NUM_CLASSES, ), dtype = bool )
        self.all_counters = np.zeros( ( NUM_CLASSES, 3 ), dtype = int )

        # The number of points is not specified
        self.pointclouds_pl, self.labels_pl = MODEL.placeholder_inputs( numPoints, None )
        self.is_training_pl = tf.placeholder( tf.bool, shape = () )

        # simple model
        self.pred, self.end_points, _, self.feature_vec = MODEL.get_model( self.pointclouds_pl, self.is_training_pl )
        self.classify_loss = MODEL.get_loss( self.pred, self.labels_pl, self.end_points )

    def getGradient( self, sess, poolingMode, class_activation_vector, feed_dict ):
        # Compute gradient of the class prediction vector w.r.t. the feature vector. Use class_activation_vector[classIndex] to set which class shall be probed.
        maxgradients = sess.run( tf.gradients( ys = class_activation_vector, xs = self.feature_vec )[0], feed_dict = feed_dict )
        maxgradients = tf.squeeze( maxgradients, axis = [0, 2] )
        # Average pooling of the weights over all batches
        if poolingMode == "avgpooling":
            maxgradients = tf.reduce_mean( maxgradients, axis = 1 )    # Average pooling
        elif poolingMode == "maxpooling":
            maxgradients = tf.reduce_max( maxgradients, axis = 1 )    # Max pooling
    #             maxgradients = tf.squeeze(tf.map_fn(lambda x: x[100:101], maxgradients)) # Stride pooling
        # Multiply with original pre maxpool feature vector to get weights
        feature_vector = tf.squeeze( self.feature_vec, axis = [0, 2] )    # Remove empty dimensions of the feature vector so we get [batch_size,1024]
        multiply = tf.constant( feature_vector[1].get_shape().as_list() )    # Feature vector matrix
        multMatrix = tf.reshape( tf.tile( maxgradients, multiply ), [multiply[0], maxgradients.get_shape().as_list()[0]] )    # Reshape [batch_size,] to [1024, batch_size] by copying the row n times
        maxgradients = tf.matmul( feature_vector, multMatrix )    # Multiply [batch_size, 1024] x [1024, batch_size]
        maxgradients = tf.diag_part( maxgradients )    # Due to Matmul the interesting values are on the diagonal part of the matrix.
        # ReLU out the negative values
        maxgradients = tf.maximum( maxgradients, 0 )
        return maxgradients

    def drop_points( self, pointclouds_pl, labels_pl, sess, poolingMode, thresholdMode, numDeletePoints = 0 ):
        pcTempResult = pointclouds_pl.copy()
        classIndex = testLabel
        delCount = 0

        # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
#         class_activation_vector = tf.tensordot( self.pred, tf.one_hot( indices = classIndex, depth = 40 ), axes = [[1], [0]] )
        class_activation_vector = tf.multiply( self.pred, tf.one_hot( indices = classIndex, depth = 40 ) )

        for i in range( 100 ):
            print( "ITERATION: ", i )
            # Setup feed dict for current iteration
            feed_dict = {self.pointclouds_pl: pcTempResult,
                     self.labels_pl: labels_pl,
                     self.is_training_pl: self.is_training}

            maxgradients = self.getGradient( sess, poolingMode, class_activation_vector, feed_dict )

            ops = {'pred':self.pred,
                   'loss':self.classify_loss,
                   'maxgradients':maxgradients}
            # Drop points now
            pred_value, loss_value, heatGradient = sess.run( [ops['pred'], ops['loss'], ops['maxgradients']] , feed_dict = feed_dict )
            pred_value = np.argmax( pred_value, 1 )

            # Perform visual stuff here
            if thresholdMode == "+average" or thresholdMode == "+median" or thresholdMode == "+midrange":
                resultPCloudThresh, heatGradient, Count = gch.delete_above_threshold( heatGradient, pcTempResult, thresholdMode )
            if thresholdMode == "-average" or thresholdMode == "-median" or thresholdMode == "-midrange":
                resultPCloudThresh, heatGradient, Count = gch.delete_below_threshold( heatGradient, pcTempResult, thresholdMode )
            if thresholdMode == "nonzero":
                resultPCloudThresh, heatGradient, Count = gch.delete_all_nonzeros( heatGradient, pcTempResult )
            if thresholdMode == "zero":
                resultPCloudThresh, heatGradient, Count = gch.delete_all_zeros( heatGradient, pcTempResult )
            if thresholdMode == "random":
                resultPCloudThresh, heatGradient, Count = gch.delete_random_points( pcTempResult, numDeletePoints )

            delCount += Count
            print( "GROUND TRUTH: ", getShapeName( classIndex ) )
            print( "PREDICTION: ", getPrediction( pred_value ) )
            print( "LOSS: ", loss_value )
            print( "DELETED POINTS: ", delCount )
            pcTempResult = copy.deepcopy( resultPCloudThresh )

#             truncGrad = gch.truncate_to_threshold(heatGradient, 'median')
#             plt.plot(np.arange(len(heatGradient)), heatGradient, 'C0', label='Original gradient')
#             plt.plot(np.arange(len(heatGradient)), heatGradient, 'C0o', alpha=0.3)
#             plt.plot(np.arange(len(heatGradient)), truncGrad, 'C1', label='Truncated gradient')
#             plt.plot(np.arange(len(heatGradient)), truncGrad, 'C1o', alpha=0.3)
# #             plt.axhline(y=average, color='r', linestyle='-', label='Average (Zeros ignored)')
#             plt.axhline(y=gch.get_median(heatGradient), color='b', linestyle='-', label='Median (Zeros ignored)')
#             plt.legend(title='Gradient value plot:')
#             plt.show()
#         gch.draw_heatcloud(pcTempResult, heatGradient, thresholdMode)
        return pcTempResult, delCount

    def drop_and_store_results( self, pointclouds_pl, labels_pl, sess, poolingMode, thresholdMode, numDeletePoints = None ):
        # Some profiling
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        
        pcTempResult = pointclouds_pl.copy()
        classIndex = testLabel
        delCount = []
        vipPcPointsArr = []
        weightArray = []
        i = 0
        
        # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
        class_activation_vector = tf.multiply( self.pred, tf.one_hot( indices = classIndex, depth = 40 ) )

        while True:
            i += 1
            print( "ITERATION: ", i )
            # Setup feed dict for current iteration
            feed_dict = {self.pointclouds_pl: pcTempResult,
                         self.labels_pl: labels_pl,
                         self.is_training_pl: self.is_training}

            maxgradients = self.getGradient( sess, poolingMode, class_activation_vector, feed_dict )

            ops = {'pred':self.pred,
                   'loss':self.classify_loss,
                   'maxgradients':maxgradients}
            
            #===================================================================
            # Evaluate over n batches now to get the accuracy for this iteration.
            #===================================================================
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            pcEvalTest = copy.deepcopy( pcTempResult )
            for _ in range( numTestRuns ):
                pcEvalTest = provider.rotate_point_cloud_XYZ( pcEvalTest )
                feed_dict2 = {self.pointclouds_pl: pcEvalTest,
                              self.labels_pl: labels_pl,
                              self.is_training_pl: self.is_training}
                eval_prediction, eval_loss, heatGradient = sess.run( [ops['pred'], ops['loss'], ops['maxgradients']],
                                          feed_dict = feed_dict2 )
                eval_prediction = np.argmax( eval_prediction, 1 )
                correct = np.sum( eval_prediction == labels_pl )
                total_correct += correct
                total_seen += 1
                loss_sum += eval_loss * BATCH_SIZE
                
            print( "GROUND TRUTH: ", getShapeName( classIndex ) )
            print( "PREDICTION: ", getPrediction( eval_prediction ) )
            print( "LOSS: ", eval_loss )
            print( "ACCURACY: ", (total_correct / total_seen) )
            accuracy = total_correct / float( total_seen )
            
            # Stop iterating when the eval_prediction deviates from ground truth
            if classIndex != eval_prediction and accuracy <= 0.5:
                print("GROUND TRUTH DEVIATED FROM PREDICTION AFTER %s ITERATIONS" % i)
                break
            
            # Perform visual stuff here
#             resultPCloudThresh, vipPointsArr, Count = gch.delete_max_point( heatGradient, pcTempResult )

            if thresholdMode == "+average" or thresholdMode == "+median" or thresholdMode == "+midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_above_threshold( heatGradient, pcTempResult, thresholdMode )
            if thresholdMode == "-average" or thresholdMode == "-median" or thresholdMode == "-midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_below_threshold( heatGradient, pcTempResult, thresholdMode )
            if thresholdMode == "nonzero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_nonzeros( heatGradient, pcTempResult )
            if thresholdMode == "zero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_zeros( heatGradient, pcTempResult )
            if thresholdMode == "+random" or thresholdMode == "-random":
                resultPCloudThresh, vipPointsArr = gch.delete_random_points( heatGradient, pcTempResult, numDeletePoints[i] )
#                 Count = numDeletePoints[i]
            print("REMOVING %s POINTS." % Count)

            delCount.append( Count )
            vipPcPointsArr.extend( vipPointsArr[0] )
            weightArray.extend( vipPointsArr[1] )
            pcTempResult = copy.deepcopy( resultPCloudThresh )
            
#             testSetName = "XYZ_" + poolingMode + "_" + thresholdMode
#             storeTestResults( testSetName, total_correct, total_seen, loss_sum, eval_prediction )

        pr.disable()
        pr.print_stats()
        totalRemoved = sum(delCount)
        print("TOTAL REMOVED POINTS: ", totalRemoved)
        print("TOTAL REMAINING POINTS: ", maxNumPoints - totalRemoved)
        gch.draw_pointcloud(pcTempResult) #-- Residual point cloud
#         gch.draw_NewHeatcloud(vipPcPointsArr, weightArray) #-- Important points only
        vipPcPointsArr.extend(pcTempResult[0])
        gch.draw_NewHeatcloud(vipPcPointsArr, weightArray) #--All points combined
        return delCount

def evaluate():
    global testLabel
    is_training = False
    adversarial_attack = AdversialPointCloud()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session( config = config )

    # Init variables
    init = tf.global_variables_initializer()
    # To fix the bug introduced in TF 0.12.1 as in
    # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
    # sess.run(init)
    sess.run( init, {adversarial_attack.is_training_pl: False} )

    # Restore variables from disk.
    trained_model = os.path.join( LOG_DIR, "model.ckpt" )
    saver.restore( sess, trained_model )
    log_string( "Model restored." )

    # # ops built on attributes defined in adversarial_attack
    ops = {'pointclouds_pl': adversarial_attack.pointclouds_pl,
           'labels_pl': adversarial_attack.labels_pl,
           'is_training_pl': adversarial_attack.is_training_pl,
           'pred': adversarial_attack.pred,
           'loss': adversarial_attack.classify_loss}

    NUM_POINT = FLAGS.num_point
#     NUM_POINT_ADV = NUM_POINT - num_drop*num_steps

    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range( NUM_CLASSES )]
    total_correct_class = [0 for _ in range( NUM_CLASSES )]
    for fn in range( 1 ):
        log_string( '----' + str( fn ) + '----' )
        current_data, current_label = provider.loadDataFile( TEST_FILES[fn] )
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze( current_label )

        for shapeIndex in range( 1 ):
#             testLabel = shapeIndex
            batchStart = findCorrectLabel( current_label )
            start_idx = batchStart * BATCH_SIZE
            end_idx = ( batchStart + 1 ) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            # Produce adversarial samples
            #===================================================================
            # Max pooling
            #===================================================================
#             deletCountnZero = adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "nonzero" )
#             deletCountZero = adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "zero" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "+average" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "+median" )
            adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
                                                    current_label[start_idx:end_idx], sess, "maxpooling", "+midrange" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "-average" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "-median" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "-midrange" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "+random", deletCountnZero )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "-random", deletCountZero )
            #===================================================================
            # Average pooling
            #===================================================================
#             deletCountnZero = adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "nonzero" )
#             deletCountZero = adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "zero" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "+average" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "+median" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "+midrange" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "-average" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "-median" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "-midrange" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "+random", deletCountnZero )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "avgpooling", "-random", deletCountZero )

def plotResults( inputArray, labelName, mode ):
    plt.plot( np.arange( len( inputArray ) ), inputArray, label = mode )
    plt.plot( np.arange( len( inputArray ) ), inputArray, 'C0o', alpha = 0.3 )
    plt.legend( title = ( labelName ) )
    plt.ylabel( mode )
    plt.xlabel( "Batch size" )
    plt.show()

def plotTwoAccuracies( inputArray1, inputArray2, labelName, ylabel, label1, label2 ):
    plt.plot( np.arange( len( inputArray1 ) ), inputArray1, label = label1 )
#     plt.plot(np.arange(len(inputArray1)), inputArray1, 'C0o', alpha=0.3)
    plt.plot( np.arange( len( inputArray2 ) ), inputArray2, label = label2 )
#     plt.plot(np.arange(len(inputArray2)), inputArray2, 'C1o', alpha=0.3)
    plt.legend( title = ( labelName ) )
    plt.ylabel( ylabel )
    plt.xlabel( "Iterations" )
    plt.show()

def plotTwoResults( pointdata, maxgradients, mode ):
    if mode == "average":
        average = gch.get_average( maxgradients )
        truncGrad = gch.truncate_to_threshold( maxgradients, average )
        plt.axhline( y = average, color = 'r', linestyle = '-', label = 'Average (Zeros ignored)' )
    elif mode == "median":
        median = gch.get_median( maxgradients )
        truncGrad = gch.truncate_to_threshold( maxgradients, median )
        plt.axhline( y = median, color = 'r', linestyle = '-', label = 'Median (Zeros ignored)' )
    elif mode == "midrange":
        midrange = gch.get_midrange( maxgradients )
        truncGrad = gch.truncate_to_threshold( maxgradients, midrange )
        plt.axhline( y = midrange, color = 'r', linestyle = '-', label = 'Mid-range (Zeros ignored)' )

    plt.plot( np.arange( len( maxgradients ) ), maxgradients, 'C0', label = 'Point weight value' )
    plt.plot( np.arange( len( maxgradients ) ), maxgradients, 'C0o', alpha = 0.3 )
    plt.plot( np.arange( len( maxgradients ) ), truncGrad, 'C1', label = 'Truncated gradient' )
    plt.plot( np.arange( len( maxgradients ) ), truncGrad, 'C1o', alpha = 0.3 )

    plt.legend( title = 'Average pooled weight plot:' )
    plt.show()
    gch.draw_heatcloud( pointdata, truncGrad )

if __name__ == "__main__":
    with tf.Graph().as_default():
        with tf.device( '/gpu:' + str( GPU_INDEX ) ):
            evaluate()

#     curShape = getShapeName( testLabel )
#     savePath = os.path.join( os.path.split( __file__ )[0], "testdata" , curShape )
#     #===========================================================================
#     # Max pooling
#     #===========================================================================
#     testResultsacc0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+average_accuracy" ) )
#     testResultsacc1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+median_accuracy" ) )
#     testResultsacc2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+midrange_accuracy" ) )
#     testResultsacc6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+random_accuracy" ) )
#     testResultsacc7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_nonzero_accuracy" ) )
#     testResultsloss0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+average_meanloss" ) )
#     testResultsloss1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+median_meanloss" ) )
#     testResultsloss2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+midrange_meanloss" ) )
#     testResultsloss6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+random_meanloss" ) )
#     testResultsloss7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_nonzero_meanloss" ) )
#      
#     vuptestResultsacc0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-average_accuracy" ) )
#     vuptestResultsacc1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-median_accuracy" ) )
#     vuptestResultsacc2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-midrange_accuracy" ) )
#     vuptestResultsacc6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-random_accuracy" ) )
#     vuptestResultsacc7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_zero_accuracy" ) )
#     vuptestResultsloss0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-average_meanloss" ) )
#     vuptestResultsloss1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-median_meanloss" ) )
#     vuptestResultsloss2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-midrange_meanloss" ) )
#     vuptestResultsloss6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-random_meanloss" ) )
#     vuptestResultsloss7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_zero_meanloss" ) )
#     #===============================================================================
#     # Average pooling
#     #===============================================================================
#     avgtestResultsacc0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+average_accuracy" ) )
#     avgtestResultsacc1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+median_accuracy" ) )
#     avgtestResultsacc2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+midrange_accuracy" ) )
#     avgtestResultsacc6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+random_accuracy" ) )
#     avgtestResultsacc7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_nonzero_accuracy" ) )
#     avgtestResultsloss0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+average_meanloss" ) )
#     avgtestResultsloss1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+median_meanloss" ) )
#     avgtestResultsloss2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+midrange_meanloss" ) )
#     avgtestResultsloss6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+random_meanloss" ) )
#     avgtestResultsloss7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_nonzero_meanloss" ) )
#      
#     vupavgtestResultsacc0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-average_accuracy" ) )
#     vupavgtestResultsacc1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-median_accuracy" ) )
#     vupavgtestResultsacc2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-midrange_accuracy" ) )
#     vupavgtestResultsacc6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-random_accuracy" ) )
#     vupavgtestResultsacc7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_zero_accuracy" ) )
#     vupavgtestResultsloss0 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-average_meanloss" ) )
#     vupavgtestResultsloss1 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-median_meanloss" ) )
#     vupavgtestResultsloss2 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-midrange_meanloss" ) )
#     vupavgtestResultsloss6 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-random_meanloss" ) )
#     vupavgtestResultsloss7 = tdh.readTestFile( os.path.join( savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_zero_meanloss" ) )
#   
#     plt.plot( np.arange( len( testResultsacc0 ) ), testResultsacc0, label = "Maxpooled average removed" )
#     plt.plot( np.arange( len( testResultsacc1 ) ), testResultsacc1, label = "Maxpooled median removed" )
#     plt.plot( np.arange( len( testResultsacc2 ) ), testResultsacc2, label = "Maxpooled midrange removed" )
#     plt.plot( np.arange( len( testResultsacc6 ) ), testResultsacc6, label = "Maxpooled Random removed" )
#     plt.plot( np.arange( len( testResultsacc7 ) ), testResultsacc7, label = "Maxpooled Non Zeros removed" )
#     plt.plot( np.arange( len( avgtestResultsacc0 ) ), avgtestResultsacc0, label = "Average pooled average removed" )
#     plt.plot( np.arange( len( avgtestResultsacc1 ) ), avgtestResultsacc1, label = "Average pooled median removed" )
#     plt.plot( np.arange( len( avgtestResultsacc2 ) ), avgtestResultsacc2, label = "Average pooled midrange removed" )
#     plt.plot( np.arange( len( avgtestResultsacc2 ) ), avgtestResultsacc6, label = "Average pooled Random removed" )
#     plt.plot( np.arange( len( avgtestResultsacc7 ) ), avgtestResultsacc7, label = "Average pooled Non Zeros removed" )
#     plt.legend( title = ( curShape + " important point removal plot" ) )
#     plt.ylabel( "Accuracy" )
#     plt.xlabel( "Iterations" )
#     plt.show()
#      
#     plt.plot( np.arange( len( vuptestResultsacc0 ) ), vuptestResultsacc0, label = "Maxpooled average removed" )
#     plt.plot( np.arange( len( vuptestResultsacc1 ) ), vuptestResultsacc1, label = "Maxpooled median removed" )
#     plt.plot( np.arange( len( vuptestResultsacc2 ) ), vuptestResultsacc2, label = "Maxpooled midrange removed" )
#     plt.plot( np.arange( len( vuptestResultsacc6 ) ), vuptestResultsacc6, label = "Maxpooled Random removed" )
#     plt.plot( np.arange( len( vuptestResultsacc7 ) ), vuptestResultsacc7, label = "Maxpooled Zeros removed" )
#     plt.plot( np.arange( len( vupavgtestResultsacc0 ) ), vupavgtestResultsacc0, label = "Average pooled average removed" )
#     plt.plot( np.arange( len( vupavgtestResultsacc1 ) ), vupavgtestResultsacc1, label = "Average pooled median removed" )
#     plt.plot( np.arange( len( vupavgtestResultsacc2 ) ), vupavgtestResultsacc2, label = "Average pooled midrange removed" )
#     plt.plot( np.arange( len( vupavgtestResultsacc6 ) ), vupavgtestResultsacc6, label = "Average pooled Random removed" )
#     plt.plot( np.arange( len( vupavgtestResultsacc7 ) ), vupavgtestResultsacc7, label = "Average pooled Zeros removed" )
#     plt.legend( title = ( curShape + " unimportant point removal plot" ) )
#     plt.ylabel( "Accuracy" )
#     plt.xlabel( "Iterations" )
#     plt.show()
#      
#     plt.plot( np.arange( len( testResultsloss0 ) ), testResultsloss0, label = "Maxpooled average removed" )
#     plt.plot( np.arange( len( testResultsloss1 ) ), testResultsloss1, label = "Maxpooled median removed" )
#     plt.plot( np.arange( len( testResultsloss2 ) ), testResultsloss2, label = "Maxpooled midrange removed" )
#     plt.plot( np.arange( len( testResultsloss6 ) ), testResultsloss6, label = "Maxpooled Random removed" )
#     plt.plot( np.arange( len( testResultsloss7 ) ), testResultsloss7, label = "Maxpooled Non Zeros removed" )
#     plt.plot( np.arange( len( avgtestResultsloss0 ) ), avgtestResultsloss0, label = "Average pooled average removed" )
#     plt.plot( np.arange( len( avgtestResultsloss1 ) ), avgtestResultsloss1, label = "Average pooled median removed" )
#     plt.plot( np.arange( len( avgtestResultsloss2 ) ), avgtestResultsloss2, label = "Average pooled midrange removed" )
#     plt.plot( np.arange( len( avgtestResultsloss6 ) ), avgtestResultsloss6, label = "Average pooled Random removed" )
#     plt.plot( np.arange( len( avgtestResultsloss7 ) ), avgtestResultsloss7, label = "Average pooled Non Zeros removed" )
#     plt.legend( title = ( curShape + " important point removal plot" ) )
#     plt.ylabel( "Loss" )
#     plt.xlabel( "Iterations" )
#     plt.show()
#      
#     plt.plot( np.arange( len( vuptestResultsloss0 ) ), vuptestResultsloss0, label = "Maxpooled average removed" )
#     plt.plot( np.arange( len( vuptestResultsloss1 ) ), vuptestResultsloss1, label = "Maxpooled median removed" )
#     plt.plot( np.arange( len( vuptestResultsloss2 ) ), vuptestResultsloss2, label = "Maxpooled midrange removed" )
#     plt.plot( np.arange( len( vuptestResultsloss6 ) ), vuptestResultsloss6, label = "Maxpooled Random removed" )
#     plt.plot( np.arange( len( vuptestResultsloss7 ) ), vuptestResultsloss7, label = "Maxpooled Zeros removed" )
#     plt.plot( np.arange( len( vupavgtestResultsloss0 ) ), vupavgtestResultsloss0, label = "Average pooled average removed" )
#     plt.plot( np.arange( len( vupavgtestResultsloss1 ) ), vupavgtestResultsloss1, label = "Average pooled median removed" )
#     plt.plot( np.arange( len( vupavgtestResultsloss2 ) ), vupavgtestResultsloss2, label = "Average pooled midrange removed" )
#     plt.plot( np.arange( len( vupavgtestResultsloss6 ) ), vupavgtestResultsloss6, label = "Average pooled Random removed" )
#     plt.plot( np.arange( len( vupavgtestResultsloss7 ) ), vupavgtestResultsloss7, label = "Average pooled Zeros removed" )
#     plt.legend( title = ( curShape + " unimportant point removal plot" ) )
#     plt.ylabel( "Loss" )
#     plt.xlabel( "Iterations" )
#     plt.show()

    LOG_FOUT.close()

