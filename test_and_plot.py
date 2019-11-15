import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
from matplotlib import pyplot as plt
import gen_contrib_heatmap as gch
import test_data_handler as tdh

#===============================================================================
# Global variables to control program behavior
#===============================================================================
usePreviousSession = True   #--Set this to true to use a previously trained model.
performTraining = False     #--Set this to true to train the model. Set to false to only test the pretrained model.
desiredLabel = 1            #--The index of the class label the object should be tested against.
numTestBatchSize = 1000     #--Amount of tests for the current test label object.
maxNumPoints = 2048         #--How many points should be considered? [256/512/1024/2048] [default: 1024]

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
def computeHeatGradient(class_activation_vector, feature_vector, classIndex):
    # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
    class_activation_vector = tf.tensordot(class_activation_vector, tf.one_hot(indices=classIndex, depth=40),axes=[[1],[0]])
    
    # Compute gradient of the class prediction vector w.r.t. the feature vector. Use class_activation_vector[classIndex] to set which class shall be probed.
    gradients = tf.gradients(ys=class_activation_vector, xs=feature_vector)
    gradients = tf.squeeze(gradients, axis=[0,1,3])

    # Average pooling of the weights over all batches
#     gradients = tf.reduce_mean(gradients, axis=1)   # Average pooling
    gradients = tf.reduce_max(gradients, axis=1)    # Max pooling
#     gradients = tf.squeeze(tf.map_fn(lambda x: x[100:101], gradients)) # Stride pooling

    # Multiply with original pre maxpool feature vector to get weights
    feature_vector = tf.squeeze(feature_vector,axis=[0,2])  # Remove empty dimensions of the feature vector so we get [batch_size,1024]
    multiply = tf.constant(feature_vector[1].get_shape().as_list()) # Feature vector matrix
    multMatrix = tf.reshape(tf.tile(gradients, multiply), [ multiply[0], gradients.get_shape().as_list()[0]])   # Reshape [batch_size,] to [1024, batch_size] by copying the row n times
    gradients = tf.matmul(feature_vector, multMatrix)   # Multiply [batch_size, 1024] x [1024, batch_size]
    gradients = tf.diag_part(gradients) # Due to Matmul the interesting values are on the diagonal part of the matrix.
    
    # ReLU out the negative values
    gradients = tf.maximum(gradients, 0)
    
    return gradients

def getOps(pointclouds_pl, labels_pl, is_training_pl, batch, pred, maxpool_out, feature_vec, loss, train_op, merged, gradients):
    if gradients is None:
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
            'gradients':gradients}
    return ops

def getShapeName(index):
    '''
    This function returns the label index as a string,
    using the shape_names.txt file for reference.
    '''
    shapeTXT = open('data\\modelnet40_ply_hdf5_2048\\shape_names.txt', 'r')
    entry = ''
    for _ in range(index + 1):
        entry = shapeTXT.readline().rstrip()
    shapeTXT.close()
    return entry

def findCorrectLabel(inputLabelArray):
    '''
    This function retrieves the correct shape from the
    test batch that matches the target feature vector.
    '''
    result = None
    compLabel = getShapeName(testLabel)
    for currentIndex in range(len(inputLabelArray)):
        curLabel = getShapeName(inputLabelArray[currentIndex:currentIndex+1][0])
        if curLabel.lower() == compLabel.lower():
            result = currentIndex
            break
    return result

def getPrediction(predIndex):
    return getShapeName(predIndex[0])
        
testLabel = desiredLabel - 1    #-- Subtract 1 to make the label match Python array enumeration, which starts from 0.
#------------------------------------------------------------------------------ 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=maxNumPoints, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during evaluation [default: 1]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
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

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if sys.platform == 'linux' or sys.platform == 'linux2':
    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
    os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
if sys.platform == 'win32':
    import shutil
    shutil.copy(MODEL_FILE, LOG_DIR)
    shutil.copy('train.py', LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points, maxpool_out, feature_vec = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)
            
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})
        
        # --Reload pretrained model
        if usePreviousSession:
            trained_model = os.path.join(LOG_DIR, "model.ckpt")
            saver.restore(sess, trained_model)
            print("Model restored.")
            
        # --Compute gradients only when evaluating the model.
        if not performTraining:
            gradients = computeHeatGradient(pred, feature_vec, classIndex=testLabel)
        else:
            gradients = None

        # --Get proper Ops according to if we want to compute gradients or not.
        ops = getOps(pointclouds_pl, labels_pl, is_training_pl, batch, pred, maxpool_out, feature_vec, loss, train_op, merged, gradients)
        
        log_string('**** TESTING MODEL ****')
        sys.stdout.flush()
        heatMap, curPointCloud = test_rotation_XYZ(sess, ops, test_writer)
        sess.close()
        return heatMap, curPointCloud
            
            
def eval_perturbations(numInputPoints, perturbedData, mode):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, numInputPoints)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points, maxpool_out, feature_vec = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)
            
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})
        
        # --Reload pretrained model
        trained_model = os.path.join(LOG_DIR, "model.ckpt")
        saver.restore(sess, trained_model)
        print("Model restored.")
            
        # --Compute gradients only when evaluating the model.
        gradients = computeHeatGradient(pred, feature_vec, classIndex=testLabel)

        # --Get proper Ops according to if we want to compute gradients or not.
        ops = getOps(pointclouds_pl, labels_pl, is_training_pl, batch, pred, maxpool_out, feature_vec, loss, train_op, merged, gradients)
        
        log_string('**** TESTING MODEL ****')
        sys.stdout.flush()
        gradients = test_perturbed_pc(sess, ops, perturbedData, mode)
        sess.close()
        return gradients

def test_perturbed_pc(sess,ops,perturbedData,mode):
    total_correct_PDel = 0
    total_seen_PDel = 0
    loss_sum_PDel = 0
    total_seen_class_PDel = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_PDel = [0 for _ in range(NUM_CLASSES)]
    is_training = False
    accuracyResults = []
    meanlossResults = []
    
    for fn in range(1):
        log_string('----Testbatch ' + str(fn) + '-----')
        _, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_label = np.squeeze(current_label)
        
        batchStart = findCorrectLabel(current_label)
        
        for _ in range(numTestBatchSize):
            start_idx = batchStart * BATCH_SIZE
            end_idx = (batchStart+1) * BATCH_SIZE
            
            # -- Experiments with reduced point clouds start here
            
            perturbedData = provider.rotate_point_cloud_XYZ(perturbedData) # Rotate randomly around XYZ axis
            feed_dict = {ops['pointclouds_pl']: perturbedData,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val, maxpool_out, gradients = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred'], ops['maxpool_out'],ops['gradients']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct_PDel += correct
            total_seen_PDel += BATCH_SIZE
            loss_sum_PDel += (loss_val*BATCH_SIZE)
            accuracyResults.append(total_correct_PDel / float(total_seen_PDel))
            meanlossResults.append(loss_sum_PDel / float(total_seen_PDel))
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class_PDel[l] += 1
                total_correct_class_PDel[l] += (pred_val[i-start_idx] == l)
        
#             print('Grad-CAM for shape: "%s"' % getShapeName(current_label[start_idx:end_idx][0]))
#             print('Grad-CAM for test class label: "%s"' % getShapeName(testLabel))
#             print(gradients)
             
#         print("LENGTH GRADIENTS: ", len(gradients))
        average = gch.get_average(gradients)
#         median = gch.get_median(gradients)
        truncGrad = gch.truncate_to_threshold(gradients, average)
#               
#             plt.plot(np.arange(len(gradients)), gradients, 'C0', label='Original gradient')
#             plt.plot(np.arange(len(gradients)), gradients, 'C0o', alpha=0.3)
#             plt.plot(np.arange(len(gradients)), truncGrad, 'C1', label='Truncated gradient')
#             plt.plot(np.arange(len(gradients)), truncGrad, 'C1o', alpha=0.3)
#             plt.axhline(y=average, color='r', linestyle='-', label='Average (Zeros ignored)')
#             plt.axhline(y=median, color='b', linestyle='-', label='Median (Zeros ignored)')
#             plt.legend(title='Gradient value plot:')
#             plt.show()
      
    #         log_string('Max Pooling Array:')
    #         print(maxpool_out)
    #         log_string('Contributing vector indices:')
    #         gch.list_contrib_vectors(maxpool_out)
    #         log_string('Contributing vector index count:')
    #         occArr = gch.count_occurance(maxpool_out)
    
#         gch.draw_heatcloud(perturbedData, truncGrad)
#         filePath = getShapeName(testLabel)+"_"+str(numTestBatchSize)+"_"+str(mode)
#         print("STORING FILES TO: ", filePath)
#         tdh.writeAllResults(filePath, testResults)

    print("POINT DELETE PREDICTION: ", getPrediction(pred_val))   
    mean_loss = loss_sum_PDel / float(total_seen_PDel)
    filePath = getShapeName(testLabel)+"_"+str(numTestBatchSize)+"_"+str(mode)+"_meanloss"
    print("STORING FILES TO: ", filePath)
    tdh.writeAllResults(filePath, meanlossResults)
    filePath = getShapeName(testLabel)+"_"+str(numTestBatchSize)+"_"+str(mode)+"_accuracy"
    print("STORING FILES TO: ", filePath)
    tdh.writeAllResults(filePath, accuracyResults)
    
    log_string('eval mean loss: %f' % mean_loss)
    log_string('eval accuracy: %f'% (total_correct_PDel / float(total_seen_PDel)))
#     log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class_PDel)/np.array(total_seen_class_PDel,dtype=np.float))))
    return gradients
        
def test_rotation_XYZ(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    accuracyResults = []
    meanlossResults = []
    
    for fn in range(1):
        log_string('----Testbatch ' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        batchStart = findCorrectLabel(current_label)
        
        for _ in range(numTestBatchSize):
            start_idx = batchStart * BATCH_SIZE
            end_idx = (batchStart+1) * BATCH_SIZE
            currentPointCloud = current_data[start_idx:end_idx, :, :]

            rotated_data = provider.rotate_point_cloud_XYZ(currentPointCloud)
#             rotated_data = provider.rotate_point_cloud(currentPointCloud)
#             rotated_data = currentPointCloud
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val, maxpool_out, gradients = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred'], ops['maxpool_out'],ops['gradients']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            accuracyResults.append(total_correct / float(total_seen))
            meanlossResults.append(loss_sum / float(total_seen))
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
        
#             print('Gradients for shape: "%s"' % getShapeName(current_label[start_idx:end_idx][0]))
#             print('With grad-CAM for test class label: "%s"' % getShapeName(testLabel))
#                
#             print("LENGTH GRADIENTS: ", len(gradients))
#         plotTwoResults(rotated_data, gradients, "average")
#         plotTwoResults(rotated_data, gradients, "median")
#         plotTwoResults(rotated_data, gradients, "midrange")
#         print("ROTATION PREDICTION: ", getPrediction(pred_val))
#         sys.exit()
            
#         filePath = getShapeName(testLabel)+"_"+str(numTestBatchSize)+"_"+str(fn)+"_average_Y"
#         print("STORING FILES TO: ", filePath)
#         tdh.writeAllResults(filePath, accuracyResults)
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
#     log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    return gradients, currentPointCloud

def plotResults(inputArray, labelName):
    plt.plot(np.arange(len(inputArray)), inputArray, label='Prediction Accuracy')
    plt.plot(np.arange(len(inputArray)), inputArray, 'C0o', alpha=0.3)
    plt.legend(title=(labelName+' Accuracy value plot:'))
    plt.ylabel("Accuracy")
    plt.xlabel("Batch size")
    plt.show()
    
def plotTwoAccuracies(inputArray1, inputArray2, labelName):
    plt.plot(np.arange(len(inputArray1)), inputArray1, label='Average removed points')
    plt.plot(np.arange(len(inputArray1)), inputArray1, 'C0o', alpha=0.3)
    plt.plot(np.arange(len(inputArray1)), inputArray2, label='Random removed points')
    plt.plot(np.arange(len(inputArray1)), inputArray2, 'C1o', alpha=0.3)
    plt.legend(title=(labelName))
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.show()
    
def plotTwoResults(pointdata, gradients, mode):
    if mode == "average":
        average = gch.get_average(gradients)
        truncGrad = gch.truncate_to_threshold(gradients, average)
        plt.axhline(y=average, color='r', linestyle='-', label='Average (Zeros ignored)')
    elif mode == "median":
        median = gch.get_median(gradients)
        truncGrad = gch.truncate_to_threshold(gradients, median)
        plt.axhline(y=median, color='r', linestyle='-', label='Median (Zeros ignored)')
    elif mode =="midrange":
        midrange = gch.get_midrange(gradients)
        truncGrad = gch.truncate_to_threshold(gradients, midrange)
        plt.axhline(y=midrange, color='r', linestyle='-', label='Mid-range (Zeros ignored)')
    
    plt.plot(np.arange(len(gradients)), gradients, 'C0', label='Point weight value')
    plt.plot(np.arange(len(gradients)), gradients, 'C0o', alpha=0.3)
    plt.plot(np.arange(len(gradients)), truncGrad, 'C1', label='Truncated gradient')
    plt.plot(np.arange(len(gradients)), truncGrad, 'C1o', alpha=0.3)
    

    plt.legend(title='Average pooled weight plot:')
    plt.show()
    gch.draw_heatcloud(pointdata, truncGrad)

if __name__ == "__main__":
#     gradients = None
#     gradients, curPointCloud = train()
#     print("GRADIENT SIZE: ", len(gradients))
#     print("POINT CLOUD SIZE: ", curPointCloud.shape)
#     curPointCloudResult = curPointCloud
#     curPointCloudRand = curPointCloud
#     for _ in range(10):
#         resultPCloudThresh, delCount = gch.delete_above_threshold(gradients, curPointCloud, "average")
#         gradients = eval_perturbations(resultPCloudThresh.shape[1], resultPCloudThresh, "XYZ_average_removed")
# #         gch.draw_pointcloud(resultPCloudThresh)
# #         resultPCloud, delCount = gch.delete_all_above_average(gradients, curPointCloud)
# #         eval_perturbations(resultPCloud.shape[1], resultPCloud, "above_average_removed")
# #         resultPCloudNonzero, delCount = gch.delete_all_nonzeros(gradients, curPointCloudResult)
# #         gch.draw_pointcloud(resultPCloudNonzero)
# #         gradients = eval_perturbations(resultPCloudNonzero.shape[1], resultPCloudNonzero, "non_zeros_removed")
# #         resultPCloud, delCount = gch.delete_all_zeros(gradients, curPointCloud)
# #         gch.draw_pointcloud(resultPCloud)
# #         eval_perturbations(resultPCloud.shape[1], resultPCloud, "zeros_removed")
#         resultPCloudRand = gch.delete_randon_points(delCount, curPointCloudRand)
# #         gch.draw_pointcloud(resultPCloudRand)
#         _ = eval_perturbations(resultPCloudRand.shape[1], resultPCloudRand, "XYZ_random_removed")
#         curPointCloudResult = resultPCloudThresh
#         curPointCloudRand = resultPCloudRand

#     testResults = tdh.readTestFile("airplane_1000_0_average_Y")
#     plotResults(testResults, "Airplane")
    testResults1 = tdh.readTestFile("airplane_1000_XYZ_average_removed_meanloss")
    testResults2 = tdh.readTestFile("airplane_1000_XYZ_random_removed_meanloss")
    plotTwoAccuracies(testResults1,testResults2,"Airplane meanloss plot:")
    testResults1 = tdh.readTestFile("airplane_1000_XYZ_average_removed_accuracy")
    testResults2 = tdh.readTestFile("airplane_1000_XYZ_random_removed_accuracy")
    plotTwoAccuracies(testResults1,testResults2,"Airplane accuracy plot:")
    LOG_FOUT.close()




