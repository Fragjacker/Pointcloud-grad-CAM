import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import copy
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'shared_functions'))
import provider
import matplotlib as mpl

mpl.use('pdf')
from help_functions import getShapeName, findCorrectLabel, getPrediction
import gen_contrib_heatmap as gch
import test_data_handler as tdh
import codeProfiler as cpr

# ===============================================================================
# Global variables to control program behavior
# ===============================================================================
usePreviousSession = True   # Set this to true to use a previously trained model.
performTraining = False     # Set this to true to train the model. Set to false to only test the pretrained model.
desiredLabel = 1            # The index of the class label the object should be tested against. It matches with the line numbers of the shapes.txt files e.g. line 1 = airplane etc.
numTestRuns = 500           # Amount of tests for the current test label object.
maxNumPoints = 2048         # How many points should be considered? [256/512/1024/2048] [default: 1024]
storeResults = False        # Should the results of the algorithm be stored to files or not.


# ===============================================================================
# Help Functions
# ===============================================================================


def storeTestResults(mode, total_correct, total_seen, loss_sum, pred_val):
    '''
    This function stores the test data into seperate files for later retrieval.
    '''
    curShape = getShapeName(desiredClassLabel)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata", curShape)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    mean_loss = loss_sum / float(total_seen)
    accuracy = total_correct / float(total_seen)
    filePath = os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_" + str(mode) + "_meanloss")
    print("STORING FILES TO: ", filePath)
    tdh.writeResult(filePath, mean_loss)
    filePath = os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_" + str(mode) + "_accuracy")
    print("STORING FILES TO: ", filePath)
    tdh.writeResult(filePath, accuracy)
    filePath = os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_" + str(mode) + "_prediction")
    print("STORING FILES TO: ", filePath)
    tdh.writeResult(filePath, pred_val)
    log_string('eval mean loss: %f' % mean_loss)
    log_string('eval accuracy: %f' % accuracy)


def storeAmountOfPointsRemoved(numPointsRemoved):
    '''
    This function stores the amount of points removed per iteration.
    '''
    curShape = getShapeName(desiredClassLabel)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata", "p-grad-CAM_ppi")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    filePath = os.path.join(savePath, curShape + "_points_removed")
    print("STORING FILES TO: ", filePath)
    tdh.writeResult(filePath, numPointsRemoved)


def storeAccuracyPerPointsRemoved(accuracy):
    '''
    This function stores the amount of points removed per iteration.
    '''
    curShape = getShapeName(desiredClassLabel)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata", "p-grad-CAM_ppi")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    filePath = os.path.join(savePath, curShape + "_accuracy")
    print("STORING FILES TO: ", filePath)
    tdh.writeResult(filePath, accuracy)


def storeAmountOfUsedTime(usedTime):
    '''
    This function stores the amount of total time used per object.
    '''
    curShape = getShapeName(desiredClassLabel)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata", "p-grad-CAM_performance")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    filePath = os.path.join(savePath, curShape)
    print("STORING FILES TO: ", filePath)
    tdh.writeResult(filePath, usedTime)

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--desired_label', type=int, default=desiredLabel, help='The desired class label for the target shape. For example 1 for airplane, 2 for bathtub etc.')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=maxNumPoints,
                    help='Point Number [256/512/1024/2048] [default: 1024]')
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

desiredClassLabel = FLAGS.desired_label - 1  # -- Subtract 1 to make the label match Python array enumeration, which starts from 0.

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if sys.platform == 'linux' or sys.platform == 'linux2':
    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
    os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
if sys.platform == 'win32':
    import shutil

    shutil.copy(MODEL_FILE, LOG_DIR)
    shutil.copy('train.py', LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

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
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


class AdversialPointCloud():
    def __init__(self, numPoints=BATCH_SIZE):
        self.is_training = False
        self.count = np.zeros((NUM_CLASSES,), dtype=bool)
        self.all_counters = np.zeros((NUM_CLASSES, 3), dtype=int)

        # The number of points is not specified
        self.pointclouds_pl, self.labels_pl = MODEL.placeholder_inputs(numPoints, None)
        self.is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        self.pred, self.end_points, _, self.feature_vec = MODEL.get_model(self.pointclouds_pl, self.is_training_pl)
        self.classify_loss = MODEL.get_loss(self.pred, self.labels_pl, self.end_points)

    def getGradient(self, sess, poolingMode, class_activation_vector, feed_dict):
        # Compute gradient of the class prediction vector w.r.t. the feature vector. Use class_activation_vector[classIndex] to set which class shall be probed.
        maxgradients = sess.run(tf.gradients(ys=class_activation_vector, xs=self.feature_vec)[0], feed_dict=feed_dict)
        maxgradients = tf.squeeze(maxgradients, axis=[0, 2])
        # Average pooling of the weights over all batches
        if poolingMode == "avgpooling":
            maxgradients = tf.reduce_mean(maxgradients, axis=1)  # Average pooling
        elif poolingMode == "maxpooling":
            maxgradients = tf.reduce_max(maxgradients, axis=1)  # Max pooling
        #             maxgradients = tf.squeeze(tf.map_fn(lambda x: x[100:101], maxgradients)) # Stride pooling
        # Multiply with original pre maxpool feature vector to get weights
        feature_vector = tf.squeeze(self.feature_vec, axis=[0,
                                                            2])  # Remove empty dimensions of the feature vector so we get [batch_size,1024]
        multiply = tf.constant(feature_vector[1].get_shape().as_list())  # Feature vector matrix
        multMatrix = tf.reshape(tf.tile(maxgradients, multiply), [multiply[0], maxgradients.get_shape().as_list()[
            0]])  # Reshape [batch_size,] to [1024, batch_size] by copying the row n times
        maxgradients = tf.matmul(feature_vector, multMatrix)  # Multiply [batch_size, 1024] x [1024, batch_size]
        maxgradients = tf.diag_part(
            maxgradients)  # Due to Matmul the interesting values are on the diagonal part of the matrix.
        # ReLU out the negative values
        maxgradients = tf.maximum(maxgradients, 0)
        return maxgradients

    def drop_and_store_results(self, pointclouds_pl, labels_pl, sess, poolingMode, thresholdMode, numDeletePoints=None):
        # Some profiling
        import time
        start_time = time.time()
        cpr.startProfiling()

        pcTempResult = pointclouds_pl.copy()
        delCount = []
        vipPcPointsArr = []
        weightArray = []
        i = 0

        # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
        class_activation_vector = tf.multiply(self.pred, tf.one_hot(indices=desiredClassLabel, depth=40))

        while True:
            i += 1
            print("ITERATION: ", i)
            # Setup feed dict for current iteration
            feed_dict = {self.pointclouds_pl: pcTempResult,
                         self.labels_pl: labels_pl,
                         self.is_training_pl: self.is_training}

            maxgradients = self.getGradient(sess, poolingMode, class_activation_vector, feed_dict)

            ops = {'pred': self.pred,
                   'loss': self.classify_loss,
                   'maxgradients': maxgradients}

            # ===================================================================
            # Evaluate over n batches now to get the accuracy for this iteration.
            # ===================================================================
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            pcEvalTest = copy.deepcopy(pcTempResult)
            for _ in range(numTestRuns):
                pcEvalTest = provider.rotate_point_cloud_XYZ(pcEvalTest)
                feed_dict2 = {self.pointclouds_pl: pcEvalTest,
                              self.labels_pl: labels_pl,
                              self.is_training_pl: self.is_training}
                eval_prediction, eval_loss, heatGradient = sess.run([ops['pred'], ops['loss'], ops['maxgradients']],
                                                                    feed_dict=feed_dict2)
                eval_prediction = np.argmax(eval_prediction, 1)
                correct = np.sum(eval_prediction == labels_pl)
                total_correct += correct
                total_seen += 1
                loss_sum += eval_loss * BATCH_SIZE

            print("GROUND TRUTH: ", getShapeName(desiredClassLabel))
            print("PREDICTION: ", getPrediction(eval_prediction))
            print("LOSS: ", eval_loss)
            print("ACCURACY: ", (total_correct / total_seen))
            accuracy = total_correct / float(total_seen)

            # Store data now if desired
            if storeResults:
                curRemainingPoints = NUM_POINT - sum(delCount)
                storeAmountOfPointsRemoved(curRemainingPoints)
                storeAccuracyPerPointsRemoved(accuracy)

            # Stop iterating when the eval_prediction deviates from ground truth
            if desiredClassLabel != eval_prediction and accuracy <= 0.5:
                print("GROUND TRUTH DEVIATED FROM PREDICTION AFTER %s ITERATIONS" % i)
                break

            # Perform visual stuff here
            if thresholdMode == "+average" or thresholdMode == "+median" or thresholdMode == "+midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_above_threshold(heatGradient, pcTempResult,
                                                                                     thresholdMode)
            if thresholdMode == "-average" or thresholdMode == "-median" or thresholdMode == "-midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_below_threshold(heatGradient, pcTempResult,
                                                                                     thresholdMode)
            if thresholdMode == "nonzero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_nonzeros(heatGradient, pcTempResult)
            if thresholdMode == "zero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_zeros(heatGradient, pcTempResult)
            if thresholdMode == "+random" or thresholdMode == "-random":
                resultPCloudThresh, vipPointsArr = gch.delete_random_points(heatGradient, pcTempResult,
                                                                            numDeletePoints[i])
                Count = numDeletePoints[i]
            print("REMOVING %s POINTS." % Count)

            delCount.append(Count)
            vipPcPointsArr.extend(vipPointsArr[0])
            weightArray.extend(vipPointsArr[1])
            pcTempResult = copy.deepcopy(resultPCloudThresh)
            
        # Stop profiling and show the results
        endTime = time.time() - start_time
        storeAmountOfUsedTime(endTime)
        cpr.stopProfiling(numResults=20)
        print("TIME NEEDED FOR ALGORITHM: ", endTime)

        totalRemoved = sum(delCount)
        print("TOTAL REMOVED POINTS: ", totalRemoved)
        print("TOTAL REMAINING POINTS: ", NUM_POINT - totalRemoved)
        #         gch.draw_pointcloud(pcTempResult) #-- Residual point cloud
        gch.draw_NewHeatcloud(vipPcPointsArr, weightArray) #-- Important points only
        #         vipPcPointsArr.extend(pcTempResult[0])
        #         gch.draw_NewHeatcloud(vipPcPointsArr, weightArray) #--All points combined
        return delCount


def evaluate():
    global desiredClassLabel
    is_training = False
    adversarial_attack = AdversialPointCloud()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Init variables
    init = tf.global_variables_initializer()
    # To fix the bug introduced in TF 0.12.1 as in
    # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
    # sess.run(init)
    sess.run(init, {adversarial_attack.is_training_pl: False})

    # Restore variables from disk.
    trained_model = os.path.join(LOG_DIR, "model.ckpt")
    saver.restore(sess, trained_model)
    log_string("Model restored.")

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
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    for fn in range(1):
        log_string('----' + str(fn) + '----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)

        # This loop is left in case a larger test set should be iterated over
        for shapeIndex in range(1):
#             desiredClassLabel = shapeIndex
            batchStart = findCorrectLabel(current_label, desiredClassLabel)
            start_idx = batchStart * BATCH_SIZE
            end_idx = (batchStart + 1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            # Produce adversarial samples
            # ===================================================================
            # Max pooling
            # ===================================================================
#             deletCountnZero = adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "nonzero" )
#             deletCountZero = adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "zero" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "+average" )
#             adversarial_attack.drop_and_store_results( current_data[start_idx:end_idx, :, :],
#                                                     current_label[start_idx:end_idx], sess, "maxpooling", "+median" )
            adversarial_attack.drop_and_store_results(current_data[start_idx:end_idx, :, :],
                                                      current_label[start_idx:end_idx], sess, "maxpooling", "+midrange")


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
            # ===================================================================
            # Average pooling
            # ===================================================================
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

if __name__ == "__main__":
    with tf.Graph().as_default():
        with tf.device( '/gpu:' + str( GPU_INDEX ) ):
            evaluate()
    LOG_FOUT.close()
