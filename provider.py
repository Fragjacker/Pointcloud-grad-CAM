import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfileName = os.path.basename(www)
    targetFile = os.path.join(DATA_DIR, zipfileName)
    
    # -- Perform data fetching on Linux
    if sys.platform == 'linux' or sys.platform == 'linux2':
        os.system('wget %s; unzip %s' % (www, zipfileName))
        os.system('mv %s %s' % (zipfileName[:-4], DATA_DIR))
        os.system('rm %s' % (zipfileName))
        
    # -- Do this on Windows platforms.
    if sys.platform == 'win32':
        import zipfile
        import requests
        # -- Download the files
        with open(targetFile, "wb") as f:
            print("Downloading: %s" % (zipfileName))
            response = requests.get(www, stream=True)
            total_length = response.headers.get('content-length')
      
            if total_length is None: # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()
        f.close()
        print("\nDownload finished! Extracting archive!")
        # -- Unzip the files and remove the zip archive once done.
        zip_ref = zipfile.ZipFile(targetFile, 'r')
        zip_ref.extractall(DATA_DIR)
        zip_ref.close()
        os.remove(targetFile)
        print("Downloading data finished.")
    
    
def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_XYZ(batch_data):
    """ Randomly rotate the point clouds around XYZ axis to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        #--Rotate around the X-Axis
        rotation_matrix_X = np.array([[1, 0, 0],
                                      [0, cosval, -sinval],
                                      [0, sinval, cosval]])
        #--Rotate around the Y-Axis
        rotation_matrix_Y = np.array([[cosval, 0, sinval],
                                      [0, 1, 0],
                                      [-sinval, 0, cosval]])
        #--Rotate around the Z-Axis
        rotation_matrix_Z = np.array([[cosval, -sinval, 0],
                                      [sinval, cosval, 0],
                                      [0, 0, 1]])
        
        rotated_data[k, ...] = np.dot(batch_data[k, ...].reshape((-1, 3)), rotation_matrix_X)
        rotated_data[k, ...] = np.dot(rotated_data[k, ...].reshape((-1, 3)), rotation_matrix_Y)
        rotated_data[k, ...] = np.dot(rotated_data[k, ...].reshape((-1, 3)), rotation_matrix_Z)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
