'''
Created on 14.05.2019

@author: Dennis Struhs
'''

import os
import sys
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
