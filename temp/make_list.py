import numpy as np
import os
from PIL import Image
import os
import glob

writer = open('kitti_hdf5.txt','a')
img_dir = '/home/mizhou/MyPro/ORB_SLAM2/result'
L = os.listdir(img_dir)
L.sort()
for f in L:
    file_path = os.path.join(img_dir,f)
    print(file_path)
    writer.write(file_path)
    writer.write('\n')

writer.close()
