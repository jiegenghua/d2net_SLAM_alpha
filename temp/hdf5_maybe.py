import os
import numpy as np
import cv2
import h5py


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

    
fe = cv2.ORB_create()
orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

NEW_SIZE = 1200, 300

SEQ = 3

if not os.path.exists('stat/%02d'%SEQ):
    os.makedirs('stat/%02d'%SEQ)

MACHINE = 1
if MACHINE == 1:
    DATA_ROOT_PATH = '../dataset/sequences/%02d/image_0'%SEQ
elif MACHINE == 2:
    DATA_ROOT_PATH = '/opt/BenbihiAssia/datasets/kitti/%02d/image_2/'%SEQ
else:
    print("stats_det: Once again, get your MTF MACHINE macro correct.")
    exit(1)

OUT_DIR = './result'
MIN_MATCH_COUNT = 10
EXT = '.hdf5'

img_fn_l = [l.split("\n")[0] for l in open("kitti_image2.txt").readlines()]
img_num = len(img_fn_l)


for img_id, img_ref_fn in enumerate(img_fn_l):
    if img_id > img_num - 10:
        break
    # d2net detection
    (img_ref_fn, ext) = os.path.splitext(img_ref_fn)
    out_ref_fn = '%s/%s%s'%(OUT_DIR, img_ref_fn, EXT)
    if not os.path.exists(out_ref_fn):
        continue
    f0 = h5py.File(out_ref_fn,'r')
    des0_key = list(f0.keys())[0]
    kp0_key = list(f0.keys())[1]
    kp0_p = np.array(f0[kp0_key], np.float32)
    des0 = np.array(f0[des0_key])
    kp0 = []
    for keypoint in kp0_p:
        kp0.append(cv2.KeyPoint(x=keypoint[0],y=keypoint[1],_size=1))
    pts0 = kp0
    des0 = des0
    match_num = []
    for j in range(img_id+1,img_id+10):
        img_fn = img_fn_l[j]
        (img_fn, ext) = os.path.splitext(img_fn)
        out_fn = '%s/%s%s'%(OUT_DIR, img_fn, EXT)
        f1 = h5py.File(out_fn, 'r')
        des1_key = list(f1.keys())[0]
        kp1_key = list(f1.keys())[1]
        kp1_p = np.array(f1[kp1_key], np.float32)
        des1 = np.array(f1[des1_key])
        kp1 = []
        for keypoint in kp1_p:
            kp1.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], _size=1))

        pts1 = kp1
        des1 = des1
        # d2-net matches
        good = [] # matching features according to desc. distance
        matches = flann.knnMatch(des0, des1, k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
        
        if len(good)>MIN_MATCH_COUNT:
            kp1_good    = [ kp0[m.queryIdx] for m in good ]
            kp2_good    = [ kp1[m.trainIdx] for m in good ]
            kp1_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp1_good])
            kp2_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp2_good])
            src_pts = np.float32(kp1_v).reshape(-1,1,2)
            dst_pts = np.float32(kp2_v).reshape(-1,1,2)
        
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,3.0)
            if F is None:
                print('No satisfying F was found :(')
                match_H_img_dict[img_key] = np.zeros((100,100)).astype(np.uint8)
                
            matchesMask = mask.ravel().tolist()
            draw_params = dict( matchesMask = matchesMask, flags = 2)
        else:
            print('sp of d2net len(good): %d'%(len(good)))
        print('\nd2: good: %d\tmatchesMask: %d'%(len(good),len(matchesMask)))

'''



'''
