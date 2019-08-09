import cv2
import numpy as np
import glob
import h5py

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

f1 = h5py.File('./result/000000.hdf5','r')
des1_key = list(f1.keys())[0]
kp1_key = list(f1.keys())[1]
kp1_p = np.array(f1[kp1_key], np.float32)
des1 = np.array(f1[des1_key])
kp1 = []
for keypoint in kp1_p:
    kp1.append(cv2.KeyPoint(x=keypoint[0],y=keypoint[1],_size=1))


titles = []
# define a radius, if it is in the area of a keypoint, then bin++
r = 5
for f in glob.iglob("./result/*.hdf5"):
    titles.append(f)

titles = sorted(titles)
#print(titles)
for title in titles:
    f2 = h5py.File(title,'r')
    des2_key = list(f2.keys())[0]
    kp2_key = list(f2.keys())[1]
    kp2_p = np.array(f2[kp2_key], np.float32)
    des2 = np.array(f2[des2_key])
    kp2 = []
    for keypoint in kp2_p:
        kp2.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], _size=1))
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    # The coordinates of keypoints
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC,3.0)
    #if F is None:
    #    print('No satisfying F was found :(')
    #    match_H_img_dict[img_key] = np.zeros((100, 100)).astype(np.uint8)

    matchesMask = mask.ravel().tolist()
    print(sum(matchesMask))
    #draw_params = dict(matchesMask=matchesMask, flags=2)
    # match_H_img = cv2.drawMatches(img0, kp0, img1, kp1, good, None, **draw_params)
