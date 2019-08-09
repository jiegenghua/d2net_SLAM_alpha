import cv2
import numpy as np
import glob
import h5py

img1 = cv2.imread("../dataset/sequences/03/image_0/000000.png",cv2.IMREAD_GRAYSCALE)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params,search_params)

# all of the images compared to the first frame to see how many matches

# all_images_to_compare = []

titles = []

sift =  cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
for f in glob.iglob("../dataset/sequences/03/image_0/*.png"):
    titles.append(f)

titles = sorted(titles)

for title in titles:
    img2 = cv2.imread(title,0)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    
    # Match descriptors.
    #matches = flann.knnMatch(des1,des2,k=2)
    matches = bf.match(des1,des2)
    #matches = sorted(matches,key=lambda x: x.distance)
    good = []
    #for m,n in matches:
    #    if m.distance < 0.8*n.distance:
    #        good.append(m)
    # The coordinates of keypoints
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC,3.0)
    matchesMask = mask.ravel().tolist()
    print(sum(matchesMask))

'''
for orb:
500
236
167
123
83
64
53
54
40
36
35
28
28
27


for sift:
3452
1506
1074
838
686
527
472
420
363
292
282
242


'''
