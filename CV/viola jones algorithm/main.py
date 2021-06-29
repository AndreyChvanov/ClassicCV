import cv2
import numpy as np
import os
from Task2 import Haar_Likes as hk
import pickle
from Task2 import AdaBoost
from Task2 import Face_Detection
import matplotlib.pyplot as plt



f = open("adaboost_35_stump.pickle", 'rb')
ab = pickle.load(f)
f.close()
f = open("features_parametrs", 'rb')
params = pickle.load(f)
f.close()


# window_step = 4
# confidence_threshold = 0.8
# nms_threshold = 0.1
# imgname = "test_group_1.png"
# fd = Face_Detection.FaceDetection(ab, params)
# image = cv2.imread(os.path.join("image",imgname), 0)
# faces = fd.slide_window(image, window_step)
# faces_NMS = fd.NMS(faces, nms_threshold)
# image_color = cv2.imread(os.path.join("image",imgname))
# fd.draw_bb(faces_NMS, image_color, confidence_threshold, f"NWC = {ab.NWC} , Window step = {window_step} Conf th = {confidence_threshold}")


# im = np.float32(cv2.imread(os.path.join("image", "starwars.jpg")))
# im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# im_grey = im_rgb.mean(axis=2)
# im_smooth = np.float32(cv2.GaussianBlur(im_grey, (3,3), 10))
# r = im_smooth[50, :]
# sobel_1  = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
# sobel_2 = sobel_1.T
# im_sobel_1 = cv2.filter2D(im_smooth,ddepth=-1, kernel=sobel_1)
# im_sobel_2 = cv2.filter2D(im_smooth,ddepth=-1, kernel=sobel_2)
# G = np.sqrt(im_sobel_1**2 + im_sobel_2 ** 2)
# G_1 = G>100
# plt.figure()
# plt.imshow(G_1, cmap='gray')
# plt.show()
#conv
#np.atan2(y,x )
# cv2.filter2D(input,ddepth=-1,kernel, anchor(-1,1), delta, border_type)
