from Task2.AdditionalTasks import Cascade
from Task2.AdditionalTasks import AdaBoostC
from Task2.AdditionalTasks import Face_detectionC
from Task2 import dataset
import os
import numpy as np
import pickle
import cv2
from Task2.AdditionalTasks import Metrics


# X_train, Y_train, W = dataset.get_dataset_by_features(os.path.join("features","train"))
# print(X_train.shape, Y_train.shape)
# print("shuffle")
# print("step 1 correct")
# cascade = Cascade.Cascade(0.9, 0.001)
# cascade.fit(X_train, Y_train)

# f = open("cascade1.pickle", 'wb')
# pickle.dump(cascade, f)
# f.close()


f = open("cascade1.pickle", 'rb')
cascade = pickle.load(f)
f.close()
f = open("features_parametrs", 'rb')
params = pickle.load(f)
f.close()
print(len(cascade.adaboost_list[-1][0].decision_stumps))
# Metrics.metrics("../test", cascade, params)

print(cascade.F, cascade.D)
for i, ab in enumerate(cascade.adaboost_list):
    print("adaboost = ", i, "count stumps", len(ab[0].decision_stumps))



window_step = 4
confidence_threshold = 0.75
nms_threshold = 0.1
imgname = "lukehanleia.jpg"
fd = Face_detectionC.FaceDetection(cascade, params)
image = cv2.imread(os.path.join(imgname), 0)
faces = fd.slide_window(image, window_step)
faces_NMS = fd.NMS(faces, nms_threshold)
image_color = cv2.imread(os.path.join(imgname))
fd.draw_bb(faces_NMS, image_color, confidence_threshold, f"Cascade = {len(cascade.adaboost_list)} , Window step = {window_step} Conf th = {confidence_threshold}")


