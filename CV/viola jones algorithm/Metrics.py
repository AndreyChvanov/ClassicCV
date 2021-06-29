import cv2
import numpy as np
import os
from Task2 import Haar_Likes as hk
import pickle
from Task2.AdditionalTasks import Cascade
from Task2 import dataset


def metrics(directory, cascade, features_params):
    hl = hk.Haar_Like()
    face = dataset.load_dataset(os.path.join(directory,"face"))
    face_target = np.ones(face.shape[0])
    features_list = Cascade.get_chose_features(cascade)
    face = [hl.get_chose_feature(features_list, features_params, np.float32(im)) for im in face]
    non_face = dataset.load_dataset(os.path.join(directory,"non-face"))
    non_face_target = np.zeros(non_face.shape[0])
    non_face = [hl.get_chose_feature(features_list, features_params, np.float32(im)) for im in non_face]
    X = np.concatenate((face, non_face), axis=0)
    Y = np.concatenate((face_target, non_face_target), axis=0)
    all_predict = []
    for sample in X:
        all_predict.append(Cascade.predict_by_sample(cascade,sample))
    all_predict = np.array(all_predict)
    print(Y.shape)
    accuracy = (all_predict == Y).mean()
    TPR = Cascade.detection_rate(all_predict, Y)

    FPR = Cascade.false_positive_rate(all_predict, Y)
    str = f"set :{directory}\n accuracy {accuracy}\n detection rate cascade:{TPR}\n false positive rate cascade:{FPR}"
    f = open(f"result_test_set.txt", 'w')
    f.write(str)
    f.close()