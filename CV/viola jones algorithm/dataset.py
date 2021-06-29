import cv2
import numpy as np
import os
from Task2 import Haar_Likes as hk
import pickle
from Task2 import  AdaBoost
pattern_types = ["a", "b", "c", "d", "e"]
from Task2 import Face_Detection


def load_dataset(folder):
    data = []
    for filename in os.listdir(folder):
        print(filename)
        im = np.float32(cv2.imread(os.path.join(folder, filename), 0))
        data.append(im)
    return np.array(data)


def load_features(folder):
    data = []
    for filename in os.listdir(folder):
        f = open(os.path.join(folder, filename), 'rb')
        features = np.float32(pickle.load(f))
        if not np.isnan(features)[0]:
            data.append(features)
    return np.array(data)


def get_dataset_by_features(directory):
    face = load_features(os.path.join(directory, "face"))
    face_target = np.ones(face.shape[0])
    weights_1 = np.zeros(face.shape[0]) + 1/(2*len(face_target))
    non_face = load_features(os.path.join(directory, "non-face"))
    non_face_target = np.zeros(non_face.shape[0])
    weights_2 = np.zeros(non_face.shape[0]) + 1/(2*len(non_face_target))
    X = np.concatenate((face, non_face), axis=0)
    Y = np.concatenate((face_target, non_face_target), axis=0)
    return np.float32(X), Y, np.concatenate((weights_1, weights_2))


def save_data_features(data, directory_path):
    hl = hk.Haar_Like()
    for i, im in enumerate(data):
        features = hl.get_all_features_by_image(im)
        filename = "feature_img" + str(i) + ".picke"
        print(filename)
        filename = os.path.join(directory_path, filename)
        f = open(filename, 'wb')
        pickle.dump(features, f)
        f.close()

def metrics(directory, adaboost, features_params):
    hl = hk.Haar_Like()
    face = load_dataset(os.path.join(directory,"face"))
    face_target = np.ones(face.shape[0])
    features_list = [stump.best_feature for stump in adaboost.stumps]
    face = [hl.get_chose_feature(features_list, features_params, np.float32(im)) for im in face]
    non_face = load_dataset(os.path.join(directory,"non-face"))
    non_face_target = np.zeros(non_face.shape[0])
    non_face = [hl.get_chose_feature(features_list, features_params, np.float32(im)) for im in non_face]
    X = np.concatenate((face, non_face), axis=0)
    Y = np.concatenate((face_target, non_face_target), axis=0)
    all_predict = []
    for sample in X:
        all_predict.append(AdaBoost.predict_by_sample(sample,adaboost))
    all_predict = np.array(all_predict)
    print(Y.shape)
    accuracy = AdaBoost.accuracy(all_predict, Y)
    TPR = AdaBoost.true_positive_rate(all_predict, Y)
    FPR = AdaBoost.false_positive_rate(all_predict, Y)
    str = f"set :{directory}\n accuracy {accuracy}\n true positive rate:{TPR}\n false positive rate:{FPR}"
    f = open(f"result_{directory}_set.txt", 'w')
    f.write(str)
    f.close()







# X_train, Y_train, W = get_dataset_by_features(os.path.join("features","train"))
# print(X_train.shape, Y_train.shape)
# mask = np.array(range(len(X_train)))
# np.random.shuffle(mask)
# X_train, Y_train, W = X_train[mask], Y_train[mask], W[mask]
# print("shuffle")
# print("step 1 correct")

# ab = AdaBoost.AdaBoost(15, W)
# ab.fit(X_train, Y_train)

# f = open("AdaBoost15stump.pickle", 'wb')
# pickle.dump(ab,f)
# f.close()

# f = open("AdaBoost15stump.pickle", 'rb')
# ab = pickle.load(f)
# f.close()
# f = open("features_parametrs", 'rb')
# params = pickle.load(f)
# f.close()
#
# metrics("train", ab, params)
# f = open("second_adaboost_15_stump.picke", 'rb')
# ab = pickle.load(f)
# f.close()


# step = 4
# confidence_threshold = 0.78
# iou_threshold = 0.1
# imgname = "swep6.jpg"
# fd = Face_Detection.FaceDetection(ab, params)
# image = cv2.imread(os.path.join("image",imgname), 0)
# faces = fd.slide_window(image, step)
# print( "f",len(faces))
# faces_NMS = fd.NMS(faces, iou_threshold)
# print(faces_NMS)
# image_color = cv2.imread(os.path.join("image",imgname))
# fd.draw_bb(faces_NMS, image_color, confidence_threshold, f"NWC = {ab.NWC} , Window step = {step} Conf th = {confidence_threshold}")
# print(fd._IoU([0,0,10, 10],[2, 2, 10, 10]))



