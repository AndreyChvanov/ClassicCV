from Task2 import AdaBoost
from Task2 import decision_stump
from Task2 import Haar_Likes
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class FaceDetection:
    def __init__(self, adaboost, features_params):
        self.adaboost = adaboost
        self.chose_features = [stump.best_feature for stump in adaboost.stumps]
        self.features_param = features_params
        self.scale_factor = 1.25
        self.window_size = 19

    def slide_window(self, image, step):
        faces = []
        hl = Haar_Likes.Haar_Like()
        current_scale = 1
        h, w = image.shape[0], image.shape[1]
        while w > self.window_size and h > self.window_size:
            for i in range(0, h-self.window_size+1, step):
                for j in range(0, w-self.window_size+1, step):
                    window = np.float32(image[i:i+self.window_size, j:j+self.window_size])
                    features = hl.get_chose_feature(self.chose_features, self.features_param, window)
                    confidence = AdaBoost.confidence(self.adaboost, features)
                    if confidence > 0.5:
                        faces.append([i , j, confidence, current_scale])
            image = cv2.resize(image, (int(w/self.scale_factor), int(h/self.scale_factor)))
            h, w = image.shape[0], image.shape[1]
            current_scale = current_scale * 1.25
        return faces

    def _IoU(self, first_box, second_box):
        xA = max(first_box[0], second_box[0])
        yA = max(first_box[1], second_box[1])
        xB = min(first_box[2], second_box[2])
        yB = min(first_box[3], second_box[3])
        inter = max(0, xB-xA+1) * max(0, yB - yA + 1)
        first_box_Area = (first_box[2] - first_box[0]+1)*(first_box[3] - first_box[1]+1)
        second_box_Area = (second_box[2] - second_box[0]+1)*(second_box[3] - second_box[1]+1)
        union = float(first_box_Area+second_box_Area-inter)
        iou = inter/union
        return iou

    def NMS(self, detected_faces, lamb):
        n = len(detected_faces)
        sorted_detected = sorted(detected_faces, key=lambda unit: unit[2], reverse=True)
        faces = np.ones(n)
        for i in range(n - 1):
            if not faces[i]:
                continue
            box = sorted_detected[i]
            cur_scale = box[3]
            Ax0, Ay0 = int(box[0]*box[3]), int(box[1]*box[3])
            Ax1, Ay1 = int((box[0] + self.window_size) * cur_scale) - 1, int(
                (box[1] + self.window_size) * cur_scale) - 1
            for j in range(i + 1, n):
                if not faces[j]:
                    continue
                some_box = sorted_detected[j]
                cur_scale = some_box[3]
                Bx0, By0 = int(some_box[0]*some_box[3]), int(some_box[1]*some_box[3])
                Bx1, By1 = int((some_box[0] + self.window_size) * cur_scale) - 1, int(
                    (some_box[1] + self.window_size) * cur_scale) - 1
                iou = self._IoU([Ax0, Ay0, Ax1, Ay1], [Bx0, By0, Bx1, By1])
                if iou > lamb:
                    faces[j] = 0
        ans = []
        for i in range(n):
            if faces[i]:
                ans.append(sorted_detected[i])
        return ans

    def draw_bb(self, faces, image, confidence_threshold, title):
        plt.title(title)
        im_draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for box in faces:
            if box[2] >= confidence_threshold:
                x, y = int(box[0]*box[3]), int(box[1]*box[3])
                size = int(self.window_size * box[3])
                rect = plt.Rectangle((y, x), size, size, fill=False, linewidth=1.5, edgecolor='g')
                plt.gca().add_patch(rect)
                plt.text(y, x, '{:.2f}'.format(box[2]), fontsize=10, color='white')
        plt.imshow(np.uint8(im_draw))
        plt.show()




















