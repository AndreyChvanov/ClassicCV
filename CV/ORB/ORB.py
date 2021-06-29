import numpy as np
import cv2
from Task6 import BRIEF
from Task6 import FAST
from Task6 import Keypoint
from Task6 import utils
import matplotlib.pyplot as plt


def show_result(kp, image):
    plt.figure(figsize=(17, 10))
    plt.imshow(image, cmap="gray")
    for i, p in enumerate(kp):
        angle = p.angle
        scale = p.scale
        x, y = p.coords * scale
        c = plt.Circle((y, x), 3 * scale, fill=False, edgecolor="b", linewidth=0.8)
        plt.gca().add_patch(c)
        plt.plot([y, (3 * scale) * np.cos(angle) + y], [x, (3 * scale) * np.sin(angle) + x], 'r')
    plt.show()


class ORB:
    def __init__(self, fast_threshold, harris_k, N):
        self.pyramid_lvl = 8
        self.scale_step = 1.2
        self.fast_th = fast_threshold
        self.k = harris_k
        self.n = N


    def detection(self, image):
        orb_key_points = []
        brief = BRIEF.BRIEF()

        current_scale = 1
        h, w = image.shape[0], image.shape[1]
        for lvl in range(self.pyramid_lvl):
            print("cur shape", h, w)
            fast = FAST.FAST(self.fast_th, self.n)
            key_points_fast = np.where(fast.Detection(image.copy()) > 0)
            key_points = fast.filter_use_Harris(key_points_fast, self.k)
            angles = fast.get_orientation(key_points)
            for i in range(len(angles)):
                key_points[i].append(angles[i])
            img_blur = cv2.GaussianBlur(image, (3, 3), 1.5)
            key_points_descriptors = [brief.get_description(img_blur, kp) for kp in key_points]
            for i in range(len(key_points)):
                key_point = Keypoint.KeyPoint(key_points[i][0], key_points[i][2], current_scale, key_points_descriptors[i], key_points[i][1])
                orb_key_points.append(key_point)
            image = cv2.resize(img_blur, (int(w / self.scale_step), int(h / self.scale_step)))
            h, w = image.shape[0], image.shape[1]
            current_scale = current_scale * self.scale_step
        return orb_key_points


# orb = ORB(10, 0.04, 100)
# image = utils.load_image("blox.jpg")
#
# kp = orb.detection(image.copy())
#
# show_result(kp, image)




