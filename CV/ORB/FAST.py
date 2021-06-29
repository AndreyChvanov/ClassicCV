import numpy as np
import cv2
from Task6 import utils
from Task6 import Harris
import matplotlib.pyplot as plt


class FAST:
    def __init__(self, threshold, count_after_harris):
        self.radius = 3
        self.circle_points = utils.get_circle
        self.image = None
        self.t = threshold
        self.patch_size = 31
        self.n = count_after_harris
        self.harris_detector = None

    def fast_check_candidate(self, circle_points, x, y):
        count = 0
        for i in [0, 4, 8, 12]:
            if self.image[x, y] + self.t < self.image[circle_points[i][0], circle_points[i][1]]:
                count += 1
        if count < 3:
            count = 0
            for i in [0, 4, 8, 12]:
                if self.image[x, y] - self.t > self.image[circle_points[i][0], circle_points[i][1]]:
                    count += 1
            if count < 3:
                return False
        return True


    def isCorner(self, circle_points, x, y):
        check = self.fast_check_candidate(circle_points, x, y)
        if not check:
            return False, -1
        intensity = [self.image[c[0], c[1]] for c in circle_points]
        for i in range(0, len(intensity)):
            count_b = 0
            count_d = 0
            for j in range(0, 12):
                if self.image[x, y] + self.t < intensity[(i + j) % len(intensity)]:
                    count_b += 1
                elif self.image[x, y] - self.t > intensity[(i + j) % len(intensity)]:
                    count_d += 1
                else:
                    break
            if count_b >= 12:
                return True, 1
            if count_d >= 12:
                return True, 0
        return False, -1


    def Detection(self, image):
        arr = np.zeros_like(image)
        h, w = image.shape[0], image.shape[1]
        self.image = image
        ps = int(np.sqrt(self.patch_size ** 2 + self.patch_size**2)/2)
        circle = self.circle_points(0,0)
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                cur_circle = np.array([i, j]) + circle
                is_corner, is_brighter = self.isCorner(cur_circle, i, j)
                if is_corner:
                    if i + ps >= self.image.shape[0] or i - ps < 0 \
                            or j + ps >= self.image.shape[1] or j - ps < 0:
                        continue
                    intensity = np.array([self.image[c[0], c[1]] for c in cur_circle])
                    if is_brighter == 1:
                        count = len(np.where(intensity > self.image[i, j] + self.t)[0])
                        arr[i, j] = count
                    if is_brighter == 0:
                        count = len(np.where(intensity < self.image[i, j] - self.t)[0])
                        arr[i, j] = count
        return arr

    def get_area_for_calc_orientation(self):
        points = []
        x_coords = np.arange(-self.patch_size, self.patch_size + 1, 1)
        y_coords = np.arange(-self.patch_size, self.patch_size + 1, 1)
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                if x_coords[i] ** 2 + y_coords[j] ** 2 <= self.patch_size ** 2:
                    points.append([x_coords[i], y_coords[j]])
        return points


    def get_orientation(self, key_points):
        key_points_angles = []
        points = self.get_area_for_calc_orientation()
        for kp in key_points:
            x, y = kp[0][0], kp[0][1]
            m_01 = 0
            m_10 = 0
            for point in points:
                if 0 <= point[0] + x < self.image.shape[0] and 0 <= point[1] + y < self.image.shape[1]:
                    m_01 = m_01 + point[1] * self.image[point[0] + x, point[1] + y]
                    m_10 = m_10 + point[0] * self.image[point[0] + x, point[1] + y]
            tetta = np.arctan2(m_01, m_10)
            key_points_angles.append(tetta)
        return key_points_angles

    def filter_use_Harris(self, key_points, k):
        self.harris_detector = Harris.HarrisCornerDetector(k, self.image)
        R = np.zeros(len(key_points[0]))
        for i in range(len(key_points[0])):
            x, y = key_points[0][i], key_points[1][i]
            R[i] = self.harris_detector.filter(x, y)
        mask = R.argsort()[-self.n:]
        key_points_after_harris = []
        for index in mask:
            key_points_after_harris.append([(key_points[0][index], key_points[1][index]), R[index]])
        return key_points_after_harris

# fast = FAST(10, 512)
# image = utils.load_image("house.tif")
# R = fast.Detection(image.copy())
# corners = np.where(R > 0)
# print(len(corners[0]))
# # utils.show_image(R, 1)
# # for i in range(len(corners[0])):
# #     cv2.circle(image, (corners[1][i], corners[0][i]), 1, (0, 0, 255), -1)
# # utils.show_image(image, 1)
# kp = fast.filter_use_Harris(corners)
# angles = fast.get_orientation(kp)
#
# plt.figure(figsize=(17, 10))
# plt.imshow(image, cmap='gray')
# for i, p in enumerate(kp):
#     x, y = p[0][0], p[0][1]
#     c = plt.Circle((y, x), 3, fill=False, edgecolor="g", linewidth=0.8)
#     plt.gca().add_patch(c)
#     angle = angles[i]
#     print(x, y, angle)
#     plt.plot([y, 3*np.sin(angle)+y], [x, 3*np.cos(angle)+x], 'r')
# plt.show()
