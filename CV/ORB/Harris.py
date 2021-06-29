import numpy as np
import cv2
from Task6 import utils
import matplotlib.pyplot as plt





class HarrisCornerDetector():
    def __init__(self, k, image):
        self.k = k
        self.image = image
        Iy, Ix = utils.Sobel_Filter(image)
        Ix_2 = Ix * Ix
        Iy_2 = Iy * Iy
        IxIy = Ix * Iy
        d = np.array([[1, 4, 6, 4, 1]]) / 16
        W = np.dot(d.T, d)
        W = np.power(W, 2)
        self.Ix_2 = cv2.filter2D(Ix_2, ddepth=-1, kernel=W)
        self.Iy_2 = cv2.filter2D(Iy_2, ddepth=-1, kernel=W)
        self.IxIy = cv2.filter2D(IxIy, ddepth=-1, kernel=W)

    def filter(self, i, j):
        M = np.array(([self.Ix_2[i, j], self.IxIy[i, j]], [self.IxIy[i, j], self.Iy_2[i, j]]))
        R = np.linalg.det(M) - self.k * np.trace(M) ** 2
        return R

    def Detection(self, image):
        Iy, Ix = utils.Sobel_Filter(image)
        h, w = image.shape[0], image.shape[1]
        Ix_2 = Ix * Ix
        Iy_2 = Iy * Iy
        IxIy = Ix * Iy
        d = np.array([[1, 4, 6, 4, 1]]) / 16
        W = np.dot(d.T, d)
        W = np.power(W, 2)
        Ix_2 = cv2.filter2D(Ix_2, ddepth=-1, kernel=W)
        Iy_2 = cv2.filter2D(Iy_2, ddepth=-1, kernel=W)
        IxIy = cv2.filter2D(IxIy, ddepth=-1, kernel=W)
        R = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                M = np.array(([Ix_2[i, j], IxIy[i, j]], [IxIy[i, j], Iy_2[i, j]]))
                R[i, j] = np.linalg.det(M) - self.k*np.trace(M)**2
        return R

# image = utils.load_image("blox.jpg")
# detector = HarrisCornerDetector(0.04, image)
# R = detector.Detection(image)
# print(R.min(), R.max())
#
# #R = utils.NMS_for_matrix(R, 30)
# corners = np.where(R > 15000)
# for i in range(len(corners[0])):
#     cv2.circle(image, (corners[1][i], corners[0][i]), 2, (0, 255, 0), -1)
# utils.show_image(image, 1)








