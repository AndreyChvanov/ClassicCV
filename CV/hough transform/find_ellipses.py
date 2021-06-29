import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Ellipse

#paper: Xie, Yonghong, and Qiang Ji. “A new efficient ellipse detection method.” Pattern Recognition, 2002. Proceedings. 16th International Conference on. Vol. 2. IEEE, 2002 


def load_image(path):
    im = cv2.imread(path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im_rgb


def show_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(np.uint8(image))
    plt.show()


def convert_image_to_gray(image):
    im_grey = image.mean(axis=2)
    return np.float32(im_grey)


def get_edges(image):
    return cv2.Canny(image, 140, 205)


def ellipse_detector(image, votes_th, min_distance):
    image = cv2.GaussianBlur(image, (3, 3), 25)
    edges = get_edges(image)
    edges_coord = np.nonzero(edges)
    params = []
    x_edges, y_edges = edges_coord[1], edges_coord[0]
    vote = [0] * len(x_edges)
    for i in range(0, len(x_edges)):
        if vote[i] == 1:
            continue
        x1, y1 = x_edges[i], y_edges[i]
        for j in range(i + 1, len(x_edges)):
            if vote[j] == 1:
                continue
            x2, y2 = x_edges[j], y_edges[j]
            acc = np.zeros(max(image.shape[0], image.shape[1]))
            x0, y0 = (x1 + x2) / 2, (y1 + y2) / 2
            a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2
            if a < min_distance:
                continue
            alpha = np.arctan((y2 - y1) / (x2 - x1)) if x1 != x2 else np.pi/2
            cur_dots = [[]] * len(acc)
            for k in range(0, len(x_edges)):
                if vote[k] == 1:
                    continue
                if k == i or k == j:
                    continue
                x, y = x_edges[k], y_edges[k]
                d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                if d >= a or d < min_distance:
                    continue
                f1 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                f2 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                f = min(f1, f2)
                cos_t = (a ** 2 + d ** 2 - f ** 2) / (2 * a * d)
                if cos_t > 1:
                    cos_t = 1
                b_2 = (a ** 2 * d ** 2 * (1 - cos_t ** 2)) / (a ** 2 - d ** 2 * cos_t ** 2)
                b = int((np.sqrt(b_2)))
                if 0 < b < len(acc):
                    acc[b] += 1
                    cur_dots[b].append(k)
            candidate = acc.argmax()
            if acc[candidate] > votes_th:
                params.append([int(x0), int(y0), int(a), candidate, alpha])
                for dot in cur_dots[candidate]:
                    vote[dot] = 1
    return params


def draw_ellipse(image, params):
    plt.imshow(image)
    for i in range(len(params)):
        x0, y0, a, b, alpha = params[i]
        print(params[i])
        c = Ellipse((x0, y0), 2 * a, 2 * b, np.degrees(alpha), fill=False, edgecolor='r', linewidth=1.0)
        plt.gca().add_patch(c)

    plt.show()


image = load_image("ellipse_small.jpg")

params = ellipse_detector(image, 30, 30)
draw_ellipse(image, params)
