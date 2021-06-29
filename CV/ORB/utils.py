import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_image_to_gray(image):
    im_grey = image.mean(axis=2)
    return np.float32(im_grey)

def load_image(path):
    im = cv2.imread(path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return convert_image_to_gray(im_rgb)


def show_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(np.uint8(image))
    plt.show()


def NMS_for_matrix(matrix, n):
    w, h = matrix.shape[0], matrix.shape[1]
    if matrix.shape[0] % (n+1) != 0:
        zeros = np.zeros((n+1-matrix.shape[0]%(n+1), matrix.shape[1]))
        matrix = np.concatenate((matrix,zeros), axis=0)
    if matrix.shape[1] % (n+1) != 0:
        zeros = np.zeros((matrix.shape[0], n+1-matrix.shape[1]%(n+1)))
        matrix = np.concatenate((matrix, zeros), axis=1)
    blocks = np.zeros_like(matrix)
    max_el = []
    local_max = []
    for i in range(0, matrix.shape[0] - n, n + 1):
        for j in range(0, matrix.shape[1] - n, n + 1):
            m = matrix[i:i + n + 1, j:j + n + 1]
            cur_block = blocks[i:i + n + 1, j:j + n + 1]
            cur_block[0] = m[0]
            cur_block[n] = m[n]
            for c in range(1, (n+1)//2):
                cur_block[c] = np.maximum(m[c], blocks[i:i + n + 1, j:j + n + 1][c])
                cur_block[n-c] = np.maximum(m[n-c], blocks[i:i + n + 1, j:j + n + 1][n-c + 1])
            max_l = np.maximum(m[(n+1) // 2], cur_block[(n+1) // 2 - 1])
            cur_block[(n+1) // 2] = np.maximum(max_l, cur_block[(n+1) // 2 + 1])
            blocks[i:i + n + 1, j:j + n + 1] = cur_block
            max_ind = np.argmax(m)
            max_x, max_y = max_ind // (n + 1), max_ind % (n + 1)
            max_el.append([max_x, max_y, i, j])
    for el in max_el:
        x, y, i, j = el[0], el[1], el[2], el[3]
        x, y = x + i, y + j
        top_x, top_l_y = max(x - n, 0), max(y - n, 0)
        bot_x, bot_y = min(x + n, matrix.shape[0] - 1), min(y + n, matrix.shape[1] - 1)
        center = i + n // 2
        if top_l_y < j:
            center_line_left = blocks[center][top_l_y:j]
            left_max = center_line_left.max()
            if matrix[x, y] < left_max:
                continue
        if j + n + 1 <= bot_y:
            center_line_right = blocks[center][j + n + 1:bot_y + 1]
            right_max = center_line_right.max()
            if matrix[x, y] < right_max:
                continue
        center_up = i - n // 2 - 1
        if 0 < center_up < top_x:
            up_line = blocks[top_x][top_l_y:bot_y + 1]
            up_max = up_line.max()
            if matrix[x, y] < up_max:
                continue
        if top_x <= center_up:
            line1 = blocks[center_up + 1][top_l_y:bot_y + 1]
            m = matrix[top_x:center_up + 1, top_l_y:bot_y + 1]
            max1 = np.max(line1)
            max2 = np.max(m)
            if matrix[x, y] < max1 or matrix[x, y] < max2:
                continue
        center_down = i + n // 2 + 1 + n
        if bot_x < center_down < matrix.shape[0]:
            down_line = blocks[bot_x][top_l_y:bot_y + 1]
            down_max = down_line.max()
            if matrix[x, y] < down_max:
                continue
        if bot_x >= center_down < matrix.shape[0]:
            line1 = blocks[center_down - 1][top_l_y:bot_y + 1]
            m = matrix[center_down:center_down + 1, top_l_y:bot_y + 1]
            max1 = np.max(line1)
            max2 = np.max(m)
            if matrix[x, y] < max1 or matrix[x, y] < max2:
                continue
        local_max.append([x, y, matrix[x,y]])
    nms = np.zeros_like(matrix)
    for lm in local_max:
        nms[lm[0], lm[1]] = lm[2]
    return nms[:w, :h]


def Sobel_Filter(image):
    sobel_1 = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    sobel_2 = sobel_1.T
    im_sobel_1 = cv2.filter2D(image, ddepth=-1, kernel=sobel_1)
    im_sobel_2 = cv2.filter2D(image, ddepth=-1, kernel=sobel_2)
    return im_sobel_1, im_sobel_2

def circle(r, xc, yc):
    points = []
    x = 0
    y = r
    d = 1 - 2*r
    while y >= 0:
        points.append([xc+x, yc+y])
        points.append([xc+x, yc-y])
        points.append([xc-x, yc+y])
        points.append([xc-x, yc-y])
        error = 2 * (d+y) - 1
        if d < 0 and error <= 0:
            x = x + 1
            d = d + 2*x + 1
            continue
        else:
            y = y - 1
            d = d - 2 * y + 1
            continue
        x = x + 1
        y = y - 1
        d = d + 2 * (x - y)
    return np.array(points)

def get_circle(x, y):
    points_circle = []
    points_circle.append([x-3, y])
    points_circle.append([x-3, y+1])
    points_circle.append([x-2, y+2])
    points_circle.append([x-1, y+3])
    points_circle.append([x, y+3])
    points_circle.append([x+1, y+3])
    points_circle.append([x+2, y+2])
    points_circle.append([x+3, y+1])
    points_circle.append([x+3, y])
    points_circle.append([x+3, y-1])
    points_circle.append([x+2, y-2])
    points_circle.append([x+1, y-3])
    points_circle.append([x, y-3])
    points_circle.append([x-1, y-3])
    points_circle.append([x-2, y-2])
    points_circle.append([x-3, y-1])
    return np.array(points_circle)



def load_pairs(path):
    pairs_coords = []
    with open(path) as f:
        p = f.read()
    p = p.split("\n")
    for pairs in p:
        pairs = pairs.split()
        x1, y1, x2, y2 = float(pairs[0]), float(pairs[1]), float(pairs[2]), float(pairs[3])
        pairs_coords.append([int(x1), int(y1)])
        pairs_coords.append([int(x2), int(y2)])
    return np.array(pairs_coords)



def rotate(points, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotate_points = np.dot(points, R)
    return rotate_points[:, :2].copy()


