import numpy as np
import matplotlib.pyplot as plt
import cv2






def load_image(path):
    im = cv2.imread(path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return convert_image_to_gray(im_rgb)


def show_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(np.uint8(image))
    plt.show()


def convert_image_to_gray(image):
    im_grey = image.mean(axis=2)
    return np.float32(im_grey)

def get_edges(image):
    return cv2.Canny(image, 100, 200)

def get_cumulative_array_for_lines(image, opt_type):
    image = cv2.GaussianBlur(image, (3, 3), 25)
    edges = get_edges(image)
    tetta = np.degrees(np.arange(-np.pi / 2, np.pi+np.pi/180, np.pi / 180))
    A, B = edges.shape[0], edges.shape[1]
    tetta = np.round(tetta, 3)
    po = np.arange(0, np.sqrt(A ** 2 + B ** 2)+1, 1)
    cum = np.zeros((len(tetta), len(po)))
    edges_coord = np.nonzero(edges)
    x_edges, y_edges = edges_coord[0], edges_coord[1]
    cos = np.cos(np.radians(tetta))
    sin = np.sin(np.radians(tetta))
    if opt_type == 0:
        for i in range(len(x_edges)):
            x = x_edges[i]
            y = y_edges[i]
            for t_i in range(len(tetta)):
                p = x*cos[t_i]+y*sin[t_i]
                if 0 <= p <= np.sqrt(A ** 2 + B ** 2):
                    cum[t_i, int(p)] += 1
    if opt_type == 1:
        edges_indexes = np.arange(len(edges_coord[0]))
        np.random.shuffle(edges_indexes)
        x_edges = x_edges[edges_indexes]
        y_edges = y_edges[edges_indexes]
        for i in range(int(len(x_edges)*0.2)):
            x = x_edges[i]
            y = y_edges[i]
            for t_i in range(len(tetta)):
                p = x * cos[t_i] + y * sin[t_i]
                if 0 <= p <= np.sqrt(A ** 2 + B ** 2):
                    cum[t_i, int(p)] += 1
    if opt_type == 2:
        count_pairs = 10000
        for i in range(count_pairs):
            pairs = [np.random.randint(0, len(x_edges)), np.random.randint(0, len(x_edges))]
            x0, y0 = x_edges[pairs[0]], y_edges[pairs[0]]
            x1, y1 = x_edges[pairs[1]], y_edges[pairs[1]]
            k = (y1-y0)/(x1-x0) if (x1-x0) != 0 else 0
            b = y1-k*x1
            alpha = np.degrees(np.arctan(k))
            t = -90+alpha
            p = np.sin(np.radians(t))*b
            if 0 <= p <= np.sqrt(A ** 2 + B ** 2):
                t = np.round(t)
                cum[int(t+90), int(p)] += 1
    if opt_type == 3:
        gradX = cv2.Sobel(image.mean(axis=2), cv2.CV_64F, 1, 0, ksize=3)
        gradY = cv2.Sobel(image.mean(axis=2), cv2.CV_64F, 0, 1, ksize=3)
        directions = np.degrees(np.arctan2(gradY, gradX))
        for i in range(len(x_edges)):
            x = x_edges[i]
            y = y_edges[i]
            t = directions[x, y] - 90
            p = x*np.cos(np.radians(t))+y*np.sin(np.radians(t))
            if 0 <= p <= np.sqrt(A ** 2 + B ** 2):
                cum[int(t+90), int(p)] += 1
    return cv2.GaussianBlur(cum, (3,3), 25)


def get_cumulative_array_for_circle(image, opt_type):
    image = cv2.GaussianBlur(image, (5, 5), 25)
    edges = get_edges(image)
    a_max, b_max = image.shape[1], image.shape[0]
    r_max = np.sqrt(a_max**2 + b_max ** 2)
    a = np.arange(0, a_max, 1)
    b = np.arange(0, b_max, 1)
    r = np.arange(0, r_max, 1)
    cum = np.zeros((len(a), len(b), len(r)))
    edges_coord = np.nonzero(edges)
    x_edges, y_edges = edges_coord[0], edges_coord[1]
    if opt_type == 0:
        for i in range(len(x_edges)):
            x = x_edges[i]
            y = y_edges[i]
            for a_i, c_a in enumerate(a):
                for b_i, c_b in enumerate(b):
                    c_r = np.sqrt((x-c_a)**2 + (y-c_b) ** 2)
                    if 0 <= c_r <= r_max:
                        cum[a_i, b_i, int(c_r)] += 1
    if opt_type == 1:
        edges_indexes = np.arange(len(edges_coord[0]))
        np.random.shuffle(edges_indexes)
        x_edges = x_edges[edges_indexes]
        y_edges = y_edges[edges_indexes]
        for i in range(int(len(x_edges) * 0.2)):
            x = x_edges[i]
            y = y_edges[i]
            for a_i, c_a in enumerate(a):
                for b_i, c_b in enumerate(b):
                    c_r = np.sqrt((x-c_a)**2 + (y-c_b) ** 2)
                    if 0 <= c_r <= r_max:
                        cum[a_i, b_i, int(c_r)] += 1
    if opt_type == 2:
        count_pairs = 5000
        for i in range(count_pairs):
            pairs = [np.random.randint(0, len(x_edges)), np.random.randint(0, len(x_edges))]
            x0, y0 = x_edges[pairs[0]], y_edges[pairs[0]]
            x1, y1 = x_edges[pairs[1]], y_edges[pairs[1]]
            k = (y1-y0)/(x1-x0) if (x1-x0) != 0 else np.tan(np.pi/2)
            x_center = (x0+x1)/2
            y_center = (y0+y1)/2
            k_mid = -(1/k) if k !=0 else np.tan(np.pi/2)
            b_mid = y_center - k_mid*x_center
            for x in np.arange(0, a_max, 1):
                y = int(k_mid*x+b_mid)
                if y >= 0 and y < b_max:
                    radius = np.sqrt((x-x0)**2 + (y-y0)**2)
                    if 0 <= radius <= r_max:
                        cum[x, y, int(radius)] += 1
    if opt_type == 3:
        gradX = cv2.Sobel(image.mean(axis=2), cv2.CV_64F, 1, 0, ksize=3)
        gradY = cv2.Sobel(image.mean(axis=2), cv2.CV_64F, 0, 1, ksize=3)
        directions = np.arctan2(gradY, gradX)
        directions[directions < 0] += 2 * np.pi
        for i in range(len(x_edges)):
            x = x_edges[i]
            y = y_edges[i]
            k_grad = np.tan(directions[x, y])
            b_grad = y - k_grad*x
            for a_c in range(0, a_max, 1):
                b_c = int(a_c*k_grad + b_grad)
                if 0 <= b_c < b_max:
                    radius = np.sqrt((x-a_c)**2 + (y-b_c)**2)
                    if 0 <= radius <= r_max:
                        cum[a_c, b_c, int(radius)] += 1
    return cum



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


def draw_line(image, opt_type):
    cum_arr = get_cumulative_array_for_lines(image, opt_type)
    cum_nms = NMS_for_matrix(cum_arr, 30)
    show_image(image, "Исходное изображение")
    edges = get_edges(image)
    show_image(edges, "Границы")
    show_image(cum_arr,"Кумулятивный массив")
    show_image(cum_nms, "Кумулятивный массив после NMS")
    tetta = np.degrees(np.arange(-np.pi / 2, np.pi+np.pi/180, np.pi / 180))
    plt.imshow(np.uint8(image))
    tetta = np.round(tetta, 3)
    params = np.where(cum_nms > cum_nms.max()-40)
    for i in range(len(params[0])):
        p, t = params[1][i], np.radians(tetta[params[0][i]])
        x, y = [], []
        if np.cos(t) != 0 and 0 <= (p-(image.shape[1]-1) * np.sin(t))/np.cos(t) < image.shape[0]:
            x.append(int((p-(image.shape[1]-1) * np.sin(t))/np.cos(t)))
            y.append(image.shape[1]-1)
        if np.sin(t) != 0 and 0 <= (p - (image.shape[0]-1) * np.cos(t))/np.sin(t) < image.shape[1]:
            x.append(image.shape[0]-1)
            y.append(int((p - (image.shape[0]-1) * np.cos(t))/np.sin(t)))
        if np.cos(t) != 0 and 0 <= p/np.cos(t) < image.shape[0]:
            x.append(int(p/np.cos(t)))
            y.append(0)
        if np.sin(t) != 0 and 0 <= p/np.sin(t) < image.shape[1]:
            x.append(0)
            y.append(int(p/np.sin(t)))
        cv2.line(image, (y[0], x[0]), (y[1], x[1]), (np.random.randint(255), np.random.randint(255), np.random.randint(255)), 1)
    plt.imshow(image)
    plt.title("Линии")
    plt.show()


def draw_circle(image, opt_type):
    show_image(image, "Исходное изображение")
    edges = get_edges(image)
    show_image(edges, "Границы")
    cum_arr = get_cumulative_array_for_circle(image, opt_type)
    cum_nms = np.zeros_like(cum_arr)
    for i in range(cum_arr.shape[0]):
        cum_nms[i] = NMS_for_matrix(cum_arr[i], 70)
    for j in range(cum_arr.shape[1]):
        cum_nms[:, j] = NMS_for_matrix(cum_arr[:, j], 70)
    for k in range(cum_arr.shape[2]):
        cum_nms[:, :, k] = NMS_for_matrix(cum_arr[:, :, k], 70)
    params = np.where(cum_nms> cum_nms.max()-5)
    for i in range(len(params[0])):
        cv2.circle(image, (params[0][i], params[1][i]), params[2][i], (255,0,0),0)

    show_image(image, "Окружности")









                
            





image = load_image("coins_37.png")
image1= load_image("lines-rectangle.png")
opt_type_for_line = 0
opt_type_for_circle = 3



draw_circle(image, opt_type_for_circle)
#draw_line(image1, opt_type_for_line)






