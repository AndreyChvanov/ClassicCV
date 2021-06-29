import numpy as np
import matplotlib.pyplot as plt
import cv2

def convert_image_to_gray(image):
    im_grey = image.mean(axis=2)
    return np.float32(im_grey)

def load_image(path):
    im = cv2.imread(path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return convert_image_to_gray(im_rgb)

def show_image(image, title):
    plt.figure()
    plt.title(f'th = {title}')
    plt.imshow(np.uint8(image), cmap='gray')
    plt.show()

def partial_derivatives(first_frame, second_frame):
    dx = np.array([[-1, 8, 0, -8, 1]])/12
    dy = dx.T
    Ix = cv2.filter2D(first_frame, ddepth=-1, kernel=dx)
    Iy = cv2.filter2D(first_frame, ddepth=-1, kernel=dy)
    It = second_frame - first_frame
    d = np.array([[1, 4, 6, 4, 1]]) / 16
    W = np.dot(d.T, d)
    W = np.power(W, 2)
    Ix_2 = Ix * Ix
    Iy_2 = Iy * Iy
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It
    Ix_2 = cv2.filter2D(Ix_2, ddepth=-1, kernel=W)
    Iy_2 = cv2.filter2D(Iy_2, ddepth=-1, kernel=W)
    IxIy = cv2.filter2D(IxIy, ddepth=-1, kernel=W)
    IxIt = cv2.filter2D(IxIt, ddepth=-1, kernel=W)
    IyIt = cv2.filter2D(IyIt, ddepth=-1, kernel=W)
    return Ix_2, Iy_2, IxIy, IxIt, IyIt, Ix, Iy, It

def OpticalFlow(first_frame, second_frame, threshold):
    first_frame = cv2.GaussianBlur(first_frame, (3, 3), 1.5)
    second_frame = cv2.GaussianBlur(second_frame, (3, 3), 1.5)
    vectors = []
    h, w = first_frame.shape[0], second_frame.shape[1]
    Ix_2, Iy_2, IxIy, IxIt, IyIt, Ix, Iy, It = partial_derivatives(first_frame, second_frame)
    for i in range(0, h, 1):
        for j in range(0, w, 1):
            A = np.array(([Ix_2[i, j], IxIy[i, j]], [IxIy[i, j], Iy_2[i, j]]))
            lamb1 = (A[0, 0] + A[1, 1])/2 + (np.sqrt(4*A[0, 1] ** 2 + (A[0, 0]-A[1, 1]) ** 2))/2
            lamb2 = (A[0, 0] + A[1, 1])/2 - (np.sqrt(4*A[0, 1] ** 2 + (A[0, 0]-A[1, 1]) ** 2))/2
            if lamb1 >= threshold and lamb2 >= threshold:
                B = np.array([[-IxIt[i, j], -IyIt[i, j]]]).T
                v = np.dot(np.linalg.inv(A), B)
                vectors.append([i, j, v])
            if lamb1 >= threshold and lamb2 < threshold:
                gradient = np.array([Ix[i, j], Iy[i, j]])
                v = -It[i, j]/np.linalg.norm(gradient) * gradient/np.linalg.norm(gradient)
                vectors.append([i, j, v])
            if lamb1 < threshold and lamb2 < threshold:
                vectors.append([i, j, (0, 0)])
    opt_flow = np.zeros_like(first_frame, dtype=np.float64)
    for vec in vectors:
        opt_flow[vec[0], vec[1]] = np.linalg.norm(np.array([vec[2][0], vec[2][1]]))
    show_image(opt_flow, threshold)


    

frame1 = load_image("tennis454.jpg")
frame2 = load_image("tennis455.jpg")
OpticalFlow(frame1, frame2, 0.0039)
