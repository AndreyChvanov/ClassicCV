import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


fast_th = 10
harris_k = 0.04
points_after_harris = 512


def to_projective(coords):
    return np.concatenate([coords[:, :2].copy(), np.ones(coords.shape[0]).reshape(-1, 1)], axis=1)

def convert_image_to_gray(image):
    im_grey = image.mean(axis=2)
    return np.float32(im_grey)

def load_image(path):
    im = cv2.imread(path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return convert_image_to_gray(im_rgb)

def hamming_distance(des1, des2):
    return np.count_nonzero(des1 != des2)

def brute_force(q_key_points,t_key_points ):
    pairs_distance = np.zeros((len(q_key_points), len(t_key_points)))
    for i in range(pairs_distance.shape[0]):
        for j in range(pairs_distance.shape[1]):
            pairs_distance[i, j] = hamming_distance(q_key_points[i].descriptor, t_key_points[j].descriptor)
    matches = np.argmin(pairs_distance, axis=1)
    pairs = []
    for i in range(len(matches)):
        pairs.append([q_key_points[i], t_key_points[matches[i]]])
    return pairs

def test_Lowe(q_key_points, t_key_points, R_Lowe = 0.8):
    pairs_distance = np.zeros((len(q_key_points), len(t_key_points)))
    for i in range(pairs_distance.shape[0]):
        for j in range(pairs_distance.shape[1]):
            pairs_distance[i, j] = hamming_distance(q_key_points[i].descriptor, t_key_points[j].descriptor)
    matches = []
    for i in range(pairs_distance.shape[0]):
        des_t = np.argsort(pairs_distance[i].copy())[0:2]
        R = pairs_distance[i, des_t[0]]/pairs_distance[i, des_t[1]]
        if R < R_Lowe:
            matches.append([i, des_t[0]])
    return matches, pairs_distance

def cross_check(matches, pairs_distance):
    pairs = []
    for match in matches:
        q = match[0]
        t = match[1]
        closest_for_test = np.argmin(pairs_distance[:, t])
        if closest_for_test == q:
            pairs.append([q, t])
    return pairs


def prepare_matrix_for_affine(quary_p):
    h = 2*quary_p.shape[0]
    A = np.zeros((h, 6))
    A[:, 4][0:h:2] = 1
    A[:, 5][1:h:2] = 1
    A[:, 0][0:h:2] = quary_p[:, 0]
    A[:, 1][0:h:2] = quary_p[:, 1]
    A[:, 2][1:h:2] = quary_p[:, 0]
    A[:, 3][1:h:2] = quary_p[:, 1]
    return A

def solve_system_equations_affine(quary_p, test_p):
    A = np.zeros((6, 6))
    A[:, 4][0:6:2] = 1
    A[:, 5][1:6:2] = 1
    A[:, 0][0:6:2] = quary_p[:, 0]
    A[:, 1][0:6:2] = quary_p[:, 1]
    A[:, 2][1:6:2] = quary_p[:, 0]
    A[:, 3][1:6:2] = quary_p[:, 1]
    b = np.zeros(6)
    b[0:6:2] = test_p[:, 0]
    b[1:6:2] = test_p[:, 1]
    if np.linalg.det(A) == 0:
        return np.zeros(6)

    x = np.linalg.solve(A, b)
    return x


def RANSAC_for_affine_transfsorm(pairs, q_keypoints, t_keypoints, N=2000, proj_th=10):
    indexes = np.arange(len(pairs))
    np.random.shuffle(indexes)
    top_params = np.zeros(6)
    inlier_count = 0
    q_keypoints_coords = []
    t_keypoints_coords = []
    for pair in pairs:
        q, t = pair[0], pair[1]
        q_keypoints_coords.append(q_keypoints[q].coords*q_keypoints[q].scale)
        t_keypoints_coords.append(t_keypoints[t].coords*t_keypoints[t].scale)
    q_keypoints_coords = np.array(q_keypoints_coords)
    t_keypoints_coords = np.array(t_keypoints_coords)
    for n in range(N):
        cur_matches = indexes[0:3]
        cur_query_points = q_keypoints_coords[cur_matches]
        cur_test_points = t_keypoints_coords[cur_matches]
        x = solve_system_equations_affine(cur_query_points, cur_test_points)
        if x.all() == 0:
            continue
        transfsorm_points = np.dot(prepare_matrix_for_affine(q_keypoints_coords), x)
        transfsorm_points.resize(transfsorm_points.shape[0]//2, 2)
        distance = np.linalg.norm(transfsorm_points-t_keypoints_coords, axis=1)
        cur_count_inlier = len(np.where(distance <= proj_th)[0])
        if cur_count_inlier > inlier_count:
            inlier_count = cur_count_inlier
            top_params = x
        np.random.shuffle(indexes)
    print(inlier_count, top_params)
    transfsorm_points = np.dot(prepare_matrix_for_affine(q_keypoints_coords), top_params)
    transfsorm_points.resize(transfsorm_points.shape[0] // 2, 2)
    distance = np.linalg.norm(transfsorm_points - t_keypoints_coords, axis=1)
    inliers = np.where(distance <= proj_th)[0]
    A = prepare_matrix_for_affine(q_keypoints_coords[inliers])
    b = np.zeros(len(inliers)*2)
    b[0:len(b):2] = t_keypoints_coords[inliers][:, 0]
    b[1:len(b):2] = t_keypoints_coords[inliers][:, 1]
    result_params = np.linalg.inv(A.T @ A) @ A.T @ b
    print(result_params)
    return result_params


def prepare_matrix_for_homography(quary_p, test_p):
    h = 2 * quary_p.shape[0]
    A = np.zeros((h, 8))
    A[:, 2][0:h:2] = 1
    A[:, 5][1:h:2] = 1
    A[:, 0][0:h:2] = quary_p[:, 0]
    A[:, 1][0:h:2] = quary_p[:, 1]
    A[:, 3][1:h:2] = quary_p[:, 0]
    A[:, 4][1:h:2] = quary_p[:, 1]
    A[:, 6][0:h:2] = -quary_p[:, 0] * test_p[:, 0]
    A[:, 6][1:h:2] = -quary_p[:, 0] * test_p[:, 1]
    A[:, 7][0:h:2] = -quary_p[:, 1] * test_p[:, 0]
    A[:, 7][1:h:2] = -quary_p[:, 1] * test_p[:, 1]
    return A

def solve_system_equations_perspective(quary_p, test_p):
    A = prepare_matrix_for_homography(quary_p, test_p)
    h = 2*quary_p.shape[0]
    b = np.zeros(h)
    b[0:h:2] = test_p[:, 0]
    b[1:h:2] = test_p[:, 1]
    if np.linalg.det(A) == 0:
        return np.zeros(8)
    x = np.linalg.solve(A, b)
    return x


def RANSAC_for_perspective_transfsorm(pairs, q_keypoints, t_keypoints, N=2500, proj_th=3):
    indexes = np.arange(len(pairs))
    np.random.shuffle(indexes)
    top_params = np.zeros(8)
    inlier_count = 0
    q_keypoints_coords = []
    t_keypoints_coords = []
    for pair in pairs:
        q, t = pair[0], pair[1]
        q_keypoints_coords.append(q_keypoints[q].coords * q_keypoints[q].scale)
        t_keypoints_coords.append(t_keypoints[t].coords * t_keypoints[t].scale)
    q_keypoints_coords = np.array(q_keypoints_coords)
    t_keypoints_coords = np.array(t_keypoints_coords)
    for n in range(N):
        cur_matches = indexes[0:4]
        cur_query_points = q_keypoints_coords[cur_matches]
        cur_test_points = t_keypoints_coords[cur_matches]
        x = solve_system_equations_perspective(cur_query_points, cur_test_points)
        if x.all() == 0:
            continue
        proj_coords = to_projective(q_keypoints_coords)
        H = np.append(x, [1])
        H.resize(3, 3)
        transfsorm_points = np.dot(H, proj_coords.T)
        transfsorm_points = transfsorm_points.T
        transfsorm_points[:, 0] /= transfsorm_points[:, 2]
        transfsorm_points[:, 1] /= transfsorm_points[:, 2]
        transfsorm_points = transfsorm_points[:, :2]
        distance = np.linalg.norm(transfsorm_points - t_keypoints_coords, axis=1)
        cur_count_inlier = len(np.where(distance <= proj_th)[0])
        if cur_count_inlier > inlier_count:
            inlier_count = cur_count_inlier
            top_params = x
        np.random.shuffle(indexes)
    print(inlier_count)
    H = np.append(top_params, [1])
    H.resize(3, 3)
    proj_coords = to_projective(q_keypoints_coords)
    transfsorm_points = np.dot(H, proj_coords.T)
    transfsorm_points = transfsorm_points.T
    transfsorm_points[:, 0] /= transfsorm_points[:, 2]
    transfsorm_points[:, 1] /= transfsorm_points[:, 2]
    transfsorm_points = transfsorm_points[:, :2]
    distance = np.linalg.norm(transfsorm_points - t_keypoints_coords, axis=1)
    inliers = np.where(distance <= proj_th)[0]
    A = prepare_matrix_for_homography(q_keypoints_coords[inliers], t_keypoints_coords[inliers])
    b = np.zeros(len(inliers) * 2)
    b[0:len(b):2] = t_keypoints_coords[inliers][:, 0]
    b[1:len(b):2] = t_keypoints_coords[inliers][:, 1]
    result_params = np.linalg.inv(A.T @ A) @ A.T @ b
    print(result_params)
    return result_params


def show_result(q_key_points, t_key_points, t_type):
    query = load_image("box.png")
    h_q, w_q = query.shape[0], query.shape[1]
    test = load_image("box_in_scene.png")
    h_t, w_t = test.shape[0], test.shape[1]
    if t_type == "affine":
        pairs, pairs_distance = test_Lowe(q_key_points, t_key_points)
        print(len(pairs))
        pairs = cross_check(pairs, pairs_distance)
        print(len(pairs))
        affine_params = RANSAC_for_affine_transfsorm(pairs, q_key_points, t_key_points)
        result_image = np.zeros((h_t, w_q + w_t))
        result_image[0:h_q, 0:w_q] = query
        result_image[0:h_t, w_q:w_q + w_t] = test
        plt.figure(figsize=(17, 10))
        plt.imshow(result_image, cmap="gray")
        for i, p in enumerate(pairs):
            p0, p1 = p[0], p[1]
            p0 = q_key_points[p0]
            p1 = t_key_points[p1]
            scale0 = p0.scale
            x0, y0 = p0.coords * scale0
            scale1 = p1.scale
            x1, y1 = p1.coords * scale1
            c = plt.Circle((y0, x0), 3 * scale0, fill=False, edgecolor="r", linewidth=1)
            plt.gca().add_patch(c)
            c1 = plt.Circle((y1 + w_q, x1), 3 * scale1, fill=False, edgecolor="r", linewidth=1)
            plt.gca().add_patch(c1)
            plt.plot((y1 + w_q, y0), (x1, x0))
        bb_q = np.array(
            [[0, 0], [0, query.shape[1] - 1], [query.shape[0] - 1, 0], [query.shape[0] - 1, query.shape[1] - 1]])
        bb_t = np.dot(prepare_matrix_for_affine(bb_q), affine_params)
        bb_t.resize(bb_t.shape[0] // 2, 2)
        x1, y1 = bb_t[0][0], bb_t[0][1]
        x2, y2 = bb_t[1][0], bb_t[1][1]
        x3, y3 = bb_t[2][0], bb_t[2][1]
        x4, y4 = bb_t[3][0], bb_t[3][1]
        plt.plot((y1 + w_q, y2 + w_q), (x1, x2), 'y', linewidth=5)
        plt.plot((y4 + w_q, y2 + w_q), (x4, x2), 'y', linewidth=5)
        plt.plot((y4 + w_q, y3 + w_q), (x4, x3), 'y', linewidth=5)
        plt.plot((y1 + w_q, y3 + w_q), (x1, x3), 'y', linewidth=5)
        plt.title("Lowe, cross check, affine")
        print(bb_t)
        plt.show()
    if t_type == "perspective":
        pairs, pairs_distance = test_Lowe(q_key_points, t_key_points)
        print(len(pairs))
        pairs = cross_check(pairs, pairs_distance)
        print(len(pairs))
        perspective_params = RANSAC_for_perspective_transfsorm(pairs, q_key_points, t_key_points)
        H = np.append(perspective_params, [1])
        H.resize(3, 3)
        print(H)
        result_image = np.zeros((h_t, w_q + w_t))
        result_image[0:h_q, 0:w_q] = query
        result_image[0:h_t, w_q:w_q + w_t] = test
        plt.figure(figsize=(17, 10))
        plt.imshow(result_image, cmap="gray")
        for i, p in enumerate(pairs):
            p0, p1 = p[0], p[1]
            p0 = q_key_points[p0]
            p1 = t_key_points[p1]
            scale0 = p0.scale
            x0, y0 = p0.coords * scale0
            scale1 = p1.scale
            x1, y1 = p1.coords * scale1
            c = plt.Circle((y0, x0), 3 * scale0, fill=False, edgecolor="r", linewidth=1)
            plt.gca().add_patch(c)
            c1 = plt.Circle((y1 + w_q, x1), 3 * scale1, fill=False, edgecolor="r", linewidth=1)
            plt.gca().add_patch(c1)
            plt.plot((y1 + w_q, y0), (x1, x0))
        bb_q = np.array(
            [[0, 0], [0, query.shape[1] - 1], [query.shape[0] - 1, 0], [query.shape[0] - 1, query.shape[1] - 1]])
        bb_q = to_projective(bb_q)
        bb_t = np.dot(H, bb_q.T)
        bb_t = bb_t.T
        bb_t[:, 0] /= bb_t[:, 2]
        bb_t[:, 1] /= bb_t[:, 2]
        bb_t = bb_t[:, :2]
        x1, y1 = bb_t[0][0], bb_t[0][1]
        x2, y2 = bb_t[1][0], bb_t[1][1]
        x3, y3 = bb_t[2][0], bb_t[2][1]
        x4, y4 = bb_t[3][0], bb_t[3][1]
        plt.plot((y1 + w_q, y2 + w_q), (x1, x2), 'y', linewidth=5)
        plt.plot((y4 + w_q, y2 + w_q), (x4, x2), 'y', linewidth=5)
        plt.plot((y4 + w_q, y3 + w_q), (x4, x3), 'y', linewidth=5)
        plt.plot((y1 + w_q, y3 + w_q), (x1, x3), 'y', linewidth=5)
        print(bb_t)
        plt.title("Lowe, cross check, perspective")
        plt.show()
































# orb = detector.ORB(fast_th, harris_k, points_after_harris)
# q_key_points = orb.detection(query)
# t_key_points = orb.detection(test)
# print(len(q_key_points))
# with open("query_keypoints1488.pickle", 'wb') as f:
#     pickle.dump(q_key_points, f)
# with open("test_keypoints1488.pickle", 'wb') as f:
#     pickle.dump(t_key_points, f)

with open("query_keypoints.pickle", 'rb') as f:
    q_key_points = pickle.load(f)
with open("test_keypoints.pickle", 'rb') as f:
    t_key_points = pickle.load(f)



show_result(q_key_points, t_key_points, "affine")
show_result(q_key_points, t_key_points, "perspective")



# detector.show_result(q_key_points, query)
# detector.show_result(t_key_points, test)



