from Task6 import ORB as detector
from Task6 import utils
import pickle



fast_th = 10
harris_k = 0.04
points_after_harris = 100


orb = detector.ORB(fast_th, harris_k, points_after_harris)
image = utils.load_image("blox.jpg")

kp = orb.detection(image.copy())

with open("keypoints.pickle", 'wb') as f:
    pickle.dump(kp, f)



detector.show_result(kp, image)