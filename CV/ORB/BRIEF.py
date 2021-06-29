import numpy as np
from Task6 import utils





class BRIEF:
    def __init__(self):
        self.step_angle = 2*np.pi/30
        self.angles = np.arange(-np.pi, np.pi+self.step_angle, self.step_angle)
        self.descriptor_positions = utils.load_pairs("orb_descriptor_positions.txt")
        self.S = [utils.rotate(self.descriptor_positions, angle).astype(np.int) for angle in self.angles]

    def get_description(self, image, key_point):
        description = []
        x, y = key_point[0][0], key_point[0][1]
        orientation = key_point[2]
        min_angle = np.abs(self.angles-orientation)
        angle_index = np.argmin(min_angle)
        print(angle_index)
        S = self.S[angle_index]
        for i in range(0, S.shape[0]-1, 2):
            p1 = S[i] + np.array([x, y])
            p2 = S[i+1] + np.array([x, y])
            description.append(1 if image[int(p1[0]), int(p1[1])] < image[int(p2[0]), int(p2[1])] else 0)
            print(p1, p2, description[-1], image[int(p1[0]), int(p1[1])], image[int(p2[0]), int(p2[1])])
            print(int(p1[0]),int(p1[1]), int(p2[0]), int(p2[1]))
        return np.array(description)

















