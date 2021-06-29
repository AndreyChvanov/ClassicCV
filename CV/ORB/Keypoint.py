import  numpy as np



class KeyPoint:
    def __init__(self, coords, angle, scale, descriptor, r):
        self.coords = np.array(coords)
        self.angle = angle
        self.scale = scale
        self.descriptor = descriptor
        self.r = r

