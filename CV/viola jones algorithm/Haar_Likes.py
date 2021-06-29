import numpy as np



class Haar_Like:
    def __init__(self):
        self.pattern_types = ["a", "b", "c", "d", "e"]

    def to_integral_image(self, image):
        integral_image = np.cumsum(image, axis=0).cumsum(axis=1)
        return integral_image

    def get_value_in_integral_image(self, integral_im, x, y):
        A = 0 if x[0] == 0 or y[0] == 0 else integral_im[x[0] - 1, y[0] - 1]
        B = 0 if y[0] == 0 else integral_im[x[1], y[0] - 1]
        C = 0 if x[0] == 0 else integral_im[x[0] - 1, y[1]]
        D = integral_im[x[1], y[1]]
        return D + A - (B + C)


    def get_features_by_pattern(self, integral_image, pattern_type):
        features = []
        features_param = []
        if pattern_type == self.pattern_types[0]:
            for i in range(integral_image.shape[0]):
                for j in range(integral_image.shape[1]):
                    w = 1
                    while j + 2 * w - 1 < integral_image.shape[0]:
                        h = 1
                        while i + h - 1 < integral_image.shape[0]:
                            sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                            sum_2 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + w, j + 2 * w - 1])
                            features.append(sum_1 - sum_2)
                            features_param.append([i, j, h, w, pattern_type])
                            h += 1
                        w += 1

        if pattern_type == self.pattern_types[1]:
            for i in range(integral_image.shape[0]):
                for j in range(integral_image.shape[1]):
                    w = 1
                    while j + 3 * w - 1 < integral_image.shape[0]:
                        h = 1
                        while i + h - 1 < integral_image.shape[0]:
                            sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                            sum_2 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + w, j + 2 * w - 1])
                            sum_3 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + 2 * w, j + 3 * w - 1])
                            features.append(sum_1 - sum_2 + sum_3)
                            features_param.append([i, j, h, w, pattern_type])
                            h += 1
                        w += 1

        if pattern_type == self.pattern_types[2]:
            for i in range(integral_image.shape[0]):
                for j in range(integral_image.shape[1]):
                    w = 1
                    while j + w - 1 < integral_image.shape[0]:
                        h = 1
                        while i + 2 * h - 1 < integral_image.shape[0]:
                            sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                            sum_2 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1], [j, j + w - 1])
                            features.append(sum_1 - sum_2)
                            features_param.append([i, j, h, w, pattern_type])
                            h += 1
                        w += 1

        if pattern_type == self.pattern_types[3]:
            for i in range(integral_image.shape[0]):
                for j in range(integral_image.shape[1]):
                    w = 1
                    while j + w - 1 < integral_image.shape[0]:
                        h = 1
                        while i + 3 * h - 1 < integral_image.shape[1]:
                            sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                            sum_2 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1], [j, j + w - 1])
                            sum_3 = self.get_value_in_integral_image(integral_image, [i + 2 * h, i + 3 * h - 1], [j, j + w - 1])
                            features.append(sum_1 - sum_2 + sum_3)
                            features_param.append([i, j, h, w, pattern_type])
                            h += 1
                        w += 1

        if pattern_type == self.pattern_types[4]:
            for i in range(integral_image.shape[0]):
                for j in range(integral_image.shape[1]):
                    w = 1
                    while j + 2 * w - 1 < integral_image.shape[0]:
                        h = 1
                        while i + 2 * h - 1 < integral_image.shape[1]:
                            sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                            sum_2 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1], [j, j + w - 1])
                            sum_3 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + w, j + 2 * w - 1])
                            sum_4 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1],
                                                                [j + w, j + 2 * w - 1])
                            features.append(sum_1 - sum_2 - sum_3 + sum_4)
                            features_param.append([i, j, h, w, pattern_type])
                            h += 1
                        w += 1
        return features

    def get_all_features_by_image(self, im):
        all_features = []
        im = (im - im.mean()) / im.std()
        ii = self.to_integral_image(im)
        for pattern_type in self.pattern_types:
            features = self.get_features_by_pattern(ii, pattern_type)
            all_features.extend(np.array(features))
        return np.array(all_features)

    def get_chose_feature(self, features_list, features_params, image):
        image = (image-image.mean())/image.std() if image.std() > 1 else image
        integral_image = self.to_integral_image(image)
        features = dict()
        for feature in features_list:
            i, j, h, w, pattern_type = features_params[feature]
            i, j, h, w = int(i), int(j), int(h), int(w)
            if pattern_type == "a":
                sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                sum_2 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + w, j + 2 * w - 1])
                features[feature] = (sum_1 - sum_2)
            if pattern_type == "b":
                sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                sum_2 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + w, j + 2 * w - 1])
                sum_3 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + 2 * w, j + 3 * w - 1])
                features[feature] = (sum_1 - sum_2 + sum_3)
            if pattern_type == "c":
                sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                sum_2 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1], [j, j + w - 1])
                features[feature] = (sum_1 - sum_2)
            if pattern_type == "d":
                sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                sum_2 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1], [j, j + w - 1])
                sum_3 = self.get_value_in_integral_image(integral_image, [i + 2 * h, i + 3 * h - 1], [j, j + w - 1])
                features[feature] = (sum_1 - sum_2 + sum_3)
            if pattern_type == "e":
                sum_1 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j, j + w - 1])
                sum_2 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1], [j, j + w - 1])
                sum_3 = self.get_value_in_integral_image(integral_image, [i, i + h - 1], [j + w, j + 2 * w - 1])
                sum_4 = self.get_value_in_integral_image(integral_image, [i + h, i + 2 * h - 1],
                                                         [j + w, j + 2 * w - 1])
                features[feature] = (sum_1 - sum_2 - sum_3 + sum_4)
        return features








