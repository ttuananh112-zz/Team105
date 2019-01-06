import cv2, numpy as np


def cal_error(layer, point1, point2, point3):
    left_point = []
    right_point = []

    for i in range(int(point3[0]), layer.shape[0]):

        # left half of triangle
        for j in range(int(point3[1])):
            if (i - point1[0]) / (point3[0] - point1[0]) - (j - point1[1]) / (point3[1] - point1[1]) < 0:
                for k in range(j, int(point3[1])):
                    left_point.append(layer[i, k])
                break

        # right half of triangle
        for j in reversed(range(int(point3[1]), layer.shape[1])):
            if (i - point2[0]) / (point3[0] - point2[0]) - (j - point2[1]) / (point3[1] - point2[1]) < 0:
                for k in reversed(range(j, int(point3[1]) + 1)):
                    right_point.append(layer[i, k])
                break

    point_in_triangle = left_point + right_point
    average = np.mean(point_in_triangle, axis=0)
    error = ((point_in_triangle - average) ** 2).mean()
    return error


def centroid(layer, drop_rate=0.6, shrink_rate=0.25, delta=0.):
    """
    :param layer: image of size (H.W.C)
    :param drop_rate: the propotion of the upper part of image to be cropped
    :param shrink_rate: image will be resized by this rate for better performance
    :param delta: the bigger data is, the more likely the function will return the middle point of view regardless error calculated
    :return: 
    """
    top = layer.shape[0] * drop_rate
    error = float('inf')
    point1 = np.array([layer.shape[0] - 1, 0])  # left
    point2 = np.array([layer.shape[0] - 1, layer.shape[1] - 1])  # right
    result = layer.shape[1] / 2

    for i in range(layer.shape[1] / 16, layer.shape[1], layer.shape[1] / 8):
        point3 = np.array([top, i])
        err = cal_error(cv2.resize(layer, (0, 0), fx=shrink_rate, fy=shrink_rate), point1 * shrink_rate,
                        point2 * shrink_rate, point3 * shrink_rate)
        if (err < error):
            error = err
            result = i

    middle_error = cal_error(cv2.resize(layer, (0, 0), fx=shrink_rate, fy=shrink_rate), point1 * shrink_rate, point2 * shrink_rate, np.array([top, layer.shape[1] / 2]) * shrink_rate)
    if (middle_error - error < delta):
        result = layer.shape[1] / 2

    cen = np.mean((point1, point2, np.array([top, result])), 0)
    # # print cen
    # point1 = (point1[1], point1[0])
    # point2 = (point2[1], point2[0])
    # point3 = (result, int(top))
    # cv2.line(layer, point1, point3, (255, 0, 0), 3)
    # cv2.line(layer, point2, point3, (0, 255, 0), 3)
    # cv2.circle(layer, (int(cen[1]), int(cen[0])), 5, (0, 0, 255), -1, )
    # cv2.imshow('.', layer)
    # cv2.waitKey(0)
    return int(cen[0]), int(cen[1])   #y x

# for i in range(1, 10):
#     dir = '../other imgs/thr' + str(i) + '.png'
#     img = np.array(cv2.imread(dir))
#     print centroid(img)
#     print
