import mmcv
import numpy as np


def rotate_img(img):
    min_projection = None
    min_angle = 0
    for i in range(1, 180):
        img_rotated = mmcv.imrotate(
            img, i, auto_bound=True, interpolation='nearest')
        projection = np.count_nonzero(np.sum(img_rotated, axis=0) > 0)
        if min_projection is None:
            min_projection = projection
        else:
            if projection < min_projection:
                min_angle = i
                min_projection = projection
    img_rotated = mmcv.imrotate(
        img, min_angle, auto_bound=True, interpolation='nearest')
    poses = np.where(img_rotated > 0)
    img_rotated = img_rotated[min(poses[0]):max(poses[0]),
                              min(poses[1]):max(poses[1])]
    return img_rotated
