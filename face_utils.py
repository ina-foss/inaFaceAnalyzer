import numpy as np
import cv2

def _rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def _extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)


def _extract_eye_center(shape, eye_indices):
    points = _extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def extract_left_eye_center(shape):
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    return _extract_eye_center(shape, LEFT_EYE_INDICES)


def extract_right_eye_center(shape):
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    return _extract_eye_center(shape, RIGHT_EYE_INDICES)


def _angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 != x2:
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))
    elif y2 > y1:
        return np.degrees(np.pi / 2)
    
    else:
        return np.degrees(-np.pi / 2)



def get_rotation_matrix(p1, p2):
    angle = _angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def crop_image(image, det):
    left, top, right, bottom = _rect_to_tuple(det)
    return image[top:bottom, left:right]


