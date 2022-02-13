#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019 Ina (Zohra Rezgui & David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from .rect import Rect


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


def alignCrop(frame, bb, left_eye, right_eye, verbose=False):
    """
    Rotate image such as the eyes lie on a horizontal line

    Parameters
    ----------
    frame : numpy.ndarray (height,with, 3)
        RGB image data
    bb : (x1, y1, x2, y2)
    left_eye: x, y
    right_eye: x, y

    Returns
    -------
    rotated_frame : numpy.ndarray (height,with, 3)
        RGB rotated and cropped image data
    """
    w = int(bb[2] - bb[0])
    h = int(bb[3] - bb[1])

    angle = _angle_between_2_points(left_eye, right_eye)
    xc = (left_eye[0] + right_eye[0]) / 2
    yc = (left_eye[1] + right_eye[1]) / 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    M += np.array([[0, 0, -bb[0]], [0, 0, -bb[1]]])


    rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

    if verbose:
        print('after rotation & crop')
        plt.imshow(rotated_frame)
        plt.show()
    return rotated_frame



def preprocess_face(frame, detection, squarify, bbox_scale, face_alignment, output_shape, verbose=False):
    """
    Apply preprocessing pipeline to a detected face and returns the
    corresponding image with the following optional processings
    Parameters
    ----------
    frame : numpy nd.array
        RGB image data
    detection : a Detection instance containing a field 'bbox' of type Rect
        corresponding to the detected face bounding box
        if not None, the provided bounding box will be used, else the
        whole image will be used
    squarify: boolean
        if set to True, the bounding box will be extented to be the
        smallest square containing the bounding box. This avoids distortion
        if faces are resized in a latter stage
    bbox_scale: float
        resize the bounding box to consider larger (bbox_scale > 1) or
        smaller (bbox_scale < 1) areas around faces. If set to 1, the
        original bounding box is used
    face_alignment: instance of class defined in face_alignment.py or None
        if set to None, do nothing
        if set to an instance of class defined in face_alignment.py
        (for instance Dlib68FaceAlignment) face will be rotated based on the
        estimation of facial landmarks such the eyes lie on a horizontal line
    output_shape: (width, height) or None
        if not None, face will be resized to the provided output shape
    Returns
    -------
    frame: np.array RGB image data
    bbox: up-to-data bounding box after the proprocessing pipeline

    """

    (frame_h, frame_w, _) = frame.shape

    # if no bounding box is provided, use the whole image
    if detection is None:
        bbox = Rect(0, 0, frame_w, frame_h)
    elif isinstance(detection, Rect):
        bbox = detection
    else:
        bbox = detection.bbox

    if face_alignment is not None:
        left_eye, right_eye = face_alignment(frame, bbox)

    # if True, extend the bounding box to the smallest square containing
    # the orignal bounding box
    if squarify:
        bbox = bbox.square

    # perform bounding box scaling to march larger/smaller areas around
    # the detected face
    bbox = bbox.scale(bbox_scale)

    bbox = bbox.to_int()


    # performs face alignment based on facial landmark detection
    if face_alignment is not None:
        frame = alignCrop(frame, bbox, left_eye, right_eye, verbose=verbose)
    else:
        # crop image to the bounding box
	# TODO: replace by a wrap affine => management of out of frame
        frame = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

    # resize image to the required output shape
    if output_shape is not None:
        frame = cv2.resize(frame, output_shape)

    if verbose:
        print('resulting image')
        plt.imshow(frame)
        plt.show()

    return frame, bbox
