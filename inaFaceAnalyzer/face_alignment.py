#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan & Thomas Petit - http://www.ina.fr/)

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

# This code is a modification of nlhkh's project face-alignment-dlib
# See github.com/nlhkh/face-alignment-dlib/blob/master/utils.py
# It has been adapted by Thomas PETIT (github.com/w2ex) & David Doukhan

import dlib
from .rect import Rect
from .remote_utils import get_remote

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

class Dlib68FaceAlignment:
    """
    Methods for detecting eye centers based on Dlib's set of 68 facial landmarks
    this information may be used to rotate the input image such as the eyes lie
    on a horizontal line
    Aligned faces allow to obtain better results on face classication tasks
    """

    def __init__(self, verbose=False):
        """
        Load dlib's 68 facial landmark detection model

        Parameters
        ----------
        verbose : boolean, optional. The default is False.
            If set to True, resulting rotated image will be displayed.
        """
        self.model = dlib.shape_predictor(get_remote('shape_predictor_68_face_landmarks.dat'))
        self.verbose = verbose

    def __call__(self, frame, bb):
        """
        Detects left and right eye centers

        Parameters
        ----------
        frame : numpy.ndarray (height,with, 3)
            RGB image data
        bb : (x1, y1, x2, y2) or None
            Location of the face in the frame
            If set to None, the whole image is considered as a face

        Returns
        -------
        left_eye : (x,y)
            Center of the left eye in the input image
        right_eye : (x,y)
            Center of the right eye in the input image

        """
        if bb is None:
            bb = Rect(0, 0, frame.shape[1], frame.shape[0])
        shape = self.model(frame, bb.to_dlibInt())
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)
        return left_eye, right_eye
