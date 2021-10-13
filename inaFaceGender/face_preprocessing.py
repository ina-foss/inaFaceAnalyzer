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
from matplotlib import pyplot as plt

# def crop_image_fill(frame, bb):
#     x1, y1, x2, y2 = bb
#     hframe, wframe = (frame.shape[0], frame.shape[1])
#     if x1 >= 0 and y1 >= 0 and x2 <= wframe and y2 <= hframe:
#         return frame[top:bottom, left:right]
#     w = x2 - x1
#     h = y2 - y1
#     sframe = frame[max(0, y1):min(y2, hframe), max(0, x1):min(x2, wframe),:]
#     ret = np.zeros((h, w, 3), np.uint8)
#     yoff = (h - sframe.shape[0]) // 2
#     xoff = (w - sframe.shape[1]) // 2
#     print('xoff', xoff, 'yoff', yoff)
#     ret[yoff:(h+yoff), xoff:(w+xoff), :] = sframe[:,:,:]
#     return ret

def _scale_bbox(x1, y1, x2, y2, scale):
    w = x2 - x1
    h = y2 - y1

    x1 = x1 - (w*scale - w)/2
    y1 = y1 - (h*scale -h)/2
    x2 = x2 + (w*scale - w)/2
    y2 = y2 + (h*scale -h)/2

    return x1, y1, x2, y2

def _squarify_bbox(bbox):
    """
    Convert a rectangle bounding box to a square
    width sides equal to the maximum between width and height
    """
    #print('sq' , bbox)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    offset = max(w, h) / 2
    x_center = (x1+x2) / 2
    y_center = (y1+y2) / 2
    return x_center - offset, y_center - offset, x_center + offset, y_center + offset

# TODO: this may cause a stretching in a latter stage
# TODO conversion to int should be done after scaling
def _norm_bbox(bbox, frame_width, frame_height):
    """
    convert to int and crop bbox to 0:frame_shape
    """
    x1, y1, x2, y2 = [int(e) for e in bbox]
    return x1, y1, x2, y2

def preprocess_face(frame, bbox, squarify, bbox_scale, norm, face_alignment, output_shape, verbose=False):
    """
    Apply preprocessing pipeline to a detected face and returns the
    corresponding image with the following optional processings
    Parameters
    ----------
    frame : numpy nd.array
        RGB image data
    bbox : tuple (x1, y1, x2, y2) or None
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
    norm : boolean
        if set to True, bounding boxes will be converted from float to int
        and will be cropped to fit inside the input frame
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
    if bbox is None:
        bbox = (0, 0, frame_w, frame_h)

    # if True, extend the bounding box to the smallest square containing
    # the orignal bounding box
    if squarify:
        bbox = _squarify_bbox(bbox)

    # perform bounding box scaling to march larger/smaller areas around
    # the detected face
    bbox = _scale_bbox(*bbox, bbox_scale)

    # bounding box normalization to int and to fit in input frame
    if norm:
        bbox = _norm_bbox(bbox, frame_w, frame_h)

    # if verbose, display the image and the bounding box
    if verbose:
       tmpframe = frame.copy()
       cv2.rectangle(tmpframe, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 8)
       plt.imshow(tmpframe)
       plt.show()

    # performs face alignment based on facial landmark detection
    if face_alignment is not None:
        frame, left_eye, right_eye = face_alignment(frame, bbox)
    else:
        # crop image to the bounding box
	# TODO: replace by a wrap affine => management of out of frame
        frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # resize image to the required output shape
    if output_shape is not None:
        frame = cv2.resize(frame, output_shape)

    if verbose:
        print('resulting image')
        plt.imshow(frame)
        plt.show()

    return frame, bbox
