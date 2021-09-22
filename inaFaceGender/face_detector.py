#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan & Zohra Rezgui- http://www.ina.fr/)

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

import os
import cv2

def _opencv_get_bbox_pts(detections, face_idx, frame_width, frame_height):
    """
    Extract bounding boxes from opencv CNN detection output
    """
    
    x1 = int(detections[0, 0, face_idx, 3] * frame_width)
    y1 = int(detections[0, 0, face_idx, 4] * frame_height)
    x2 = int(detections[0, 0, face_idx, 5] * frame_width)
    y2 = int(detections[0, 0, face_idx, 6] * frame_height)

    # TODO: check here if x1 < x2 ???

    width = x2 - x1
    height = y2 - y1
    max_size = max(width, height)
    
    # TODO : are these 4 lines usefull ??
    x1, x2 = max(0, (x1 + x2) // 2 - max_size // 2), min(frame_width, (x1 + x2) // 2 + max_size // 2)
    y1, y2 = max(0, (y1 + y2) // 2 - max_size // 2), min(frame_height, (y1 + y2) // 2 + max_size // 2)

    return x1, y1, x2, y2


def _scale_bbox(x1, y1, x2, y2, scale, frame_shape=None):
    w = x2 - x1
    h = y2 - y1
    
    x1 = int(x1 - (w*scale - w)/2)
    y1 = int(y1 - (h*scale -h)/2)
    x2 = int(x2 + (w*scale - w)/2)
    y2 = int(y2 + (h*scale -h)/2)

    if frame_shape is not None:
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame_shape[1])
        y2 = min(y2, frame_shape[0])

    return x1, y1, x2, y2


class OcvCnnFacedetector:
    """
    opencv default CNN face detector
    Future : define an abstract class allowing to implement several detection methods
    """
    def __init__(self, minconf=0.65, bbox_scaling=1.0):
        """
        Parameters
        ----------
        minconf : float, optional
           minimal face detection confidence. The default is 0.65.
        bbox_scaling : float, optional
            scaling factor to be applied to the face bounding box.
            The default is 1.0.

        """
        self.minconf = minconf
        self.bbox_scaling = bbox_scaling
        p = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        self.model = cv2.dnn.readNetFromTensorflow(p + "opencv_face_detector_uint8.pb",
                                                   p + "opencv_face_detector.pbtxt")

        
    def __call__(self, frame):
        """ 
        Detect faces from an image
  
        Parameters: 
            frame (array): Image to detect faces from.
          
        Returns: 
            faces_data (list) : List containing :
                                - the bounding box after scaling
                                - face detection confidence score
        """
        
        faces_data = []

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        # The CNN is intended to work images resized to 300*300
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.model.setInput(blob)
        detections = self.model.forward()
        
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.minconf:
                bbox = _opencv_get_bbox_pts(detections, i, frame_width, frame_height)
                
                #x1, y1, x2, y2 = bbox[:]
                #x1, y1, x2, y2 = _scale_bbox(x1, y1, x2, y2, self.bbox_scaling, frame.shape)
                x1, y1, x2, y2 = _scale_bbox(*bbox, self.bbox_scaling, frame.shape)
                
                if x1 < x2 and y1 < y2:
                    dets = (x1, y1, x2, y2)
                else:
                    ## TODO WARNING - THIS HACK IS STRANGE
                    dets = (0, 0, frame_width, frame_height)

                faces_data.append((dets, confidence))
        return faces_data