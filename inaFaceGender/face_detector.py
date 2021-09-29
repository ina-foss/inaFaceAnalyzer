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
import numpy as np

def _get_opencvcnn_bbox(detections, face_idx):
    """
    Extract bounding boxes from opencv CNN detection output
    Results are in relative coordinates 0...1
    """
    
    x1 = detections[0, 0, face_idx, 3]
    y1 = detections[0, 0, face_idx, 4]
    x2 = detections[0, 0, face_idx, 5]
    y2 = detections[0, 0, face_idx, 6]
    return x1, y1, x2, y2

def _rel_to_abs(bbox, frame_width, frame_height):
    """
    Map relative coordinates 0...1 to absolute corresponding to
    frame width (w) and frame height (h)
    """
    #print('rel', bbox)
    x1, y1, x2, y2 = bbox
    return x1*frame_width, y1*frame_height, x2*frame_width, y2*frame_height



class OcvCnnFacedetector:
    """
    opencv default CNN face detector
    Future : define an abstract class allowing to implement several detection methods
    """
    def __init__(self, minconf=0.65):
        """
        Parameters
        ----------
        minconf : float, optional
           minimal face detection confidence. The default is 0.65.
        """
        self.minconf = minconf
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
                                - the bounding box
                                - face detection confidence score
        """
        
        faces_data = []

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        # The CNN is intended to work images resized to 300*300
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.model.setInput(blob)
        detections = self.model.forward()
        
        assert(np.all(-np.sort(-detections[:,:,:,2]) == detections[:,:,:,2]))
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.minconf:
                break
            
            bbox = _get_opencvcnn_bbox(detections, i)
            # remove noisy detections coordinates
            if bbox[0] >= 1 or bbox[1] >= 1 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            
            bbox = _rel_to_abs(bbox, frame_width, frame_height)
            faces_data.append((bbox, confidence))
            
        return faces_data
