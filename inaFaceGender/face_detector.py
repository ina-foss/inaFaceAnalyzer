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

import cv2
import numpy as np
# from retinaface import RetinaFace

from .remote_utils import get_remote
from .opencv_utils import disp_frame_bblist
from .face_preprocessing import _squarify_bbox
from .face_utils import intersection_over_union, intersection_over_e1


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

def _center(bbox):
    """
    returns center (x,y) of bounding box bbox(x1, y1, x2, y2)
    """
    return ((bbox[0] + bbox[2]) / 2), ((bbox[1] + bbox[3]) / 2)

def _sqdist(p1, p2):
    '''
    return squared distance between points p1(x,y) and p2(x,y)
    '''
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _blackpadd(frame, paddpercent):
    # add black around image
    y, x, z = frame.shape
    #offset = int(max(x, y) * paddpercent)
    xoffset = int(x * paddpercent)
    yoffset = int(y * paddpercent)
    ret = np.zeros((y + 2 * yoffset, x + 2 * xoffset, z), dtype=frame.dtype)
    ret[yoffset:(y + yoffset), xoffset:(x + xoffset), :] = frame
    return ret, yoffset, xoffset



class OcvCnnFacedetector:
    """
    opencv default CNN face detector
    Future : define an abstract class allowing to implement several detection methods
    """
    def __init__(self, minconf=0.65, paddpercent = 0.15, max_rel_dim = None):
        """
        Parameters
        ----------
        minconf : float, optional
           minimal face detection confidence. The default is 0.65.
        paddpercent : float, optional
            input frame is copy passted within a black image with black pixel
            padding. the resulting dimensions is width * (1+2*paddpercent)

        """
        self.minconf = minconf
        self.paddpercent = paddpercent
        self.max_rel_dim = max_rel_dim

        fpb = get_remote('opencv_face_detector_uint8.pb')
        fpbtxt = get_remote('opencv_face_detector.pbtxt')
        self.model = cv2.dnn.readNetFromTensorflow(fpb, fpbtxt)


    def __call__(self, frame, verbose=False):
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
        frame, yoffset, xoffset = _blackpadd(frame, self.paddpercent)
        h, w, z = frame.shape

        # The CNN is intended to work images resized to 300*300

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.model.setInput(blob)
        detections = self.model.forward()

        assert(np.all(-np.sort(-detections[:,:,:,2]) == detections[:,:,:,2]))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.minconf:
                break

            x1, y1, x2, y2 = bbox = _get_opencvcnn_bbox(detections, i)
            # remove noisy detections coordinates
            if x1 >= 1 or y1 >= 1 or x2 <= 0 or y2 <= 0:
                continue
            if x1 >= x2 or y1 >= y2:
                continue
            if x2-x1 > .7 or y2-y1 > .7:
                continue
            if x1 < -.1 or y1 < -.1 or x2 > 1.1 or y2 > 1.1:
                continue
#            print('confidence', confidence)
#            print('rel diff', x2 - x1, y2 - y1)
#            print('rel box', bbox)

            bbox = _rel_to_abs(bbox, w, h)
            #bbox = [e - offset for e in bbox]
            if verbose:
                print('detected face at %s with confidence %s' % (bbox, confidence))
                disp_frame_bblist(frame, [bbox])


            bbox = [bbox[0] - xoffset, bbox[1] - yoffset, bbox[2] - xoffset, bbox[3] - yoffset]
            faces_data.append((bbox, confidence))


        return faces_data

    def get_most_central_face(self, frame, verbose=False):
        """
        Several face annotated datasets may contain several faces per image
        with annotated face at the center of the image
        This method returns the detected face which is closest from the center

        Parameters
        ----------
        frame : nd.array (height, width, 3)
            RGB image data.
        verbose : boolean, optional
            Display detected faces. The default is False.

        Returns
        -------
        The

        """
        faces_data = self.__call__(frame, verbose)
        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)
        ldists = [_sqdist(_center(bbox), frame_center) for (bbox, conf) in faces_data]
        if len(ldists) == 0:
            if verbose:
                print('no face detected')
            return None
        am = np.argmin(ldists)
        bbox, conf = faces_data[am]
        if bbox[0] > frame_center[0] or bbox[2] < frame_center[0] or bbox[1] > frame_center[1] or bbox[3] < frame_center[1]:
            if verbose:
                print('detected faces do not include the center of the image')
            return None
        if verbose:
            print('most closest face with bounding box %s and confidence %f' % (bbox, conf))
            disp_frame_bblist(frame, [bbox])
        return (bbox, conf)

    def get_closest_face(self, frame, ref_bbox, min_iou=.7, squarify=True, verbose=False):
        # get closest detected faces from ref_bbox
        if squarify:
            f = _squarify_bbox
        else:
            f = lambda x: x

        ref_bbox = f(ref_bbox)

        lfaces = self.__call__(frame, verbose)
        if len(lfaces) == 0:
            return None

        liou = [intersection_over_union(f(ref_bbox), f(bbox)) for bbox, _ in lfaces]
        if verbose:
            print([f(bb) for bb, _, in lfaces])
            print('liou', liou)
        am = np.argmax(liou)
        if liou[am] < min_iou:
            return None
        return lfaces[am]


class IdentityFaceDetector:
    def __init__(self):
        pass
    def __call__(self, frame):
        return [((0, 0, frame.shape[1], frame.shape[0]), np.NAN)]


class OverlapFaceDetector:
    def __init__(self, win_len = 300, win_hop = 200):
        self.large_detect = OcvCnnFacedetector(paddpercent=.15)
        self.small_detect = OcvCnnFacedetector(paddpercent=0.)
        self.win_len = win_len
        self.win_hop = win_hop

    def __call__(self, frame, verbose = False):
        ldetect = []

        # detect at small scale on subportions of image
        for y in range(0, frame.shape[0], self.win_hop):
            for x in range(0, frame.shape[1], self.win_hop):
                faces = self.small_detect(frame[y:(y+self.win_len), x:(x+self.win_len), :])
                for (x1, y1, x2, y2), conf in faces:
                    ldetect.append(((x1+x, y1+y, x2+x, y2+y), conf))
        # detect at image scale
        ldetect += self.large_detect(frame)

        # compute intersection over union between all detected faces
        #ioumat = np.zeros((len(ldetect), len(ldetect)))
        #for i, (bb1, conf1) in enumerate(ldetect):
        #    for j, (bb2, conf2) in enumerate(ldetect[:i]):
        #        ioumat[i, j] = ioumat[j, i] = intersection_over_union(bb1, bb2)
        # compute intersection over union between all detected faces
        ioumat = np.zeros((len(ldetect), len(ldetect)))
        for i, (bb1, conf1) in enumerate(ldetect):
            for j, (bb2, conf2) in enumerate(ldetect):
                if i != j:
                    ioumat[i, j] = intersection_over_e1(bb1, bb2)

        # if 2 detections have IOU > 0.5, keep the detection with the biggest confidence
        while len(ioumat) > 0:
            am = ioumat.argmax()
            m = ioumat.ravel()[am]
            if m < 0.5:
                break
            #print(am, m)
            x = am // len(ioumat)
            y = am % len(ioumat)
            #print(x, ldetect[x])
            #print(y, ldetect[y])
            #todelete = y if ldetect[x][1] > ldetect[y][1] else x
            todelete = x
            ldetect.pop(todelete)
            ioumat = np.delete(ioumat, todelete, axis = 0)
            ioumat = np.delete(ioumat, todelete, axis = 1)

        if verbose:
            disp_frame_bblist(frame, [e[0] for e in ldetect])
            for bbox, conf in ldetect:
                x1, y1, x2, y2 = [int(e) for e in bbox]
                print(bbox, conf)
                disp_frame(frame[y1:y2, x1:x2, :])
        return ldetect



# class RetinaFaceDetector:
#     def __init__(self, minconf=0.65):
#         """
#         Parameters
#         ----------
#         minconf : float, optional
#            minimal face detection confidence. The default is 0.65.
#         """
#         self.minconf = minconf

#     def __call__(self, frame, verbose=False):
#         """
#         Detect faces from an image

#         Parameters:
#             frame (array): Image to detect faces from.

#         Returns:
#             faces_data (list) : List containing :
#                                 - the bounding box
#                                 - face detection confidence score
#         """

#         if not frame.any():
#             return []

#         faces_data = []
#         ret = RetinaFace.detect_faces(frame, threshold=self.minconf)
#         print(ret)

#         if not isinstance(ret, dict):
#             return []

#         for k in ret:
#             e = ret[k]
#             faces_data.append((e['facial_area'], e['score']))

#         return faces_data
