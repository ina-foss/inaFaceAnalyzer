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

import dlib
import numpy as np
from .opencv_utils import disp_frame_bblist
from .face_utils import _rect_to_tuple, tuple2rect, intersection_over_union, tuple2drect

def _matrix_argmax(m):
    x = np.argmax(m)
    dim = m.shape[1]
    return x // dim, x % dim

class Tracker:
    def __init__(self, frame, bb, detect_conf): #, min_confidence):
        self.t = dlib.correlation_tracker()
        self.t.start_track(frame, tuple2rect(bb))
        self.fshape = frame.shape
        self.detect_conf = detect_conf
        self.track_conf = None
#        self.min_confidence = min_confidence

    def update(self, frame, verbose=False):
        update_val = self.t.update(frame)

        fh, fw, _ = self.fshape

        e = self.t.get_position()
        x1, y1, x2, y2 = pos = _rect_to_tuple(e)


        if verbose:
            print('update', pos, update_val)
            disp_frame_bblist(frame, [pos])

        if not ((x2 > 0) and (x1 < fw) and (y2 > 0) and (y1 < fh)):
            update_val = -1

        self.detect_conf = None
        self.track_conf = update_val
        return update_val

    def update_from_bb(self, frame, bb, detect_conf, verbose=True):
        update_val = self.t.update(frame, tuple2drect(bb))
        if verbose:
            e = self.t.get_position()
            print('dest', bb, 'new position', e)
            disp_frame_bblist(frame, [_rect_to_tuple(e), _rect_to_tuple(bb)])
        self.t.start_track(frame, bb)
        self.track_conf = update_val
        self.detect_conf = detect_conf
        return update_val


class TrackerDetector:
    # confidence is estimated as the peak to side-lobe ration returned
    # on tracker's update
    min_confidence = 7

    # output labels
    out_names = ['bb', 'face_id', 'face_detect_conf', 'tracking_conf']

    def __init__(self, detector, detection_period):
        # dictionnary of tracked objects
        self.d = {}
        # number of instantiated trackers
        # used to provide a numeric identifier to each new Tracker instance
        self.nb_tracker = 0
        # a face/object detection instance
        # could be any face detector class defined in face_detector.py
        self.detector = detector
        # Detection will be performed once every 'detection_period' frames
        # ex: if detection_period is set to 5, detection will be performed
        # for 1 frame, and tracking for the 4 remaining frames
        # if detection_period is set to 1, the detection will be performed on
        # every frame, and tracking will also be performed to know the detected
        # faces belong to the same person
        self.detection_period = detection_period
        # count the amount of processed frames
        # used to switch between detection and tracking every detection_period
        self.iframe = 0


    def update_trackers(self, frame, verbose = False):
        if verbose:
            print('update trackers')
        # if tracked element is lost, remove tracker
        for fid in list(self.d):
            if self.d[fid].update(frame, verbose) < self.min_confidence:
                del self.d[fid]
                if verbose:
                    print('deleting tracker', fid)

    def update_from_detection(self, frame, lbox, verbose):

        if verbose:
            print('update from detection')

        lkeys = list(self.d.keys())

        # compute intersection over union matrix[#tracker, #detected bounding box]
        # between tracker positions and detected bounding boxes
        ioumat = np.ones((len(lkeys), len(lbox))) * -1
        for i, k in enumerate(lkeys):
            for j, (bb, detect_conf) in enumerate(lbox):
                ioumat[i, j] = intersection_over_union(self.d[k].t.get_position(), tuple2drect(bb))

        # while matrix not empty and IOU > 70%
        while np.prod(ioumat.shape):
            # find the largest intersection over union
            #print(matrix_argmax(ioumat))
            itracker, idetection = am = _matrix_argmax(ioumat)
            if ioumat[am] <= 0.7:
                break
            # update closest bounding box and trackers and remove them from matrix
            track_score = self.d[lkeys[itracker]].update_from_bb(frame, *lbox[idetection], verbose)
            ioumat = np.delete(ioumat, itracker, axis = 0)
#            ioutmat = np.delete(ioutmat, idetection, axis=1)
            lkeys.pop(itracker)

            # if close bounding box and tracker do not match, delete tracker
            if track_score < self.min_confidence:
                del self.d[itracker]
            else:
                # if bounding box and detected face match, remove
                # the detection from the set
                lbox.pop(idetection)

        # remove trackers that do not match any detected box
        for k in lkeys:
            del self.d[k]

        # add new trackers corresponding to detected faces that did not match
        # any existing tracker
        for bb, detect_conf in lbox:
            self.d[self.nb_tracker] = Tracker(frame, bb, detect_conf)
            self.nb_tracker += 1


    def __call__(self, frame, verbose=False):

        if self.iframe % self.detection_period == 0:
            self.update_from_detection(frame, self.detector(frame, verbose), verbose)
        else:
            self.update_trackers(frame, verbose)

        lret = []
        for faceid in self.d:
            t = self.d[faceid]
            bb = t.t.get_position()
            lret.append((_rect_to_tuple(bb), faceid, t.detect_conf, t.track_conf))

        self.iframe += 1

        return lret
