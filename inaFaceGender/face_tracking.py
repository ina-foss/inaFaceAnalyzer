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

class Tracker:
    
    def __init__(self, frame, bb, tid):
        self.last_frame = frame
        self.t = dlib.correlation_tracker()
        self.t.start_track(frame, bb)
        self.tid = tid
        
    def intersect_area(self, bb):
        pos = self.t.get_position()
        tmp = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        return bb.intersect(tmp).area()
    
    def iou(self, bb):
        # intersection over union
        inter = self.intersect_area(bb)
        return inter / (inter + self.t.get_position().area() + bb.area())
    
    def update(self, frame):
        self.last_frame = frame
        return self.t.update(frame)
    
    def tracking_quality(self, frame, bb):
        # should be above 7
        ret = self.t.update(frame, bb)
        self.t.start_track(frame, bb)
        return ret
    
    def is_in_frame(self):
        # at least one pixel in the frame
        fheight, fwidth, _ = self.last_frame.shape
        pos = self.t.get_position()
        return (pos.right() > 0) and (pos.left() < fwidth) and (pos.top() < fheight) and (pos.bottom() > 0)
        


class TrackerList:
    
    def __init__(self):
        self.d = {}
        self.i = 0
        
    def update(self, frame):
        for fid in list(self.d):
            if self.d[fid].update(frame) < 7 or not self.d[fid].is_in_frame():
                del self.d[fid]

    def ingest_detection(self, frame, lbox):
        # le nombre de tracker restant doit être égal au nombre de bbox
        # au debut on envisage de tout enlever
        # pour toute bbox, si intersection, alors on enleve pas le match
        # si pas d'intersection, on enleve
        to_remove = list(self.d)
        to_add = {}
        
        for bb in lbox:
            rmscore = [self.d[k].tracking_quality(frame, bb) if self.d[k].iou(bb) > 0.5 else 0 for k in to_remove]
            am = np.argmax(rmscore) if len(rmscore) > 0 else None
            if am is not None and rmscore[am] > 7:
                # update tracker with the new bb coords
                idrm = to_remove[am]
                to_remove.remove(idrm)
                to_add[idrm] = Tracker(frame, bb, idrm)
            else:
                # new tracker
                to_add[self.i] = Tracker(frame, bb, self.i)
                self.i += 1
        self.d = to_add