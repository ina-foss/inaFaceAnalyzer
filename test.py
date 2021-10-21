#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:57:54 2021

@author: ddoukhan
"""
from inaFaceGender.face_tracking import Tracker, TrackerDetector
from inaFaceGender.opencv_utils import video_iterator, disp_frame_bblist
from inaFaceGender.face_detector import OcvCnnFacedetector
from inaFaceGender.face_utils import tuple2rect, intersection_over_union, _rect_to_tuple

vit = video_iterator('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=30, verbose=False)
lframes = [e[1] for e in vit]

detector = OcvCnnFacedetector(paddpercent=0.15, minconf=.5)

td = TrackerDetector(detector, 2)

td(lframes[0], True)
td(lframes[1], True)
td(lframes[2], True)
