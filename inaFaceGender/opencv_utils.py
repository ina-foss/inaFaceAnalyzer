#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan - http://www.ina.fr/)

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

def video_iterator(src, time_unit='frame', start=None, stop=None, subsamp_coeff=1):
    
    # cv2.CAP_PROP_POS_MSEC property was not used because it is buggy
     
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise Exception("Video file %s does not exist or is invalid" % src)
    
    unit = cv2.CAP_PROP_POS_FRAMES

    if time_unit == 'frame':
        pass
    elif time_unit == 'ms':
        fps = cap.get(cv2.CAP_PROP_FPS)
        if start is not None:
            start = int(start * fps // 1000)
        if stop is not None:
            stop = int(stop * fps // 1000)
    else:
        raise NotImplementedError

    if start is not None:
        cap.set(unit, start)

    while cap.isOpened():
        
        if stop is not None and cap.get(unit) > stop:
            break
        
        ret, frame = cap.read()
        if not ret:
            break

        # skip frames for subsampling reasons
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % subsamp_coeff != 0:
            continue


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield cap.get(cv2.CAP_PROP_POS_FRAMES) - 1, frame

    cap.release()

def get_fps(src):
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise Exception("Video file %s does not exist or is invalid" % src)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps