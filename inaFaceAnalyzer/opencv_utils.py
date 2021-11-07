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
import numpy as np
import pylab as plt

def video_iterator(src, time_unit='frame', start=None, stop=None, subsamp_coeff=1, verbose=False):

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
        iframe = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if iframe % subsamp_coeff != 0:
            continue


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if verbose:
            print('frame ', iframe)
            plt.imshow(frame)
            plt.show()

        yield int(iframe), frame

    cap.release()

def image_iterator(lfiles, verbose = False):
    for f in lfiles:
        yield f, imread_rgb(f, verbose)

def get_video_properties(src):
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise Exception("Video file %s does not exist or is invalid" % src)

    dret = {}
    dret['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    dret['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dret ['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dret

def analysisFPS2subsamp_coeff(vid_src, analysisFPS):
    props = get_video_properties(vid_src)
    return int(np.round(props['fps'] / analysisFPS))


def disp_frame(frame):
    plt.imshow(frame)
    plt.show()
    return frame

def disp_frame_shapes(frame, lrect = [], lpoint = []):
    tmpframe = frame.copy()
    for bbox in lrect:
        print('bbox', bbox)
        bbox = [int(e) for e in bbox]
        cv2.rectangle(tmpframe, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 8)
    for point in lpoint:
        print('point', point)
        point = [int(e) for e in point]
        cv2.circle(tmpframe, point, radius=4, color=(0, 0, 255), thickness=-1)
    return disp_frame(tmpframe)

def imwrite_rgb(dst, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(dst, img):
        raise Exception('cannot write image %s' % dst)

def imread_rgb(img_path, verbose=False):
    img = cv2.imread(img_path)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if verbose:
        print('raw image ' + img_path)
        plt.imshow(frame)
        plt.show()
    return frame
