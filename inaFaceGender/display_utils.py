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
import tempfile
import os
import pandas as pd
from .opencv_utils import video_iterator, get_fps


def frames2mp4v(frames_list, file_name, fps):
    """
    Writes a list of frames into a video using MP4V encoding.

    Parameters:
    frames_list (list): List of the frames to write
    file_name (string): video output path
    fps (int) : Number of frames per second used in output video


    """
    frame_width = frames_list[0].shape[0]
    frame_height = frames_list[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name,fourcc,
                      fps, (frame_height,frame_width))

    for frame in frames_list:

        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out.release()


def incrust_faces_in_video(invid, incsv, outvid, subsamp_coeff=1, collabel='label', coldecision='decision'):
    """
    Use a video and its inafacegender analysis in CSV to generate a video
    with incrusted faces bounding boxes and predictions informations
    Parameters
    ----------
    invid : str
        path to the input video.
    incsv : str
        path to the csv corresponding to the analysis of the input video.
    outvid : str
        output path which will store the corresponding video.
    collabel : str, optional
        Column to be used for classification label between label and smoothed_label.
        The default is 'label'.
    coldecision : str, optional
        Column corresponding to classification decision function.
        Can be 'decision' or 'smoothed_decision'
        The default is 'decision'.
    """
    assert outvid[-4:].lower() == '.mp4', outvid

    df = pd.read_csv(incsv)

    font = cv2.FONT_HERSHEY_SIMPLEX
    processed_frames = []


    for iframe, frame in video_iterator(invid, subsamp_coeff=subsamp_coeff):

        sdf = df[df.frame == iframe]

        for _, e in sdf.T.iteritems():

            x1, y1, x2, y2 = eval(e.bbox)
#                text1 = 'det: ' + str(round(conf[i][1], 3))
#            text3 = label[i][1] + ' Decision_func_value: '+ str(round(decision[i][1],3))
            label = e[collabel]
            text3 = label + ' Decision_func_value: '+ str(round(e[coldecision],3))

          #  cv2.putText(frame,str(text1),(x1 - 100, y1 - 30 ), font, 0.7, (255,255,255),2,cv2.LINE_AA)

            #print(tmplab, type(tmplab), len(tmplab.strip()), 'm', type('m'), tmplab == 'm', len('m'))

            if label == 'm': # blue
                color = (0,0,255)
            else: # red
                color = (255,0,0)
            cv2.putText(frame,text3,(x1 - 100, y1 - 10 ), font, 0.7, color,2,cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 8)

        processed_frames.append(frame)

    with tempfile.TemporaryDirectory() as p:
        tmpout = '%s/%s.mp4' % (p, os.path.splitext(os.path.basename(invid))[0])
        frames2mp4v(processed_frames, tmpout, get_fps(invid) / subsamp_coeff)
        os.system('ffmpeg -y -i %s -i %s -map 0:v:0 -map 1:a? -vcodec libx264 -acodec copy %s' % (tmpout, invid, outvid))
