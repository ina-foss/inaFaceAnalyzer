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
from Cheetah.Template import Template
import datetime
from .opencv_utils import video_iterator, get_video_properties, analysisFPS2subsamp_coeff


def _sec2hmsms(s):
    td = datetime.timedelta(seconds=s)
    h,m,s = str(td).split(':')
    return '%d:%d:%.2f' % (int(h), int(m), float(s))


def _analysis2displaydf(df, fps, subsamp_coeff):
    ret = pd.DataFrame()
    ret['frame'] = df.frame
    ret['bbox'] = df.bbox
    ret[['x1', 'y1', 'x2', 'y2']] = df.apply(lambda x: x.bbox, axis=1, result_type="expand")
    ret['rgb_color'] = df.sex_label.map(lambda x: '0000FF' if x == 'm' else '00FF00')
    ret['bgr_color'] = ret.rgb_color.map(lambda x: x[4:] + x[2:4] + x[:2])
    ret['start'] = df.frame.map(lambda x: _sec2hmsms(x / fps))
    ret['stop'] = df.frame.map(lambda x: _sec2hmsms((x + subsamp_coeff) / fps))
    ret['text'] = df.apply(lambda x: 'sex: %s (%.1f) - age: %.1f' % (x.sex_label, x.sex_decfunc, x.age_label), axis=1)
    return ret

def ass_subtitle_export(vid_src, result_df, dst, analysis_fps=None):

    if isinstance(result_df, str):
        result_df = pd.read_csv(result_df)
    assert isinstance(result_df, pd.DataFrame)

    video_props = get_video_properties(vid_src)
    fps, width, height = [video_props[e] for e in ['fps', 'width', 'height']]

    if analysis_fps is None:
        subsamp_coeff = 1
    else:
        subsamp_coeff = analysisFPS2subsamp_coeff(vid_src, analysis_fps)

    displaydf = _analysis2displaydf(result_df, fps, subsamp_coeff)

    p = os.path.dirname(__file__)
    t = Template(file = p + '/template.ass')

    t.height = height
    t.width = width
    t.display_df = displaydf

    with open(dst, 'wt') as fid:
        print(t, file=fid)


# def frames2mp4v(frames_list, file_name, fps):
#     """
#     Writes a list of frames into a video using MP4V encoding.

#     Parameters:
#     frames_list (list): List of the frames to write
#     file_name (string): video output path
#     fps (int) : Number of frames per second used in output video


#     """
#     frame_width = frames_list[0].shape[0]
#     frame_height = frames_list[0].shape[1]
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(file_name,fourcc,
#                       fps, (frame_height,frame_width))

#     for frame in frames_list:

#         out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     out.release()


def incrust_faces_in_video(invid, incsv, outvid, subsamp_coeff=1, collabel='label', coldecision='decision'):
    """
    Use a video and its face analysis in CSV to generate a video
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

    invid_props = get_video_properties(invid)

    with tempfile.TemporaryDirectory() as p:
        tmpout = '%s/%s.mp4' % (p, os.path.splitext(os.path.basename(invid))[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmpout,fourcc,
                      invid_props['fps'] / subsamp_coeff, (invid_props['width'],invid_props['height']))
        #print('out', out)


        for iframe, frame in video_iterator(invid, subsamp_coeff=subsamp_coeff):

            sdf = df[df.frame == iframe]

            for _, e in sdf.T.iteritems():

                x1, y1, x2, y2 = eval(e.bbox)
                label = e[collabel]
                text3 = label + ' Decision_func_value: '+ str(round(e[coldecision],3))


                if label == 'm': # blue
                    color = (0,0,255)
                else: # red
                    color = (255,0,0)
                cv2.putText(frame,text3,(x1 - 100, y1 - 10 ), font, 0.7, color,2,cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 8)

            ret = out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #print(ret)

        out.release()
        os.system('ffmpeg -y -i %s -i %s -map 0:v:0 -map 1:a? -vcodec libx264 -acodec copy %s' % (tmpout, invid, outvid))
