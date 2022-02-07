#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019-2021 Ina (David Doukhan & Zohra Rezgui - http://www.ina.fr/)

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


def _hex2rgb(hx):
    return (int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16))


def _sec2hmsms(s):
    td = datetime.timedelta(seconds=s)
    h,m,s = str(td).split(':')
    return '%d:%d:%.2f' % (int(h), int(m), float(s))


def _analysis2displaydf(df, fps, subsamp_coeff, text_pat = None, cols=None):
    """
    Convert analysis results to a generic pandas dataframe containing
    formated information to be displayed using export functions defined bellow
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
        df.bbox = df.bbox.map(lambda x: eval(x))

    ret = pd.DataFrame()
    ret['frame'] = df.frame
    ret['bbox'] = df.bbox
    ret[['x1', 'y1', 'x2', 'y2']] = df.apply(lambda x: x.bbox, axis=1, result_type="expand")
    ret['start'] = df.frame.map(lambda x: _sec2hmsms(x / fps))
    ret['stop'] = df.frame.map(lambda x: _sec2hmsms((x + subsamp_coeff) / fps))
    if text_pat is None:
        if 'face_id' in df.columns:
            ret['rgb_color'] = df.sex_label_avg.map(lambda x: '0000FF' if x == 'm' else '00FF00')
            text_pat = 'id: %s'
            cols = ['face_id']
            if 'sex_label_avg' in df.columns:
                text_pat += ' - sex: %s (%.1f)'
                cols += ['sex_label_avg', 'sex_decfunc_avg']
            if 'age_label_avg' in df.columns:
                text_pat += ' - age: %.1f'
                cols += ['age_label_avg']
        else:
            text_pat = 'sex: %s (%.1f)'
            cols = ['sex_label', 'sex_decfunc']
            ret['rgb_color'] = df.sex_label.map(lambda x: '0000FF' if x == 'm' else '00FF00')
            if 'age_label' in df.columns:
                text_pat += '- age: %.1f'
                cols += ['age_label']

    ret['bgr_color'] = ret.rgb_color.map(lambda x: x[4:] + x[2:4] + x[:2])
    ret['text'] = df.apply(lambda x: text_pat % tuple([x[e] for e in cols]), axis=1)
    return ret

def ass_subtitle_export(vid_src, result_df, subtitle_dst, analysis_fps=None):
    """
    Export inaFaceAnalyzer results to an ASS subtitle file allowing to display
    faces bounding boxes and other analysis information
    ASS subtitles can be displayed using VLC or Aegisub

    Parameters
    ----------
    vid_src : str
        path to the input video.
    result_df : str or pandas.DataFrame
        path to the csv or pandas DataFrame corresponding to the analysis of the input video.
    subtitle_dst : str
        output path which will store the resulting subtitle file.
        Must have ass extension
    analysis_fps: numeric or None
        The amount of frames per second which were analyzed
        if set to None, then consider that all video frames were processed
    """


    assert subtitle_dst[-4:].lower() == '.ass', subtitle_dst

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
    # text font size set to 4% video height
    t.text_font_size = int(0.04 * height)

    with open(subtitle_dst, 'wt') as fid:
        print(t, file=fid)


def video_export(vid_src, result_df, vid_dst, analysis_fps=None):
    """
    Export inaFaceAnalyzer results to a video with incrusted faces bounding
    boxes and other analysis information

    Parameters
    ----------
    vid_src : str
        path to the input video.
    result_df : str or pandas.DataFrame
        path to the csv or pandas DataFrame corresponding to the analysis of the input video.
    vid_dst : str
        output path which will store the corresponding video.
        Must have MP4 extension
    analysis_fps: numeric or None
        The amount of frames per second which were analyzed
        if set to None, then consider that all video frames were processed
    """
    assert vid_dst[-4:].lower() == '.mp4', vid_dst


    video_props = get_video_properties(vid_src)
    fps, width, height = [video_props[e] for e in ['fps', 'width', 'height']]

    if analysis_fps is None:
        subsamp_coeff = 1
    else:
        subsamp_coeff = analysisFPS2subsamp_coeff(vid_src, analysis_fps)

    displaydf = _analysis2displaydf(result_df, fps, subsamp_coeff)

    font = cv2.FONT_HERSHEY_SIMPLEX

    with tempfile.TemporaryDirectory() as p:
        tmpout = '%s/%s.mp4' % (p, os.path.splitext(os.path.basename(vid_src))[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmpout, fourcc, fps / subsamp_coeff, (width, height))
        #print('out', out)

        for iframe, frame in video_iterator(vid_src, subsamp_coeff=subsamp_coeff):

            sdf = displaydf[displaydf.frame == iframe]

            for e in sdf.itertuples():

                x1, y1, x2, y2 = e.bbox
                color = _hex2rgb(e.rgb_color)
                cv2.putText(frame,e.text,(x1 - 100, y1 - 10 ), font, 0.7, color,2,cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 8)

            ret = out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        out.release()
        os.system('ffmpeg -y -i %s -i %s -map 0:v:0 -map 1:a? -vcodec libx264 -acodec copy %s' % (tmpout, vid_src, vid_dst))
