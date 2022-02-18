#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019-2022 Ina (David Doukhan - http://www.ina.fr/)

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

# Here are defined the common pieces of code used by command line argument programs

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import inaFaceAnalyzer

epilog = '''
If you are using inaFaceAnalyzer in your research, please cite
the current version number used (%s) together with a reference to the following
paper: David Doukhan and Thomas Petit (2022). inaFaceAnalyzer: a Python toolbox
for large-scale face-based description of gender representation in media with
limited gender, racial and age biases. Submitted to JOSS - The journal of Open
Source Software (submission in progress).
''' % inaFaceAnalyzer.__version__


def new_parser(description):
    parser = ArgumentParser(description=description,
                            epilog=epilog,
                            formatter_class= ArgumentDefaultsHelpFormatter)# ,
                            #add_help=False)
    return parser

hfps = '''Amount of video frames to be processed per second.
Remaining frames will be skipped.
If not provided, all video frames will be processed (generally between 25 and 30 per seconds).
Lower FPS values results in faster processing time'''
def add_fps(parser):
    parser.add_argument('--fps', default=None, type=float, help=hfps)

# hkeyframes = '''Face detection and analysis from video limited to video key frames.
# Allows fastest video analysis time associated to a summary with
# non uniform frame sampling rate. Incompatible with the --fps, --ass_subtitle_export or --mp4_export arguments.'''
# def _add_keyframes(parser):
#     parser.add_argument('--keyframes', action='store_true', help=hkeyframes)

# def add_framerate(parser):
#     # keyframes and fps arguments are mutually exclusive
#     group = parser.add_mutually_exclusive_group()
#     _add_fps(group)
#     _add_keyframes(group)

htracking = '''Face tracking works jointly with face detection systems :
a face should be first detected before being tracked.
Face detection will be performed one every DETECT_PERIOD frames and allow to
detect new faces, or faces that were lost due to occulusions.
Face tracking is performed on every frame.
Default DETECT_PERIOD (1) is the more costly and the more robust: face detection will
be performed for every video frames together with face tracking allowing to smooth results.
Larger DETECT_PERIOD are efficient as long as analysis FPS is high, and users
should define a proper trade-off between DETECT_PERIOD and the FPS.
High DETECT_PERIOD and low FPS will results in missed faces due to occlusions
or fast changes in face position.
'''
def add_tracking(parser):
    # TODO: add tracking threshold in the options
    tg = parser.add_argument_group('Arguments specific to "videotracking" engine')
    tg.add_argument('--detect_period', type=int, default=1, dest='detect_period', help=htracking)
