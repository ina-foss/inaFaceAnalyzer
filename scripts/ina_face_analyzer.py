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

#import argparse
import os
import sys

import inaFaceAnalyzer.commandline_utils as ifacu
import inaFaceAnalyzer.face_classifier
from inaFaceAnalyzer.face_detector import facedetection_cmdline
from inaFaceAnalyzer.face_classifier import faceclassifier_cmdline

import inaFaceAnalyzer.display_utils


# TODO: add options to perform detection only and/or preprocessing only
# options to export image results to PDF with incrusted bounding boxes ?


description = 'inaFaceAnalyzer %s: detects and classify faces from media collections and export results in csv' % inaFaceAnalyzer.__version__
parser = ifacu.new_parser(description)

## Required arguments
ra = parser.add_argument_group('required arguments')

hengine = '''Analysis engine to be used.
"video" should be the default choice for video materials.
"image" should be the default choice for image files.
"videotracking" uses face tracking methods to lower computation time and smooth
face classification results obtained on video materials.
Tracked faces are associated to a numeric identifier.
Smoothed classification estimates are associated to "_avg" suffix in resulting column names,
and are more robust than frame-isolated predictions.
"videokeyframes" restricts analyses to video key frames. It allows the fastest
video analysis summary, but is is associated to a non-uniform frame sampling rate.
"preprocessed_image" skips face detection and alignment steps and requires face
images that can be used directly by face classifiers.
These images should be already detected, cropped, centered, aligned and rescaled to 224*224 pixels.
'''
ra.add_argument('--engine', choices=['video', 'image', 'videotracking', 'videokeyframes', 'preprocessed_image'],
                required=True, help=hengine)

h = '''INPUT is a list of documents to analyse. ex: /home/david/test.mp4 /tmp/mymedia.avi.
INPUT can be a list of video paths OR a list of image paths.
Videos and images can have heterogenous formats but cannot be mixed in a single command.
'''
ra.add_argument('-i', nargs='+', required=True, dest='input', help = h)

h = '''When used with "video" and "videotracking" engines, OUTPUT is the path
to an existing directory storing one resulting CSV for each processed video.
When used with "image" and "preprocessed_image" engines, OUTPUT is the path to
a resulting csv that will be created and will contain a line for each detected faces.
'''
ra.add_argument('-o', required=True, help = h, dest='output')


## Optional arguments
oa = parser.add_argument_group('optional arguments')

#oa.add_argument("-h", "--help", action="help", help="show this help message and exit")


# classifier
faceclassifier_cmdline(parser)

ifacu.add_batchsize(oa)

## Video only parameters
dv = parser.add_argument_group('optional arguments to be used only with "video" and "videotracking" engines')
ifacu.add_fps(dv)

dv.add_argument('--ass_subtitle_export', action='store_true',
                help='export analyses into a rich ASS subtitle file which can be displayed with VLC or ELAN')

dv.add_argument('--mp4_export', action='store_true',
                help='export analyses into a a MP4 video with incrusted bounding boxes and analysis estimates')


ifacu.add_tracking(parser)

# face detection
facedetection_cmdline(parser)


#### parse arguments
args = parser.parse_args()


# deal with incompatible arguments

class ParserError:
    def __init__(self, parser):
        self.parser = parser
    def __call__(self, a1, a2):
        raise self.parser.error('%s argument cannot be mixed with %s' % (a1, a2))
pe = ParserError(parser)

# Management of incompatible uses

for k in ['fps', 'ass_subtitle_export', 'mp4_export']:
    if args.__dict__[k] and (args.engine not in ['video', 'videotracking']):
        pe('--' + k, '--engine ' + args.engine)


# analysis engine constructor
engine = ifacu.engine_factory(args)

# image engines
if args.engine in ['image', 'preprocessed_image']:
    # check provided output extension
    out_ext = os.path.splitext(args.output)[1]
    if out_ext.lower() != '.csv':
        raise ValueError('OUTPUT value %s provided to -o argument should have .csv extension' % args.output)

    # run engine
    if args.engine == 'image':
        df = engine(args.input)
    elif args.engine == 'preprocessed_image':
        df = engine.preprocessed_img_list(args.input, batch_len=args.batch_size)
    # save results
    df.to_csv(args.output, index=False)
    sys.exit(0)



# VIDEO engines

# test that provided output path is an existing directory
if not os.path.isdir(args.output):
    raise ValueError('OUTPUT directory %s provided to -o argument should be an existing directory' % args.output)

dargs = {}
if args.fps:
    dargs = {'fps': args.fps}


nbvid = len(args.input)
for i, f in enumerate(args.input):
    # TODO: add a try/catch system
    # TODO: add a progressbar.ProgressBar
    print('analyzing video %d/%d: %s' % (i, nbvid, f))
    base, _ = os.path.splitext(os.path.basename(f))
    df = engine(f, **dargs)
    df.to_csv('%s/%s.csv' % (args.output, base), index=False)
    if args.ass_subtitle_export:
        inaFaceAnalyzer.display_utils.ass_subtitle_export(f, df, '%s/%s.ass' % (args.output, base), analysis_fps=args.fps)
    if args.mp4_export:
        inaFaceAnalyzer.display_utils.video_export(f, df, '%s/%s.mp4' % (args.output, base), analysis_fps=args.fps)

