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

import argparse
import os
import sys

import inaFaceAnalyzer.face_classifier
from inaFaceAnalyzer.face_detector import facedetection_cmdlineparser, facedetection_factory
import inaFaceAnalyzer.inaFaceAnalyzer
import inaFaceAnalyzer.display_utils


# TODO: add options to perform detection only and/or preprocessing only

epilog = '''
If you are using inaFaceAnalyzer in your research-related documents, please cite
the current version number used (%s) together with a reference to the following
paper: David Doukhan and Thomas Petit (2022). inaFaceAnalyzer: a Python toolbox
for large-scale face-based description of gender representation in media with
limited gender, racial and age biases. Submitted to JOSS - The journal of Open
Source Software (submission in progress).
''' % inaFaceAnalyzer.__version__

parser = argparse.ArgumentParser(description='inaFaceAnalyzer %s: detects and classify faces from media collections and export results in csv' % inaFaceAnalyzer.__version__,
                                 # TODO: add bibliographic reference
                                 epilog=epilog,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

## Required arguments
ra = parser.add_argument_group('required arguments')

h = '''INPUT is a list of documents to analyse. ex: /home/david/test.mp4 /tmp/mymedia.avi.
INPUT can be a list of video paths OR a list of image paths.
Videos and images can have heterogenous formats but cannot be mixed in a single command.
'''
ra.add_argument('-i', nargs='+', required=True, dest='input', help = h)

h = '''When used with an input list of videos, OUTPUT is the path to a directory
storing one resulting CSV for each processed video. OUTPUT directory should exist
before launching the program.
When used with an input list of images, OUTPUT is the path to the resulting csv
file storing a line for each detected faces. OUTPUT should have csv extension.
'''
ra.add_argument('-o', required=True, help = h, dest='output')


## Optional arguments
oa = parser.add_argument_group('optional arguments')

oa.add_argument("-h", "--help", action="help", help="show this help message and exit")

h = '''type of media to be analyzed, either a list of images (JPEG, PNG, etc...)
    or a list of videos (AVI, MP4, ...)'''
oa.add_argument('--type',
                choices = ['image', 'video'],
                default = 'video',
                help = h)


# classifier
h = '''face classifier to be used in the analysis:
   Resnet50FairFaceGRA predicts age and gender and is more accurate.
   Vggface_LSVM_YTF was used in earlier studies and predicts gender only'''
oa.add_argument ('--classifier', default='Resnet50FairFaceGRA',
                 choices = ['Resnet50FairFaceGRA', 'Vggface_LSVM_YTF'],
                 help = h)

oa.add_argument('--batch_size', default=32, type=int,
                help = '''GPU batch size. Larger values allow faster processings, but requires more GPU memory.
                Default 32 value used is fine for a Laptop Quadro T2000 Mobile GPU with 4 Gb memory.''')

# face detection
facedetection_cmdlineparser(parser)

## Video only parameters
dv = parser.add_argument_group('optional arguments to be used only with video materials (--type video)')

dv.add_argument('--ass_subtitle_export', action='store_true',
                help='export analyses into a rich ASS subtitle file which can be displayed with VLC or ELAN')

dv.add_argument('--mp4_export', action='store_true',
                help='export analyses into a a MP4 video with incrusted bounding boxes and analysis estimates')

dv.add_argument('--fps', default=None, type=float,
                help='''Amount of video frames to be processed per second.
                Remaining frames will be skipped.
                If not provided, all video frames will be processed (generally between 25 and 30 per seconds).
                Lower FPS values results in faster processing time.
                Incompatible with the --keyframes argument''')

dv.add_argument('--keyframes', action='store_true',
                help='''Face detection and analysis from video limited to video key frames.
                Allows fastest video analysis time associated to a summary with
                non uniform frame sampling rate. Incompatible with the --fps, --ass_subtitle_export or --mp4_export arguments.''')

dv.add_argument('--tracking', type=int, dest='face_detection_period',
                help='''Activate face tracking and define FACE_DETECTION_PERIOD.
                Face detection (costly) will be performed each FACE_DETECTION_PERIOD.
                Face tracking (cheap) will be performed for the remaining (FACE_DETECTION_PERIOD -1) frames.
                Tracked faces are associated to a numeric identifier.
                Tracked faces classification predictions are averaged, and more robust than frame-isolated predictions.
                To obtain the most robust result, --tracking 1 will perform face detection for each frame and track the detected faces''')

## image only parameters

# TODO : add an option for preprocessing image lists

di = parser.add_argument_group('optional arguments to be used only with image material (--type image)')
di.add_argument('--preprocessed_faces', action='store_true',
                help='''To be used when using a list of preprocessed images.
                Preprocessed images are assument to be already detected, cropped,
                centered, aligned and rescaled to 224*224 pixels.
                Result will be stored in a csv file with 1 line per image with name provided in --o argument''')

# parse arguments
args = parser.parse_args()


# deal with incompatible arguments

class ParserError:
    def __init__(self, parser):
        self.parser = parser
    def __call__(self, a1, a2):
        raise self.parser.error('%s argument cannot be mixed with %s')
pe = ParserError(parser)

# keyframe analysis incompatible, fps, tracking, mp4 and ass export
if args.keyframes:
    for a in ['fps', 'mp4_export', 'ass_subtitle_export']:
        if args.__dict__[a]:
            pe('--keyframes', '--'+a)
    if args.face_detection_period:
        pe('--keyframes', '--tracking')


if args.type == 'video':
    if args.preprocessed_faces:
        pe('--type video', '--preprocessed_faces')

    if not os.path.isdir(args.output):
        raise ValueError('OUTPUT directory %s provided to -o argument should be an existing directory' % args.output)

else: # image
    for arg in ['ass_subtitle_export', 'mp4_export', 'keyframes', 'fps']:
        if args.__dict__[arg]:
            pe('--type image', '--' + arg)
    if args.face_detection_period is not None:
        pe('--type image', '--tracking')

    out_ext = os.path.splitext(args.output)[1]
    if out_ext.lower() != '.csv':
        raise ValueError('OUTPUT value %s provided to -o argument should have .csv extension' % args.output)

# classifier constructor
if args.classifier == 'Resnet50FairFaceGRA':
    classifier = inaFaceAnalyzer.face_classifier.Resnet50FairFaceGRA()
elif args.classifier == 'Vggface_LSVM_YTF':
    classifier = inaFaceAnalyzer.face_classifier.Vggface_LSVM_YTF()

# Image list of preprocessed faces (do not need face detector)
if args.preprocessed_faces:
    df = classifier.preprocessed_img_list(args.input, batch_len=args.batch_size)
    df.to_csv(args.output, index=False)
    sys.exit(0)

# Face detection contructor
detector = facedetection_factory(args)


# List of images
if args.type == 'image':
    analyzer = inaFaceAnalyzer.inaFaceAnalyzer.ImageAnalyzer(detector, classifier, batch_len=args.batch_size)
    df = analyzer(args.input)
    df.to_csv(args.output, index=False)
    sys.exit(0)

# VIDEO
if args.face_detection_period:
    analyzer = inaFaceAnalyzer.inaFaceAnalyzer.VideoTracking(args.face_detection_period, detector, classifier, batch_len=args.batch_size)
elif args.keyframes:
    analyzer = inaFaceAnalyzer.inaFaceAnalyzer.VideoKeyframes(detector, classifier, batch_len=args.batch_size)
else:
    analyzer = inaFaceAnalyzer.inaFaceAnalyzer.VideoAnalyzer(detector, classifier, batch_len=args.batch_size)

dargs = {}
if args.fps:
    dargs = {'fps': args.fps}


nbvid = len(args.input)
for i, f in enumerate(args.input):
    # TODO: add a try/catch system
    print('analyzing video %d/%d: %s' % (i, nbvid, f))
    base, _ = os.path.splitext(os.path.basename(f))
    df = analyzer(f, **dargs)
    df.to_csv('%s/%s.csv' % (args.output, base), index=False)
    if args.ass_subtitle_export:
        inaFaceAnalyzer.display_utils.ass_subtitle_export(f, df, '%s/%s.ass' % (args.output, base), analysis_fps=args.fps)
    if args.mp4_export:
        inaFaceAnalyzer.display_utils.video_export(f, df, '%s/%s.mp4' % (args.output, base), analysis_fps=args.fps)



# bar = progressbar.ProgressBar(maxval=10000, \
# widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")

#     bar.start()
#     if track_mode == 'on':
#         for i, e in enumerate(input_files):
#             #print('\n processing file %d/%d: %s' % (i+1, len(input_files), e))
#             base, _ = os.path.splitext(os.path.basename(e))
#             info2csv(gen.detect_with_tracking(e, track_frames, n_frames, offset ), '%s/%s_tracked.csv' % (odir, base))
#             bar.update(i)
#         bar.finish()
#     else:
#         for i, e in enumerate(input_files):
#             #print('\n processing file %d/%d: %s' % (i+1, len(input_files), e))
#             base, _ = os.path.splitext(os.path.basename(e))
#             info2csv(gen(e, n_frames, offset), '%s/%s.csv' % (odir, base))
#             bar.update(i)
#         bar.finish()
