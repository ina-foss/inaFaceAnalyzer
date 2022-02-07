#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019-2022 Ina (David Doukhan & Zohra Rezgui - http://www.ina.fr/)

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
#import glob
import os
#import warnings
#import progressbar



parser = argparse.ArgumentParser(description='inaFaceAnalyzer: detects and classify faces from media collections and store results in csv. TODO ref biblio ',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

## Required arguments
ra = parser.add_argument_group('required arguments')


ra.add_argument('-i', '--input', nargs='+', required=True, 
                help = 'list of medias to analyse. ex :/home/david/test.mp4 /tmp/mymedia.avi.')

ra.add_argument('-o', '--output-directory', required=True,
                help = 'Directory used to store results in csv')


## Optional arguments
oa = parser.add_argument_group('optional arguments')

oa.add_argument("-h", "--help", action="help", help="show this help message and exit")

oa.add_argument('-type',
                choices = ['image', 'video'],
                default = 'video',
                help = 'type of media to be analyzed, either a list of images or a list of videos')


# classifier
oa.add_argument ('--classifier', default='Resnet50FairFaceGRA',
                 choices = ['Resnet50FairFaceGRA', 'Vggface_LSVM_YTF'],
                 help = '''face classifier to be used in the analysis:
                    Resnet50FairFaceGRA predicts age and gender and is more accurate.
                    Vggface_LSVM_YTF was used in earlier studies and predicts gender only''')


## Face Detection related argument
# detect; min size; confidence

da = parser.add_argument_group('optional arguments related to face detection')

da.add_argument ('--face_detector', default='LibFaceDetection',
                 choices=['LibFaceDetection', 'OcvCnnFacedetector'],
                 help='''face detection module to be used:
                     LibFaceDetection can take advantage of GPU acceleration and has a higher recall.
                     OcvCnnFaceDetector is embed in OpenCV has a better precision''')

da.add_argument('--face_detection_confidence', type=float,
                help='''minimal confidence threshold to be used for face detection.
                    Default values are 0.98 for LibFaceDetection and 0.65 for OcvCnnFacedetector''')


da.add_argument('--min_face_size_px', default=30, type=int,
                help='minimal absolute size in pixels of the faces to be considered for the analysis. Optimal classification results are obtained for sizes above 75 pixels.')

da.add_argument('--min_face_size_percent', default=0, type=float,
                help='minimal relative size (percentage between 0 and 1) of the faces to be considered for the analysis with repect to image frames minimal dimension (generally height for videos)')


## Video only parameters

# ass subtitle

# FPS

# Tracking


#parser.add_argument('-s', '--time_offset', help = 'time in milliseconds from which we begin extraction of the frames in video', required=False)
#parser.add_argument('-f', '--nframes', help = 'process every n frames', required = False)
#parser.add_argument('-t', '--mode', help = 'With or without tracking mode', choices = ['on','off'], required = False)

#parser.add_argument('-k', '--ktracking', help = 'Used in case of tracking: re-detect faces every k frames', required= False)


args = parser.parse_args()

print(args['face-detection-confidence'])

exit()

input_files = []
for e in args.input:
    input_files += glob.glob(e)
assert len(input_files) > 0, 'No existing media selected for analysis! Bad values provided to -i (%s)' % args.input

odir = args.output_directory
assert os.access(odir, os.W_OK), 'Directory %s is not writable!' % odir


if args.mode == 'on' and args.ktracking is None :
    parser.error("--mode requires --ktracking ! ")
    
    
n_frames = args.nframes
if n_frames:
    n_frames = int(n_frames)
else: 
    n_frames = 1
offset = args.time_offset
if offset:
    offset = int(offset)

track_mode = args.mode
track_frames = int(args.ktracking)

from inaFaceAnalyzer import GenderVideo, info2csv

gen = GenderVideo()

bar = progressbar.ProgressBar(maxval=10000, \
widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    bar.start()
    if track_mode == 'on':       
        for i, e in enumerate(input_files):
            #print('\n processing file %d/%d: %s' % (i+1, len(input_files), e))
            base, _ = os.path.splitext(os.path.basename(e))
            info2csv(gen.detect_with_tracking(e, track_frames, n_frames, offset ), '%s/%s_tracked.csv' % (odir, base))
            bar.update(i)
        bar.finish()
    else:       
        for i, e in enumerate(input_files):
            #print('\n processing file %d/%d: %s' % (i+1, len(input_files), e))
            base, _ = os.path.splitext(os.path.basename(e))
            info2csv(gen(e, n_frames, offset), '%s/%s.csv' % (odir, base))
            bar.update(i)
        bar.finish()
