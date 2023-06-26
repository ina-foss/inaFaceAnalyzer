#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019-2023 Ina (David Doukhan - http://www.ina.fr/)

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

"""
Module :mod:`inaFaceAnalyzer.face_extractor` contains functions allowing to
extract detected faces to image directories and to apply face preprocessing
methods on resulting images.
"""

import glob
from .face_detector import PrecomputedDetector
from .face_preprocessing import preprocess_face
from .opencv_utils import imwrite_rgb

def face_extractor(df, stream, output_dir, oshape, bbox2square, bbox_scale, face_aligner, ext='png', verbose=False):
        '''
        Extract faces found in "stream"
        '''
        iframe, frame = next(stream)

        detector = PrecomputedDetector(list(df.bbox))

        for ituple, t in enumerate(df.itertuples()):
            while iframe != t.frame:
                iframe, frame = next(stream)
            detection = detector(frame)
            assert len(detection) == 1, len(detection)
            detection = detection[0]
            img, _ = preprocess_face(frame, detection, bbox2square, bbox_scale, face_aligner, oshape, False)
            dst = '%s/%08d.%s' % (output_dir, ituple, ext)
            if verbose:
                print(dst)
            imwrite_rgb(dst, img)
        df['fname'] = sorted(glob.glob(output_dir + '/*.' + ext))
        return df
