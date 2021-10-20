#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019 Ina (David Doukhan - http://www.ina.fr/)

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


import unittest
import numpy as np
from inaFaceGender.opencv_utils import video_iterator

class TestOpenCvUtils(unittest.TestCase):
    
    vid_src = './media/pexels-artem-podrez-5725953.mp4'
    
    def test_video_iterator_seq_len(self):
        self.assertEqual(len([e for e in video_iterator(self.vid_src, start=10, stop=20)]), 11)
        self.assertEqual(len([e for e in video_iterator(self.vid_src)]), 358)
        self.assertEqual(len([e for e in video_iterator(self.vid_src, time_unit='ms', start=500, stop=1000)]), 16)

    def test_video_iterator_out_types(self):
        elts = [e for e in video_iterator(self.vid_src, start=10, stop=20)]
        self.assertIsInstance(elts[0][0], int)
        self.assertIsInstance(elts[0][1], np.ndarray)
    
    def test_video_iterator_indices(self):
        vit = video_iterator(self.vid_src)
        self.assertEqual(next(vit)[0], 0)
        self.assertEqual(next(vit)[0], 1)
        
    def test_video_iterator_indices_subsampling(self):
        vit = video_iterator(self.vid_src, subsamp_coeff=2)
        self.assertEqual(next(vit)[0], 0)
        self.assertEqual(next(vit)[0], 2)

    # TODO ... start and stop offset & indices
    # def test_video_offset_indices(self):
    #     src = './media/pexels-artem-podrez-5725953.mp4'
    #     vit = video_iterator(src)
    #     self.assertEqualt(next(vit)[0], 0)
    #     self.assertEqualt(next(vit)[0], 1)
