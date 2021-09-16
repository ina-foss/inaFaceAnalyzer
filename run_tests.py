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
import pandas as pd
from inaFaceGender import GenderVideo
from pandas.util.testing import assert_frame_equal
import numpy as np


class TestInaFaceGender(unittest.TestCase):
    
    def test_init(self):
        gv = GenderVideo()

    def test_basic(self):
        gv = GenderVideo()
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=25)
        ret.bb = ret.bb.map(str)
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv',
                        dtype={'conf': np.float32})
        assert_frame_equal(ret, df)

if __name__ == '__main__':
    unittest.main()
