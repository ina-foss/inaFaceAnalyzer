#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan - http://www.ina.fr/)

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

from typing import NamedTuple

class Rect(NamedTuple):
    x1 : float # left
    y1 : float # top
    x2 : float # right
    y2 : float # bottom
    @property
    def w(self):
        """ returns Rect width """
        return self.x2 - self.x1
    @property
    def h(self):
        """ returns Rect height """
        return self.y2 - self.y1
    @property
    def center(self):
        """ returns center (x,y) of Rect (x1, y1, x2, y2) """
        x1, y1, x2, y2 = self
        return ((x1 + x2) / 2), ((y1 + y2) / 2)
    @property
    def max_dim_len(self):
        """ Return max of Rect width and height """
        return max(self.h, self.w)
    def transpose(self, x, y):
        """ Returns a transposed Rect"""
        x1, y1, x2, y2 = self
        return Rect(x1 + x, y1 + y, x2 + x, y2 + y)
    def mult(self, x, y):
        x1, y1, x2, y2 = self
        return Rect(x1 * x, y1 * y, x2 * x, y2 * y)
    @staticmethod
    def from_dlib(x):
        return Rect(x.left(), x.top(), x.right(), x.bottom())
