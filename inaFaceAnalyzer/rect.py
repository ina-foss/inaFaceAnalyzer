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
import dlib
import numpy as np

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
    def area(self):
        return self.w * self.h
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

    def intersect(self, r):
        x1, y1, x2, y2 = self
        ret = Rect(max(x1, r.x1), max(y1, r.y1), min(x2, r.x2), min(y2, r.y2))
        if ret.h <= 0 or ret.w <= 0:
            return Rect(0,0,0,0)
        return ret

    def iou(self, r):
        inter = self.intersect(r).area
        union = self.area + r.area - inter
        return inter / union

    def __contains__(self, point):
        x1, y1, x2, y2 = self
        x, y = point
        return x >= x1 and x <= x2 and y >= y1 and y <= y2

    def to_int(self):
        return Rect(*[int(round(e)) for e in self])

    @staticmethod
    def from_dlib(x):
        return Rect(x.left(), x.top(), x.right(), x.bottom())

    def to_dlibInt(self):
        return dlib.rectangle(*[e for e in self.to_int()])

    def to_dlibFloat(self):
        return dlib.drectangle(*self)

    @property
    def square(self):
        """ returns the smallest square containing the rectangle"""
        offset = self.max_dim_len / 2
        xc, yc = self.center
        return Rect(xc - offset, yc - offset, xc + offset, yc + offset)

    def scale(self, scale_prct):
        """ scale Rectangle according to scale percentage scale_prct"""
        w, h = (self.w, self.h)
        x1, y1, x2, y2 = self
        xdiff = (w * scale_prct - w) / 2
        ydiff = (h * scale_prct -h) / 2
        return Rect(x1 - xdiff, y1 - ydiff, x2 + xdiff, y2 + ydiff)

