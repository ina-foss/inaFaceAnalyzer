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

class Rect(NamedTuple):
    """
    This class is an internal data structure allowing to manipulate rectangle shapes
    """

    #: left
    x1 : float
    #: top
    y1 : float
    #: right
    x2 : float
    #: bottom
    y2 : float

    @property
    def w(self):
        """ self width """
        return self.x2 - self.x1

    @property
    def h(self):
        """ self height """
        return self.y2 - self.y1

    @property
    def center(self):
        """ center (x,y) of self (x1, y1, x2, y2) """
        x1, y1, x2, y2 = self
        return ((x1 + x2) / 2), ((y1 + y2) / 2)

    @property
    def area(self):
        """ self Surface area """
        return self.w * self.h

    @property
    def max_dim_len(self):
        """ max (width, height)"""
        return max(self.h, self.w)

    @property
    def square(self):
        """ returns the smallest square containing the rectangle"""
        offset = self.max_dim_len / 2
        xc, yc = self.center
        return Rect(xc - offset, yc - offset, xc + offset, yc + offset)


    def transpose(self, xoffset, yoffset):
        """
        Translation

        Args:
            xoffset (float): horizontal offset.
            yoffset (float): vertical offset.
        Returns:
            Rect :
        """
        x1, y1, x2, y2 = self
        return Rect(x1 + xoffset, y1 + yoffset, x2 + xoffset, y2 + yoffset)

    def mult(self, x, y):
        """
        Multiply self coordinates by horizontal and vertical scaling factors
        Usefull for converting [0...1] coordinates to image frame dimensions in pixels

        Args:
            x (float: horizontal scaling factor.
            y (float): vertical scaling factor.

        Returns:
            Rect:

        """
        x1, y1, x2, y2 = self
        return Rect(x1 * x, y1 * y, x2 * x, y2 * y)

    def intersect(self, r):
        """
        Rectangle intersection between self and r

        Args:
            r (Rect):

        Returns:
            Rect:

        """
        x1, y1, x2, y2 = self
        ret = Rect(max(x1, r.x1), max(y1, r.y1), min(x2, r.x2), min(y2, r.y2))
        if ret.h <= 0 or ret.w <= 0:
            return Rect(0,0,0,0)
        return ret

    def iou(self, r):
        """
        Intersection Over Union between self and r

        Args:
            r (Rect):

        Returns:
            float:

        """
        inter = self.intersect(r).area
        union = self.area + r.area - inter
        return inter / union

    def __contains__(self, point):
        """
        point contained in self

        Args:
            point (tuple): (x,y)

        Returns:
            bool: True if point is in self, else False

        """

        x1, y1, x2, y2 = self
        x, y = point
        return x >= x1 and x <= x2 and y >= y1 and y <= y2

    def to_int(self):
        """
        Convert self coordinates (float) to the nearest int values
        Returns:
            Rect:
        """
        return Rect(*[int(round(e)) for e in self])

    @staticmethod
    def from_dlib(x):
        """
        create Rect from dlib's rectangle instance

        Args:
            x (dlib.rectangle or dlib.drectangle):

        Returns:
            Rect:

        """
        return Rect(x.left(), x.top(), x.right(), x.bottom())


    def to_dlibInt(self):
        """
        Convert self to dlib.rectangle (int)

        Returns:
            dlib.rectangle:

        """
        return dlib.rectangle(*[e for e in self.to_int()])

    def to_dlibFloat(self):
        """
        Convert self to dlib.drectangle (float)

        Returns:
            dlib.drectangle:

        """
        return dlib.drectangle(*self)


    def scale(self, scale_prct):
        """
        scale self according to a given scale percentage

        Args:
            scale_prct (float):

        Returns:
            Rect:

        """
        w, h = (self.w, self.h)
        x1, y1, x2, y2 = self
        xdiff = (w * scale_prct - w) / 2
        ydiff = (h * scale_prct -h) / 2
        return Rect(x1 - xdiff, y1 - ydiff, x2 + xdiff, y2 + ydiff)

