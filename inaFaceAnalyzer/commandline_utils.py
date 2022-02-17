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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import inaFaceAnalyzer

epilog = '''
If you are using inaFaceAnalyzer in your research-related documents, please cite
the current version number used (%s) together with a reference to the following
paper: David Doukhan and Thomas Petit (2022). inaFaceAnalyzer: a Python toolbox
for large-scale face-based description of gender representation in media with
limited gender, racial and age biases. Submitted to JOSS - The journal of Open
Source Software (submission in progress).
''' % inaFaceAnalyzer.__version__


def new_parser(description):
    parser = ArgumentParser(description=description,
                            epilog=epilog,
                            formatter_class= ArgumentDefaultsHelpFormatter,
                            add_help=False)
    return parser
