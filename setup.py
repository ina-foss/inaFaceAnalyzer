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


import os
from setuptools import setup #, find_packages
import versioneer

KEYWORDS = '''
gender-equality
gender-classification'''.strip().split('\n')

CLASSIFIERS=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
#    'Programming Language :: Python :: 3.10', # dependency onnxruntime-gpu does not support Python 3.10 yet
    'Topic :: Multimedia :: Video',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Image Processing',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Sociology',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Sociology',
]

DESCRIPTION='inaFaceAnalyzer is a Python toolbox for large-scale face-based \
analysis of image and video streams. It provides fast API and command line programs allowing to perform \
face detection, face tracking, gender and age prediction, and export to CSV or rich ASS subtitles'

# read the contents of your README file
with open('README.md', 'r') as fid:
    long_description = fid.read()


setup(
    name = "inaFaceAnalyzer",
    version = versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author = "David Doukhan, Zohra Rezgui, Thomas Petit",
    author_email = "david.doukhan@gmail.com, zohra.rzg@gmail.com, tpetit@ina.fr",
    test_suite="test_inaFaceAnalyzer.py",
    description = DESCRIPTION,
    license = "MIT",
    install_requires=['opencv-contrib-python', 'dlib', 'pandas', 'sklearn',
                      'h5py', 'matplotlib', 'onnxruntime-gpu', 'cheetah3', 'av', 'tensorflow'],
    url = "https://github.com/ina-foss/inaFaceAnalyzer",
    packages=['inaFaceAnalyzer'],
    keywords = KEYWORDS,
    #packages = find_packages(),
    include_package_data = True,
    data_files = ['LICENSE'], #'inaFaceAnalyzer/template.ass'],
    long_description = long_description,
    long_description_content_type='text/markdown',
    scripts=[os.path.join('scripts', script) for script in \
             ['ina_face_analyzer.py', 'ina_face_analyzer_webcam_demo.py']],
    classifiers=CLASSIFIERS,
    python_requires='>=3.7',

)
