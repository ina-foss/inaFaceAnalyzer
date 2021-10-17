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

from tensorflow.keras.utils import get_file

# inaFaceGender models are stored remotely within github releases
# this code allows to download and use them on demand

url_r1 = 'https://github.com/ina-foss/inaFaceGender/releases/download/models-init/'
url_r2 = 'https://github.com/ina-foss/inaFaceGender/releases/download/models-init-2/'

dmodels = {
    # These 2 models are provided for face detection in opencv
    # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb
    # https://github.com/opencv/opencv/blob/4eb296655958e9241f43a7bd144d5e63759f6cea/samples/dnn/face_detector/opencv_face_detector.pbtxt
    'opencv_face_detector_uint8.pb' : url_r2,
    'opencv_face_detector.pbtxt' : url_r2,
    # Dlib's Facial landmarks prediction
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    'shape_predictor_68_face_landmarks.dat' : url_r2,
    # Resnet50 architectures trained on FairFace database
    'keras_resnet50_fairface_GRA.h5' : url_r1,
    'keras_resnet50_fairface.h5' : url_r1,
    # linear SVM trained on VGG16 face embeddings
    'svm_ytf_zrezgui.hdf5': url_r2
    }


def get_remote(model_fname):
    url = dmodels[model_fname]
    return get_file(model_fname, url + model_fname)
