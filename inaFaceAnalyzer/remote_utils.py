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

# inaFaceAnalyzer models are stored remotely within github releases
# this code allows to download and use them on demand

r1_url = 'https://github.com/ina-foss/inaFaceAnalyzer/releases/download/models/'

dmodels = {
    # These 2 models are provided for face detection in opencv
    # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb
    # https://github.com/opencv/opencv/blob/4eb296655958e9241f43a7bd144d5e63759f6cea/samples/dnn/face_detector/opencv_face_detector.pbtxt
    'opencv_face_detector_uint8.pb' : r1_url,
    'opencv_face_detector.pbtxt' : r1_url,
    # Dlib's Facial landmarks prediction
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    'shape_predictor_68_face_landmarks.dat' : r1_url,
    # Resnet50 architectures trained on FairFace database
    'keras_resnet50_fairface_GRA.h5' : r1_url,
    'keras_resnet50_fairface.h5' : r1_url,
    # linear SVM trained on Youtube Faces VGG16 face embeddings
    'svm_ytf_zrezgui.hdf5' : r1_url,
    # linear SVM trained on FairFace VGG 16 embeddings
    'svm_vgg16_fairface.hdf5' : r1_url,
    # face detection and landmark estimation proovided in libfacedetection
    # https://github.com/ShiqiYu/libfacedetection
    'libfacedetection-yunet.onnx' : r1_url}

def get_remote(model_fname):
    url = dmodels[model_fname]
    return get_file(model_fname, url + model_fname, cache_subdir='inaFaceAnalyzer')

#TODO - download RC MALLI MODEL ALSO HERE
def download_all():
    # usefull at the initalisation of a Docker image
    for k in dmodels:
        get_remote(k)
