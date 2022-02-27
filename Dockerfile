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

# !!! IMPORTANT !!!
# This docker image have been tested using the following configuration
# NVIDIA-SMI 460.73.01    Driver Version: 460.73.01    CUDA Version: 11.2

FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

MAINTAINER David Doukhan david.doukhan@gmail.com

RUN apt-get update \
    && apt-get install -y cmake libgl1-mesa-glx ffmpeg \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

# download models to be used by default
# this part is non mandatory, costs 500 Mo, and ease image usage
ARG u='https://github.com/ina-foss/inaFaceAnalyzer/releases/download/models/'
ADD ${u}opencv_face_detector_uint8.pb \
    ${u}opencv_face_detector.pbtxt \
    ${u}shape_predictor_68_face_landmarks.dat \
    ${u}keras_resnet50_fairface_GRA.h5 \
    ${u}libfacedetection-yunet.onnx \
    /root/.keras/inaFaceAnalyzer/

# make models available to non-root users
RUN chmod +x /root/
RUN chmod +r /root/.keras/inaFaceAnalyzer/*

WORKDIR /app
COPY setup.py setup.cfg  LICENSE MANIFEST.in README.md test_inaFaceAnalyzer.py versioneer.py ./
COPY inaFaceAnalyzer /app/inaFaceAnalyzer
COPY tests /app/tests
COPY media /app/media
COPY scripts /app/scripts
COPY tutorial_API_notebooks /app/tutorial_API_notebooks
# required for keeping track of version - only 12 Mo
COPY .git ./.git


RUN pip install --upgrade pip && pip install . && pip cache purge
