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

FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

RUN apt-get update \
    && apt-get install -y cmake libgl1-mesa-glx ffmpeg \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY setup.py  LICENSE README.md run_tests.py ./
COPY inaFaceAnalyzer /app/inaFaceAnalyzer
COPY tests /app/tests
COPY media /app/media


RUN pip install --upgrade pip && pip install . && pip cache purge

# This line is non mandatory
# it's usefull for docker containers without internet access (it may happen)
# removing this line allows to save 500 Mo in the image
RUN echo "from inaFaceAnalyzer.remote_utils import download_all; download_all()" | python
