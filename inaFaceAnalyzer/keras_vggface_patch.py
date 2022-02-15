#!/usr/bin/env python
# encoding: utf-8
#
# !!! WARNING
#
# This code contains patches for python module keras-vggface
# keras-vggface is a python module created by Refikcanmalli
# it is released under MIT LICENSE and available at
# https://github.com/rcmalli/keras-vggface
#
# The module is no more maintained and the code bellow is a minimal copy/pasted
# adaptation allowing to use the VGG16 model provided in this package
# 
# In order to use a fully patched version of keras-vggface, see this fork
# https://github.com/DavidDoukhan/keras-vggface

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
#

from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.models import Model

import numpy as np

def VGG16(input_shape):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = GlobalAveragePooling2D()(x)


    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_vgg16')  # load weights

    url = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'
    weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                            url , cache_subdir='models/vggface')
    
    model.load_weights(weights_path, by_name=True)

    return model


def preprocess_input(x):
    x_temp = np.copy(x)

    x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 93.5940
    x_temp[..., 1] -= 104.7624
    x_temp[..., 2] -= 129.1863

    return x_temp