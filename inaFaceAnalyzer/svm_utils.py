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

import h5py
import numpy as np
from sklearn.svm import LinearSVC

# Load linear SVM serialized in a custom HDF5 format
# more robust than pickle
def svm_load(hdf5_fname):
    f = h5py.File(hdf5_fname, 'r')
    svm = LinearSVC()
    svm.classes_ = np.array(f['linearsvc/classes'][:]).astype('<U1')
    svm.intercept_ = f['linearsvc/intercept'][:]
    svm.coef_ = f['linearsvc/coef'][:]
    return svm

# TODO: test the load & dump pipeline
def svm_dump(hdf5_fname, model):
    with h5py.File(hdf5_fname, 'w') as fid:
        fid['linearsvc/intercept'] = model.intercept_
        fid['linearsvc/coef'] = model.coef_
        fid.create_dataset('linearsvc/classes', data = model.classes_.astype(h5py.special_dtype(vlen=str) ))
