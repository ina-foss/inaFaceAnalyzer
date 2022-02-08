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

import cv2
from inaFaceAnalyzer import GenderImage
from inaFaceAnalyzer.face_classifier import Resnet50FairFaceGRA



gi = GenderImage(face_classifier=Resnet50FairFaceGRA())

font = cv2.FONT_HERSHEY_SIMPLEX


video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = gi.detect_and_classify_faces_from_frame(frame)

    # Draw a rectangle around the faces
    for e in faces.itertuples():
        x1, y1, x2, y2 = e.bbox

        text = 'sex: %s - %.1f; age: %.1f' % (e.sex_label, e.sex_decfunc, e.age_label)
        if e.sex_label == 'm': # blue
            color = (0,0,255)
        else: # red
            color = (255,0,0)
        cv2.putText(frame,text,(x1 - 100, y1 - 10 ), font, 0.7, color,2,cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
