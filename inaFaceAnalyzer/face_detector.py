#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan & Zohra Rezgui- http://www.ina.fr/)

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

from abc import ABC, abstractmethod
from typing import NamedTuple
import cv2
import numpy as np
import onnxruntime

from .rect import Rect
from .remote_utils import get_remote
from .opencv_utils import disp_frame_shapes, disp_frame
from .libfacedetection_priorbox import PriorBox


class Detection(NamedTuple):
    bbox : Rect
    detect_conf : float

class DetectionEyes(NamedTuple):
    """
    Contains a detection (Rect bounding box & detection confidence)
    + eyes coordinates (x1, y1, x2, y2) for left eye and right eye
    """
    bbox : Rect
    detect_conf : float
    eyes : Rect


class FaceDetector(ABC):
    def __init__(self, minconf, min_size_px, min_size_prct, padd_prct):
        self.minconf = minconf
        self.min_size_px = min_size_px
        self.min_size_prct = min_size_prct
        self.padd_prct = padd_prct

    def __call__(self, frame, verbose=False):

        tmpframe = frame

        if self.padd_prct:
            tmpframe, yoffset, xoffset = _blackpadd(frame, self.padd_prct)

        lret = self._call_imp(tmpframe)


        # filter detected faces to return only faces with a dimension length
        # (absolute or relative)
        # face classification algorithms may be affected by small face sizes
        min_frame_dim = min(frame.shape[:2])
        min_face_size = max(self.min_size_px, self.min_size_prct * min_frame_dim)
        if min_face_size > 0:
            lret = [e for e in lret if e.bbox.max_dim_len >= min_face_size]

        if self.padd_prct:
            lret = [e._replace(bbox=e.bbox.transpose(-xoffset, -yoffset)) for e in lret]

        if verbose:
            disp_frame_shapes(frame, [e.bbox for e in lret], [])
            for detection in lret:
                x1, y1, x2, y2 = [e for e in detection.bbox.to_int()]
                print(detection)
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, frame.shape[1])
                y2 = min(y2, frame.shape[0])
                disp_frame(frame[y1:y2, x1:x2, :])


        return lret

    @classmethod
    @abstractmethod
    def output_type() : pass

    @abstractmethod
    def _call_imp(self, frame): pass

    def most_central_face(self, frame, contain_center=True, verbose=False):
        """
        This method returns the detected face which is closest from the center
        if contain_center == True, the returned face MUST include image center
        Usefull for preprocessign face datasets containing several faces per image
        Parameters
        ----------
        frame : nd.array (height, width, 3)
            RGB image data.
        verbose : boolean, optional
            Display detected faces. The default is False.
        Returns
        -------
        A Detection named tuple if a face maching the conditions has been detected else None
        """

        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)

        # keep faces containing image center
        if contain_center:
            faces = [f for f in self(frame, verbose) if frame_center in f.bbox]
        else:
            faces = [f for f in self(frame, verbose)]

        if len(faces) == 0:
            return None

        ldists = [_sqdist(f.bbox.center, frame_center) for f in faces]
        am = np.argmin(ldists)

        return faces[am]


    def get_closest_face(self, frame, ref_bbox, min_iou=.7, squarify=True, verbose=False):
        """
        Some face corpora may contain pictures with several faces
        together with the reference bounding box of annotated faces
        This function is aimed at preprocessing such face corpora
        Automatic face detection is used, and the detected face with the largest
        iou with the reference bounding box is returned
        if no detected face corresponds to this IOU criteria, returns None
        """
        if not isinstance(ref_bbox, Rect):
            ref_bbox = Rect(*ref_bbox)

        # get closest detected faces from ref_bbox
        if squarify:
            f = lambda x: x.square
        else:
            f = lambda x: x

        ref_bbox = f(ref_bbox)

        lfaces = self(frame, verbose)
        if len(lfaces) == 0:
            return None

        liou = [f(ref_bbox).iou(f(detection.bbox)) for detection in lfaces]


        if verbose:
            print([f(detection.bbox) for detection in lfaces])
            print('liou', liou)
        am = np.argmax(liou)
        if liou[am] < min_iou:
            return None
        return lfaces[am]



def _sqdist(p1, p2):
    '''
    return squared distance between points p1(x,y) and p2(x,y)
    '''
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _blackpadd(frame, paddpercent):
    # add black around image
    y, x, z = frame.shape
    #offset = int(max(x, y) * paddpercent)
    xoffset = int(x * paddpercent)
    yoffset = int(y * paddpercent)
    ret = np.zeros((y + 2 * yoffset, x + 2 * xoffset, z), dtype=frame.dtype)
    ret[yoffset:(y + yoffset), xoffset:(x + xoffset), :] = frame
    return ret, yoffset, xoffset



class OcvCnnFacedetector(FaceDetector):
    """
    opencv default CNN face detector
    """
    output_type = Detection

    def __init__(self, minconf=0.65, min_size_px=30, min_size_prct=0, padd_prct=0.15):
        """
        Parameters
        ----------
        minconf : float, optional
           minimal face detection confidence. The default is 0.65.
        paddpercent : float, optional
            input frame is copy passted within a black image with black pixel
            padding. the resulting dimensions is width * (1+2*paddpercent)

        """
        super().__init__(minconf, min_size_px, min_size_prct, padd_prct)

        fpb = get_remote('opencv_face_detector_uint8.pb')
        fpbtxt = get_remote('opencv_face_detector.pbtxt')
        self.model = cv2.dnn.readNetFromTensorflow(fpb, fpbtxt)


    def _call_imp(self, frame):
        """
        Detect faces from an image

        Parameters:
            frame (array): Image to detect faces from.

        Returns:
            faces_data (list) : List containing :
                                - the bounding box
                                - face detection confidence score
        """

        faces_data = []
        h, w, z = frame.shape

        # The CNN is intended to work images resized to 300*300
        # tests were carried on using different input size and were associated
        # to usatisfactory results
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.model.setInput(blob)
        detections = self.model.forward()

        assert(np.all(-np.sort(-detections[:,:,:,2]) == detections[:,:,:,2]))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < self.minconf:
                break

            bbox = Rect(*detections[0, 0, i, 3:7])
            # remove noisy detections coordinates
            if bbox.x1 >= 1 or bbox.y1 >= 1 or bbox.x2 <= 0 or bbox.y2 <= 0:
                continue
            if bbox.x1 >= bbox.x2 or bbox.y1 >= bbox.y2:
                continue

            # Map relative coordinates 0...1 to absolute  frame width and height
            bbox = bbox.mult(w, h)
            faces_data.append(Detection(bbox, confidence))

        return faces_data


class IdentityFaceDetector(FaceDetector):
    output_type = Detection
    def __init__(self):
        super().__init__(0, 0, 0, 0)
    def _call_imp(self, frame):
        return [Detection(Rect(0, 0, frame.shape[1], frame.shape[0]), np.NAN)]

class LibFaceDetection(FaceDetector):
    """
    This class wraps the detection model provided in libfacedetection
    See: https://github.com/ShiqiYu/libfacedetection
    """

    output_type = DetectionEyes

    def __init__(self, minconf=.98, min_size_px=30, min_size_prct=0, padd_prct=0):
        super().__init__(minconf, min_size_px, min_size_prct, padd_prct)
        model_src = get_remote('libfacedetection-yunet.onnx')
        try:
            self.model = onnxruntime.InferenceSession(model_src, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            self.model = onnxruntime.InferenceSession(model_src, providers=['CPUExecutionProvider'])
        self.nms_thresh = 0.3 # Threshold for non-max suppression
        self.keep_top_k = 750 # Keep keep_top_k for results outputing
        self.dprior = {}

    def _call_imp(self, frame):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape

        # convert to NN input
        blob = np.expand_dims(np.transpose(bgr_frame, (2, 0, 1)), axis = 0).astype(np.float32)
        # NN inference
        loc, conf, iou = self.model.run([], {'input': blob})

        # Decode bboxes and landmarks
        # TODO: set a limit of dict length ? There may be RAM issues when
        # considering a large image collection with heterogenous sizes
        if (w, h) not in self.dprior:
            self.dprior[(w, h)] = PriorBox(input_shape=(w, h), output_shape=(w, h))
        pb = self.dprior[(w, h)]

        dets = pb.decode(loc, conf, iou, self.minconf)

        # dirty hack used for google collab compatibility
        if len(dets.shape) == 3 and dets.shape[1] == 1:
            dets = dets.reshape((dets.shape[0], dets.shape[2]))
        assert len(dets.shape) == 2, dets.shape

        # NMS
        if dets.shape[0] > 0:
             # NMS from OpenCV
             keep_idx = cv2.dnn.NMSBoxes(
                  bboxes=dets[:, 0:4].tolist(),
                  scores=dets[:, -1].tolist(),
                  score_threshold=self.minconf,
                  nms_threshold=self.nms_thresh,
                  eta=1,
                  top_k=self.keep_top_k)
             dets = dets[keep_idx]
        else:
            return []

        # dirty hack used for google collab compatibility
        # it works - to be investiguated
        if len(dets.shape) == 3 and dets.shape[1] == 1:
            dets = dets.reshape((dets.shape[0], dets.shape[2]))
        assert len(dets.shape) == 2, dets.shape
        assert dets.shape[1] == 15, dets.shape

        lret = []
        for i in range(len(dets)):
            score = dets[i,-1]
            x1, y1, w, h = dets[i,:4]
            bbox = Rect(x1, y1, x1 + w, y1 + h)
            eyes = Rect(*dets[i, 4:8])
            lret.append(DetectionEyes(bbox, score, eyes))

        return lret

class PrecomputedDetector(FaceDetector):
    output_type = Detection
    def __init__(self, lbbox = []):
        super().__init__(0, 0, 0, 0)
        self.lbbox = lbbox.copy()
    def _call_imp(self, frame):
        if len(self.lbbox) == 0:
            return []
        ret = self.lbbox.pop(0)
        if isinstance(ret, tuple):
            ret = [ret]
        return [Detection(Rect(*e), None) for e in ret]


def facedetection_cmdlineparser(parser):
    '''
    Update command line parser with face detection related arguments
    Parameters
    ----------
    parser : argparse.ArgumentParser
        command line parser to be updated

    '''
    da = parser.add_argument_group('optional arguments related to face detection')

    da.add_argument ('--face_detector', default='LibFaceDetection',
                     choices=['LibFaceDetection', 'OcvCnnFacedetector'],
                     help='''face detection module to be used:
                         LibFaceDetection can take advantage of GPU acceleration and has a higher recall.
                         OcvCnnFaceDetector is embed in OpenCV. It is faster for large resolutions since it first resize input frames to 300*300. It may miss small faces''')

    da.add_argument('--face_detection_confidence', type=float,
                    help='''minimal confidence threshold to be used for face detection.
                        Default values are 0.98 for LibFaceDetection and 0.65 for OcvCnnFacedetector''')


    da.add_argument('--min_face_size_px', default=30, type=int, dest='size_px',
                    help='''minimal absolute size in pixels of the faces to be considered for the analysis.
                    Optimal classification results are obtained for sizes above 75 pixels.''')

    da.add_argument('--min_face_size_percent', default=0, type=float, dest='size_prct',
                    help='''minimal relative size (percentage between 0 and 1) of the
                    faces to be considered for the analysis with repect to image frames
                    minimal dimension (generally height for videos)''')

    da.add_argument('--face_detection_padding', default=None, type=float, dest='face_detection_padding',
                    help='''Black padding percentage to be applied to image frames before face detection.
                    0.15 Padding may help detecting large faces occupying the whole image with OcvCnnFacedetector.
                    Default padding values are 0.15 for OcvCnnFacedetector and 0 for LibFaceDetection''')

def facedetection_factory(args):
    '''
    Instanciate a face detection object from parsed command line arguments

    Parameters
    ----------
    args : Namespace
        Namespace containing fields face_detector, face_detection_confidence,
        min_face_size_px, min_face_size_percent

    Returns
    -------
    instance of class FaceDetector

    '''
    dargs = {'min_size_px': args.size_px, 'min_size_prct': args.size_prct}
    if args.face_detection_padding is not None:
        dargs['padd_prct'] = args.face_detection_padding
    if args.face_detection_confidence:
        dargs['minconf'] = args.face_detection_confidence
    if args.face_detector == 'LibFaceDetection':
        detector = LibFaceDetection(**dargs)
    elif args.face_detector == 'OcvCnnFacedetector':
        detector = OcvCnnFacedetector(**dargs)

    return detector
