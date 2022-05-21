Data Structures
---------------

This section present the 3 internal data structures implemented in inaFaceAnalyzer.

- :class:`inaFaceAnalyzer.rect.Rect` : a rectangle implementation usefull for manipulating face bounding boxes
- :class:`inaFaceAnalyzer.face_detector.Detection` : obtained from face detection systems, containing face bounding boxes together with face detection confidence
- :class:`inaFaceAnalyzer.face_tracking.TrackDetection` : an extension of :class:`Detection` used when combining face detection and tracking

The remaining data structures used to exchange data between modules are based
on :class:`pandas.DataFrame`, allowing easy exports to various table formats
(see pandas's `documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_).

.. autonamedtuple:: inaFaceAnalyzer.rect.Rect
    :members:
    :special-members: __contains__


.. autonamedtuple:: inaFaceAnalyzer.face_detector.Detection
    :members:

.. autonamedtuple:: inaFaceAnalyzer.face_tracking.TrackDetection
    :members:
