Exporting and Displaying Analysis results
-----------------------------------------

Analysis results can be exported to tables or augmented video streams.
They can be displayed in external softwares, as well as in Google Collab or Jupyter notebooks.


Exporting analysis results to table formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analysis pipelines defined in module :mod:`inaFaceAnalyzer.inaFaceAnalyzer`
return frame-coded results as :class:`pandas.DataFrame`
(see `documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_).
They can be exported to any table format supported by pandas (csv, excell, json, etc..)

>>> from inaFaceAnalyzer.inaFaceAnalyzer import VideoAnalyzer
>>> # create a video analyzer instance (costly, do it a single time)
>>> va = VideoAnalyzer()
>>> # perform video analysis, analysing a single image frame per second (fps=1)
>>> df = va('./media/pexels-artem-podrez-5725953.mp4', fps=1)
>>> # export pandas Dataframe result to csv
>>> df.to_csv('./myanalysis.csv')


Visualizing analysis results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: inaFaceAnalyzer.display_utils
    :members:

Playing videos in notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: inaFaceAnalyzer.notebook_utils
    :members:
