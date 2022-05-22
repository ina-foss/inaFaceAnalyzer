Installation
------------


`inaFaceAnalyzer` requires Python version between 3.7 and 3.9.
Python 3.10 is not yet supported due to `onnxruntime-gpu` dependency.


Installing from sources on ubuntu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ apt-get install cmake ffmpeg libgl1-mesa-glx
    $ git clone https://github.com/ina-foss/inaFaceAnalyzer.git
    $ cd inaFaceAnalyzer
    $ pip install .
    $ ./test_inaFaceAnalyzer.py # to check that the installation is ok


Installing from pypi on ubuntu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For GPU support, cuda, cudnn and nvidia drivers should be already installed

.. code-block:: console

    $ apt-get install cmake ffmpeg libgl1-mesa-glx
    $ pip install inaFaceAnalyzer


Using docker image
^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ # download latest docker image from dockerhub
    $ docker pull inafoss/inafaceanalyzer
    $ # run docker image. setting --gpu argument allows to take advantage of
    $ # GPU acceleration (non mandatory)
    $ docker run -it --gpus=all inafoss/inafaceanalyzer /bin/bash
    $ # lauch unit tests (non mandatory but recommended)
    $ python test_inaFaceAnalyzer.py
    $ # use any program or API
    $ ina_face_analyzer.py -h
