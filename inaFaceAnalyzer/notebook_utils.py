#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2022 Ina (David Doukhan - http://www.ina.fr/)

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

"""
Module :mod:`inaFaceAnalyzer.notebook_utils` contain simple functions allowing to display video in jupyter
and google collab's notebooks. These functions can be used only in a notebook environment
"""


try:
    # To be used only in ipython/collab environments
    from IPython.core.display import display
    from IPython.display import HTML
    from base64 import b64encode

    def notebook_display_remote_vid(video_path, width=600):
        """
        Display a remote video in a jupyter notebook
        Can be used only in a jupyter environment

        Args:
            video_path (str): path or url to the remote video to be displayed
            width (int, optional): video width in the notebook. Defaults to 600 pixels.
        """
        data = '<div align="middle"> <video width=%d controls> <source src=%s> </video></div>' % (width, video_path)
        display(HTML(data))

    def notebook_display_local_vid(video_path, width = 600):
        """
        Display a local video in a Jupyter Notebook
        Compatible with google collab's notebooks

        Args:
            video_path (str): path to the local video.
            width (int, optional): video width in the notebook. Defaults to 600 pixels.

        """

        # thanks https://androidkt.com/how-to-capture-and-play-video-in-google-colab/
        video_file = open(video_path, "r+b").read()
        video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
        return HTML(f"""<video width={width} controls><source src="{video_url}"></video>""")

except:
    pass
