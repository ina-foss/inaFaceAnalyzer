#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019 Ina (David Doukhan - http://www.ina.fr/)

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


import os
from setuptools import setup, find_packages
import versioneer

KEYWORDS = '''
gender-equality
gender-classification'''.strip().split('\n')

CLASSIFIERS=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Topic :: Multimedia :: Video',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Sociology',
]

DESCRIPTION='Detect faces in video streams and does gender classification'
LONGDESCRIPTION='''Detect faces in video streams and does gender classification. Designed for estimating the visual presence of women and men within TV programs.

```bibtex
@techreport{rezgui2019carthage,
  type = {Msc. Thesis},
  author = {Zohra Rezgui},
  title = {Détection et classification de visages pour la description de l’égalité femme-homme dans les archives télévisuelles},
  submissiondate = {2019/11/19},
  year = {2019},
  url = {https://www.researchgate.net/publication/337635267_Rapport_de_stage_Detection_et_classification_de_visages_pour_la_description_de_l'egalite_femme-homme_dans_les_archives_televisuelles},
  institution = {Higher School of Statistics and Information Analysis, University of Carthage}
}

@inproceedings{doukhan2019estimer,
  title={Estimer automatiquement les diff{\'e}rences de repr{\'e}sentation existant entre les femmes et les hommes dans les m{\'e}dias},
  author={Doukhan, David and Rezgui, Zohra and Poels, G{\'e}raldine and Carrive, Jean},
  year={2019}
}

'''

setup(
    name = "inaFaceAnalyzer",
    version = versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author = "David Doukhan, Zohra Rezgui",
    author_email = "david.doukhan@gmail.com, zohra.rzg@gmail.com",
    test_suite="run_tests.py",
    description = DESCRIPTION,
    license = "MIT",
    install_requires=['opencv-contrib-python', 'dlib', 'pandas', 'sklearn', 'h5py', 'matplotlib', 'onnxruntime-gpu', 'keras-vggface @ https://github.com/DavidDoukhan/keras-vggface/archive/refs/tags/vddk-0.1.tar.gz'],
    url = "https://github.com/ina-foss/inaFaceAnalyzer",
    packages=['inaFaceAnalyzer'],
    keywords = KEYWORDS,
    #packages = find_packages(),
    include_package_data = True,
    data_files = ['LICENSE'],
    long_description=LONGDESCRIPTION,
    long_description_content_type='text/markdown',
# TODO: add webcam script
#    scripts=[os.path.join('scripts', script) for script in \
#             ['ina_face_gender.py']],
    classifiers=CLASSIFIERS,
    python_requires='>=3.6.9',

)
