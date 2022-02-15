# inaFaceAnalyzer: a Python toolbox for large-scale face-based description of gender representation in media with limited gender, racial and age biases
[![test py 3.7](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-7.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-7.yml)
[![test py 3.8](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-8.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-8.yml)
[![test py 3.9](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-9.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-9.yml)
<!-- [![test py 3.10](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-10.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-10.yml) -->

`inaFaceAnalyzer` is a Python toolbox designed for large-scale analysis of faces in image or video streams.
It provides a modular processing pipeline allowing to predict age and gender from faces.
Results can be exported to tables, augmented video streams, or rich ASS subtitles.
`inaFaceAnalyzer` is designed with speed in mind to perform large-scale media monitoring campaigns.
The trained age and gender classification model provided is based on a `ResNet50` architecture.
Evaluation results are highly competitive with respect to the current state-of-the-art, and appear to reduce gender, age and racial biases.

Should you need further details regarding this work, please refer to the following [paper](https://github.com/ina-foss/inaFaceAnalyzer/blob/master/paper.md):

```bibtex
@journal{doukhan2022joss,
  author = {David Doukhan and Thomas Petit},
  title = {inaFaceAnalyzer: a Python toolbox for large-scale face-based description of gender representation in media},
  journal = {JOSS - The journal of Open Source Software (currently being reviewed)},
  year = {submission in progress}
}
```

Have a look to sibling project [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter).


## Installation

```
apt-get install cmake ffmpeg libgl1-mesa-glx
git clone https://github.com/ina-foss/inaFaceAnalyzer.git
cd inaFaceAnalyzer
pip install .
./test_inaFaceAnalyzer.py # to check that the installation is ok
```

## Using inaFaceAnalyzer command line program

Most common processings can be done using the script <code>ina_face_analyzer.py</code> provided with the distribution.
Some quick-starters commands are detailled bellow :

### Displaying detailed manual
A detailed (and definitely long) listing of all options available from the command line can be obtained using the following command.
We guess you don't want to read the whole listing at this point, but you can have a look at it later 😉.
```bash
ina_face_analyzer.py -h
```
### Process all frames from a list of video (without tracking)
Video processing requires at least a variable lenght list of input video paths, together with a directory where analysis results will be stored in CSV.
The command line program needs to load classification models before doing computations, and this step may takes several seconds.
We recommend to use long lists of files when using the program.
The following command process all available video frames and may be slow.
```bash
# directory storing result must exist
mkdir my_output_directory
# -i is followed by the list of video to analyze, and -o is followed by the name of the output_directory
ina_face_analyzer.py -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
```

The resulting CSV contain several columns
* frame: frame position in the video (here we have 5 lines corresponding to frame 0 - so 5 detected faces)
* bbox: face bounding box
* detect_conf: face detection confidence (dependent on the detection system used)
* sex_decfunc and age_decfunc: raw classifier output. Can be used to smooth results or ignored.
* sex_label: m for male and f for female
* age_label: age prediction

```bash
# display resulting CSV with same basename than input file
cat ./my_ouput_directory/pexels-artem-podrez-5725953.csv
#frame,bbox,detect_conf,sex_decfunc,age_decfunc,sex_label,age_label
#0,"(945, -17, 1139, 177)",0.999998927116394,8.408014,3.9126961,m,34.12696123123169
#0,"(71, 119, 272, 320)",0.9999958872795105,-13.514768,3.1241806,f,26.24180555343628
#0,"(311, 163, 491, 343)",0.99997878074646,-11.023162,3.035822,f,25.358219146728516
#0,"(558, 202, 728, 371)",0.9999741911888123,8.824918,2.910231,m,24.10231113433838
#0,"(745, 23, 930, 208)",0.9815391302108765,-7.9449368,2.427218,f,19.27217960357666
#1,"(946, -17, 1138, 175)",0.9999986290931702,7.916046,3.8727958,m,33.72795820236206
#1,"(66, 117, 274, 324)",0.9999975562095642,-12.532552,3.105084,f,26.0508394241333
#1,"(558, 202, 730, 373)",0.9999759793281555,7.8327827,2.8403559,m,23.4035587310791
#1,"(311, 164, 491, 344)",0.9999755620956421,-10.973633,2.9896245,f,24.896245002746582
```

### Faster processing of a video
It computation time is an issue, we recommend using <code>--fps 1</code> which will process a single frame per second, instead of the whole amount of video frames. When using GPU architectures, we also recommend setting large <code>batch_size</code> values.
```
ina_face_analyzer.py --fps 1 --batch_size 128 -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
```
### Using Tracking

### Exporting results

### Processing list of images



Using <code>ina_face_analyzer.py -h</code> display a detailed (definitely long and detailled) listing of all options available from the command-line.


tracking and provide some of the best available processing options - which may
be a little slow. 



## Using inaFaceAnalyzer API


## CREDITS
This work has been partially funded by the French National Research Agency (project GEM : Gender Equality Monitor : ANR-19-CE38-0012) and by European Union's Horizon 2020 research and innovation programme (project [MeMAD](https://memad.eu) : H2020 grant agreement No 780069).

We acknowledge contributions from [Zohra Rezgui](https://github.com/ZohraRezgui) who trained first models and wrote the first piece of code that lead to inaFaceAnalyzer during her internship at INA.
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
```
