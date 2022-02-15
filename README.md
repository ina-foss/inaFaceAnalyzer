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
A detailed listing of all options available from the command line can be obtained using the following command.
We guess you don't want to read the whole listing at this point, but you can have a look at it later 😉.
```bash
ina_face_analyzer.py -h
```
### Process all frames from a list of video (without tracking)
Video processing requires a list of input video paths, together with a directory used to store results in CSV.
Program initialization time requires several seconds, and we recommend using large list of files instead of calling the program for each file to process.
```bash
# directory storing result must exist
mkdir my_output_directory
# -i is followed by the list of video to analyze, and -o is followed by the name of the output_directory
ina_face_analyzer.py -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
# displaying the first 2 lines of the resulting CSV
head -n 2 ./my_output_directory/pexels-artem-podrez-5725953.csv 
frame,bbox,detect_conf,sex_decfunc,age_decfunc,sex_label,age_label
0,"(945, -17, 1139, 177)",0.999998927116394,8.408014,3.9126961,m,34.12696123123169
```

Resulting CSV contain several columns:
* frame: frame position in the video (here we have 5 lines corresponding to frame 0 - so 5 detected faces)
* bbox: face bounding box
* detect_conf: face detection confidence (dependent on the detection system used)
* sex_decfunc and age_decfunc: raw classifier output. Can be used to smooth results or ignored.
* sex_label: m for male and f for female
* age_label: age prediction


### Faster processing of a video
It computation time is an issue, we recommend using <code>--fps 1</code> which will process a single frame per second, instead of the whole amount of video frames. When using GPU architectures, we also recommend setting large <code>batch_size</code> values.
```bash
# here we process a single frame per second, which is 25/30 faster than processing the whole video
ina_face_analyzer.py --fps 1 --batch_size 128 -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
```
### Using Tracking
Tracking allows to lower computation time, since it is less costly than a face detection procedure. It also allows to smooth prediction results associated to a tracked face and obtain more robust estimates.
```bash
# Process 5 frames per second, use face detection for 1/3 and face tracking for 2/3 frames
ina_face_analyzer.py --fps 5 --tracking 3 -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
```

### Exporting results
Result visualization allows to validate if a give processing pipeline is suited to a specific material.
<code>--mp4_export</code> generate a new video with embeded bounding boxes and classification information.
<code>--ass_subtitle_export</code> generate a ASS subtitle file allowing to display bounding boxes and classification results in vlc or ELAN, and which is more convenient to share..

```bash
# Process 10 frames per second, use face detection for 1/2 and face tracking for 1/2 frames
# results are exported to a newly generated MP4 video and ASS subtitle
ina_face_analyzer.py --fps 10 --tracking 2 --mp4_export --ass_subtitle_export  -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
# display the resulting video
vlc ./my_output_directory/pexels-artem-podrez-5725953.mp4
# display the original video with the resulting subtitle files
vlc media/pexels-artem-podrez-5725953.mp4 --sub-file my_output_directory/pexels-artem-podrez-5725953.ass 
```

### Processing list of images
The processing of list of images can be speed up using <code>--type image</code>.
A single resulting csv will be generated with entries for each detected faces, together with a reference to its original filename path.
```bash
# process all images stored in directory media, outputs a single csv file
ina_face_analyzer.py -i media/*.jpg -o ./myresults.csv --type image
```

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
