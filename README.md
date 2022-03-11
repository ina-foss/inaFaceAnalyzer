# inaFaceAnalyzer: a Python toolbox for large-scale face-based description of gender representation in media with limited gender, racial and age biases
[![test py 3.7](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-7.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-7.yml)
[![test py 3.8](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-8.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-8.yml)
[![test py 3.9](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-9.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-9.yml)
<!-- [![test py 3.10](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-10.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-10.yml) -->
[![PyPI version](https://badge.fury.io/py/inaFaceAnalyzer.svg)](https://badge.fury.io/py/inaFaceAnalyzer)
[![Docker Pulls](https://img.shields.io/docker/pulls/inafoss/inafaceanalyzer)](https://hub.docker.com/r/inafoss/inafaceanalyzer)
[![status](https://joss.theoj.org/papers/47086151f746f32165c72870978c1398/status.svg)](https://joss.theoj.org/papers/47086151f746f32165c72870978c1398)

`inaFaceAnalyzer` is a Python toolbox designed for large-scale analysis of faces in image or video streams.
It provides a modular processing pipeline allowing to predict age and gender from faces.
Results can be exported as tables, augmented video streams, or rich ASS subtitles.
`inaFaceAnalyzer` is designed with speed in mind to perform large-scale media monitoring campaigns.
The trained age and gender classification model provided is based on a `ResNet50` architecture.
Evaluation results are highly competitive with respect to the current state-of-the-art, and appear to reduce gender, age and racial biases.

Should you need further details regarding this work, please refer to the following [paper](https://github.com/openjournals/joss-papers/blob/joss.04210/joss.04210/10.21105.joss.04210.pdf):

```bibtex
@journal{doukhan2022joss,
  author = {David Doukhan and Thomas Petit},
  title = {inaFaceAnalyzer: a Python toolbox for large-scale face-based description of gender representation in media with limited gender, racial and age biases},
  journal = {JOSS - The journal of Open Source Software (currently being reviewed)},
  year = {submission in progress}
}
```

Have a look to sibling project [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter).


## Installation
`inaFaceAnalyzer` requires Python version between 3.7 and 3.9.
Python 3.10 is not yet supported due to `onnxruntime-gpu` dependency.

### Installing from sources
```
apt-get install cmake ffmpeg libgl1-mesa-glx
git clone https://github.com/ina-foss/inaFaceAnalyzer.git
cd inaFaceAnalyzer
pip install .
./test_inaFaceAnalyzer.py # to check that the installation is ok
```
### Installing from pypi on ubuntu
```
# for GPU support, cuda, cudnn and nvidia drivers should be already installed
apt-get install cmake ffmpeg libgl1-mesa-glx
pip install inaFaceAnalyzer
```

### Using docker image
```
# download latest docker image from dockerhub
docker pull inafoss/inafaceanalyzer
# run docker image. setting --gpu argument allows to take advantage of
# GPU acceleration (non mandatory)
docker run -it --gpus=all inafoss/inafaceanalyzer /bin/bash
# lauch unit tests (non mandatory but recommended)
python test_inaFaceAnalyzer.py
# use any program or API
ina_face_analyzer.py -h
```


## Using inaFaceAnalyzer command line programs
Several scripts are provided with the distribution:
* <code>ina_face_analyzer.py</code> : can perform the most common processings provided by the framework
* <code>ina_face_analyzer_webcam_demo.py</code> : a demo script using webcam
* <code>ina_face_analyzer_distributed_server.py</code> and <code>ina_face_analyzer_distributed_worker</code> : a set of scripts allowing to perform distributed analyses on a heterogeneous cluster.

A detailed listing of all the options from the command line programs is available using the <code>-h</code> argument. We guess you don't want to read the whole listing at this point, but you can have a look at it later 😉.

### Displaying detailed manual

```bash
ina_face_analyzer.py -h
```
### Process all frames from a list of video (without tracking)
Video processing uses the <code>video</code> engine and requires a list of input video paths, together with a directory used to store results in CSV.
Program initialization time requires several seconds, and we recommend using large list of files instead of calling the program for each file to process.
```bash
# directory storing result must exist
mkdir my_output_directory
# -i is followed by the list of video to analyze, and -o is followed by the name of the output_directory
ina_face_analyzer.py --engine video -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
# displaying the first 2 lines of the resulting CSV
head -n 2 ./my_output_directory/pexels-artem-podrez-5725953.csv 
>> frame,bbox,detect_conf,sex_decfunc,age_decfunc,sex_label,age_label
>> 0,"(945, -17, 1139, 177)",0.999998927116394,8.408014,3.9126961,m,34.12696123123169
# using remote urls is also an option
ina_face_analyzer.py --engine video -i 'https://github.com/ina-foss/inaFaceAnalyzer/raw/master/media/pexels-artem-podrez-5725953.mp4' -o ./my_output_directory
```

The resulting CSV contains several columns:
* frame: frame position in the video (here we have 5 lines corresponding to frame 0 - so 5 detected faces)
* bbox: face bounding box
* detect_conf: face detection confidence (dependent on the detection system used)
* sex_decfunc and age_decfunc: raw classifier output. Can be used to smooth results or ignored.
* sex_label: m for male and f for female
* age_label: age prediction


### Faster processing of a video
If computation time is an issue, we recommend using <code>--fps 1</code> which will process a single frame per second, instead of the whole amount of video frames. When using GPU architectures, we also recommend setting large <code>batch_size</code> values.
```bash
# here we process a single frame per second, which is 25/30 faster than processing the whole video
ina_face_analyzer.py --engine video --fps 1 --batch_size 128 -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
```
### Using Tracking
Tracking allows to lower computation time, since it is less costly than a face detection procedure.
It also allows to smooth prediction results associated to a tracked face and obtain more robust estimates.
It is activated with the <code>videotracking</code> engine and requires to define <code>detect_period</code>, the time period (in frames) at which the face detector will be applied.
```bash
# Process 5 frames per second, use face detection for 1/3 and face tracking for 2/3 frames
ina_face_analyzer.py --engine videotracking --fps 5 --detect_period 3 -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
# displaying the first 2 lines of the resulting CSV
head -n 2 ./my_output_directory/pexels-artem-podrez-5725953.csv
>> frame,bbox,face_id,detect_conf,track_conf,sex_decfunc,age_decfunc,sex_label,age_label,sex_decfunc_avg,age_decfunc_avg,sex_label_avg,age_label_avg
>> 0,"(945, -17, 1139, 177)",0,0.999998927116394,,8.408026,3.9126964,m,34.12696361541748,8.391026,3.8831162,m,33.831162452697754
```
Resulting CSV will contain additional columns with <code>_avg</code> suffixes, corresponding to the smoothed estimates obtained for each tracked face. It will also contain a <code>face_id</code> with a numeric identifier associated to each tracked face.

### Exporting results
Results visualization allows to validate if a given processing pipeline is suited to a specific material.
<code>--mp4_export</code> generate a new video with embedded bounding boxes and classification information.
<code>--ass_subtitle_export</code> generate a ASS subtitle file allowing to display bounding boxes and classification results in vlc or ELAN, and which is more convenient to share..

```bash
# Process 10 frames per second, use face detection for 1/2 and face tracking for 1/2 frames
# results are exported to a newly generated MP4 video and ASS subtitle
ina_face_analyzer.py --engine videotracking --fps 10 --detect_period 2 --mp4_export --ass_subtitle_export  -i ./media/pexels-artem-podrez-5725953.mp4 -o ./my_output_directory
# display the resulting video
vlc ./my_output_directory/pexels-artem-podrez-5725953.mp4
# display the original video with the resulting subtitle files
vlc media/pexels-artem-podrez-5725953.mp4 --sub-file my_output_directory/pexels-artem-podrez-5725953.ass 
```

### Processing list of images
The processing of a list of images requires to use the <code>image</code> engine.
A single resulting CSV will be generated with entries for each detected faces, together with a reference to their original filename.
```bash
# process all images stored in directory media, outputs a single csv file
ina_face_analyzer.py --engine image -i media/* -o ./myresults.csv
# display first 2 lines of the result file
head -n 2 myresults.csv
>> frame,bbox,detect_conf,sex_decfunc,age_decfunc,sex_label,age_label
>> media/1546923312_7cc94957e8_o.jpg,"(57, 104, 435, 483)",1.0,14.436495,3.5770981,m,30.770981311798096
```

### Distributing analyses over a network

We provide two scripts allowing to perform distributed large-scale analyses.

<code>ina_face_analyzer_distributed_server.py</code> is in charge of distributing a list of documents to analyze to workers distributed over the network, and to define analysis options (fps, tracking, etc..).
The server requires 2 positional arguments: its host name (or IP) and the path to a CSV containing one line per file to process together with the destination path of the results.
Workers need to have writing permissions in the destination paths (mounted with NFS, sshfs, ...). Output directories are created on the fly if they don't exist. Jobs order is randomized before being distributed to the workers. When a destination file already exists, the corresponding analysis is skipped.

```bash
# a sample job list csv with 2 records and 4 columns
# source_path (mandatory input file path or url)
# dest_csv (mandatory output csv)
# dest_ass: to be used for exporting results to ass subtitles
# dest_mp4: to be used for exporting incrusted MP4 video
cat test.csv
>> source_path,dest_csv,dest_ass,dest_mp4
>> /home/ddoukhan/git_repos/inaFaceAnalyzer/media/pexels-artem-podrez-5725953.mp4,/tmp/csv/test1.csv,/tmp/ass/test1.ass,/tmp/mp4/test1.mp4
>> https://github.com/ina-foss/inaFaceAnalyzer/raw/master/media/pexels-artem-podrez-5725953.mp4,/tmp/csv/test2.csv,,
# the server define an analysis procedure at 1 FPS
# after initialization, it display a network adress to be passed to the workers
ina_face_analyzer_distributed_server.py blahtop.ina.fr test.csv --engine video --fps 1
>> parsing joblist test.csv
>> Total number of files to process: 2
Provide the following objet URI to remote ina_face_analyzer_distributed_workers:  PYRO:obj_4c027f06be5b40e7bcf2f3f1e235b68c@blahtop.ina.fr:33825
```

<code>ina_face_analyzer_distributed_worker.py</code> is in charge of computing analyses and writing results to a centralized storage directory.
It requires the network adress displayed by the server in order to communicate. A good practice is to launch one worker per available GPU and set <code>CUDA_AVAILABLE_DEVICES</code>. Several workers can process the list of the server in parallel.
```bash
# CUDA_AVAILABLE_DEVICES=2 is non mandatory and tells the worker
# to use a single GPU with id 2.
# the PYRO:obj_ adress is displayed when lauching the server and
# should copy/pasted when launching the worker
CUDA_AVAILABLE_DEVICES=2 ina_face_analyzer_distributed_worker.py PYRO:obj_4c027f06be5b40e7bcf2f3f1e235b68c@blahtop.ina.fr:33825
>> received job https://github.com/ina-foss/inaFaceAnalyzer/raw/master/media/pexels-artem-podrez-5725953.mp4 /tmp/test2.csv nan nan
>> received job /home/ddoukhan/git_repos/inaFaceAnalyzer/media/pexels-artem-podrez-5725953.mp4 /tmp/test1.csv /tmp/test1.ass /tmp/test1.mp4
>>all jobs are done
```

## Using inaFaceAnalyzer API

`inaFaceAnalyzer`'s API documentation is based on several tutorial notebooks stored in directory [`tutorial_API_notebooks`](https://github.com/ina-foss/inaFaceAnalyzer/tree/master/tutorial_API_notebooks):

We provide below a brief description of these notebooks, together with links allowing to run them remotely in Google's colab platform.
When using colab platform, we recommend to [take advantage of GPU acceleration](https://www.tutorialspoint.com/google_colab/google_colab_using_free_gpu.htm).

* [Video Analysis Quick-Start](https://colab.research.google.com/github/ina-foss/inaFaceAnalyzer/blob/master/tutorial_API_notebooks/quick_start.ipynb): inaFaceAnalyzer is used to process video with default analysis parameters and export results to CSV, rich ASS subtitles and incrusted MP4. We also introduce FPS runtime argument allowing to speed-up analyses.
* [Image Analysis Quick-Start](https://colab.research.google.com/github/ina-foss/inaFaceAnalyzer/blob/master/tutorial_API_notebooks/quick_start-image.ipynb): inaFaceAnalyzer is used to process image files. Final and intermediate results are displayed and exported to CSV.
* [Advanced tutorial](https://colab.research.google.com/github/ina-foss/inaFaceAnalyzer/blob/master/tutorial_API_notebooks/advanced_tutorial.ipynb): define a custom analysis pipeline by defining 3 core parametric elements: face detection, face classification and image or video processing engine.

## Contributing
Please feel free to open issues if you have any questions or suggestions, or if you want to contribute to `inaFaceAnalyzer` development.
Every contribution is very welcome!
Please read [CONTRIBUTING.md](https://github.com/ina-foss/inaFaceAnalyzer/blob/master/.github/CONTRIBUTING.md) for more details.

## CREDITS
This work has been partially funded by the French National Research Agency (project GEM : Gender Equality Monitor : ANR-19-CE38-0012) and by European Union's Horizon 2020 research and innovation programme (project [MeMAD](https://memad.eu) : H2020 grant agreement No 780069).

We acknowledge contributions from [Zohra Rezgui](https://github.com/ZohraRezgui) who trained first models and wrote the first piece of code that lead to `inaFaceAnalyzer` during her internship at INA.
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
