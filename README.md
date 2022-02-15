# inaFaceAnalyzer: a Python toolbox for large-scale face-based description of gender representation in media with limited gender, racial and age biases
[![test py 3.7](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-7.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-7.yml)
[![test py 3.8](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-8.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-8.yml)
[![test py 3.9](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-9.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-9.yml)
<!-- [![test py 3.10](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-10.yml/badge.svg)](https://github.com/ina-foss/inaFaceAnalyzer/actions/workflows/test_py3-10.yml) -->

`inaFaceAnalyzer` is a Python toolbox designed for large-scale analysis of faces in image or video streams.
It provides a modular processing pipeline allowing to predict age and gender from faces.
Results can be exported to tables, augmented video streams, or rich ASS subtitles.
`inaFaceAnalyzer` is designed with speed in mind to perform large-scale media monitoring campaigns.
The trained age and gender classification model provided is based on a `ResNet50` architecture and evaluated on 4 face databases.
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
python run_tests.py # to check that the installation is ok
```

## Using inaFaceAnalyzer program

Most common processings can be done using the script <code>ina_face_analyzer.py</code>
provided with the distribution. Using <code>-h</code> display a detailed listing
of all options available from the command-line. Defaults parameters do not use
tracking and provide some of the best available processing options - which may
be a little slow. It computation time is an issue, we recommend using
<code>--fps 1</code> which will process a single frame per second, instead of
the whole amount of video frames.


```bash
(env) ddoukhan@blahtop:~/git_repos/inaFaceAnalyzer$ ina_face_analyzer.py -h
usage: ina_face_analyzer.py -i INPUT [INPUT ...] -o OUTPUT [-h]
                            [--type {image,video}]
                            [--classifier {Resnet50FairFaceGRA,Vggface_LSVM_YTF}]
                            [--batch_size BATCH_SIZE]
                            [--face_detector {LibFaceDetection,OcvCnnFacedetector}]
                            [--face_detection_confidence FACE_DETECTION_CONFIDENCE]
                            [--min_face_size_px SIZE_PX]
                            [--min_face_size_percent SIZE_PRCT]
                            [--face_detection_padding FACE_DETECTION_PADDING]
                            [--ass_subtitle_export] [--mp4_export] [--fps FPS]
                            [--keyframes] [--tracking FACE_DETECTION_PERIOD]
                            [--preprocessed_faces]

inaFaceAnalyzer 1.0.0+55.g918ac80.dirty: detects and classify faces from media
collections and export results in csv

required arguments:
  -i INPUT [INPUT ...]  INPUT is a list of documents to analyse. ex:
                        /home/david/test.mp4 /tmp/mymedia.avi. INPUT can be a
                        list of video paths OR a list of image paths. Videos
                        and images can have heterogenous formats but cannot be
                        mixed in a single command. (default: None)
  -o OUTPUT             When used with an input list of videos, OUTPUT is the
                        path to a directory storing one resulting CSV for each
                        processed video. OUTPUT directory should exist before
                        launching the program. When used with an input list of
                        images, OUTPUT is the path to the resulting csv file
                        storing a line for each detected faces. OUTPUT should
                        have csv extension. (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --type {image,video}  type of media to be analyzed, either a list of images
                        (JPEG, PNG, etc...) or a list of videos (AVI, MP4,
                        ...) (default: video)
  --classifier {Resnet50FairFaceGRA,Vggface_LSVM_YTF}
                        face classifier to be used in the analysis:
                        Resnet50FairFaceGRA predicts age and gender and is
                        more accurate. Vggface_LSVM_YTF was used in earlier
                        studies and predicts gender only (default:
                        Resnet50FairFaceGRA)
  --batch_size BATCH_SIZE
                        GPU batch size. Larger values allow faster
                        processings, but requires more GPU memory. Default 32
                        value used is fine for a Laptop Quadro T2000 Mobile
                        GPU with 4 Gb memory. (default: 32)

optional arguments related to face detection:
  --face_detector {LibFaceDetection,OcvCnnFacedetector}
                        face detection module to be used: LibFaceDetection can
                        take advantage of GPU acceleration and has a higher
                        recall. OcvCnnFaceDetector is embed in OpenCV. It is
                        faster for large resolutions since it first resize
                        input frames to 300*300. It may miss small faces
                        (default: LibFaceDetection)
  --face_detection_confidence FACE_DETECTION_CONFIDENCE
                        minimal confidence threshold to be used for face
                        detection. Default values are 0.98 for
                        LibFaceDetection and 0.65 for OcvCnnFacedetector
                        (default: None)
  --min_face_size_px SIZE_PX
                        minimal absolute size in pixels of the faces to be
                        considered for the analysis. Optimal classification
                        results are obtained for sizes above 75 pixels.
                        (default: 30)
  --min_face_size_percent SIZE_PRCT
                        minimal relative size (percentage between 0 and 1) of
                        the faces to be considered for the analysis with
                        repect to image frames minimal dimension (generally
                        height for videos) (default: 0)
  --face_detection_padding FACE_DETECTION_PADDING
                        Black padding percentage to be applied to image frames
                        before face detection. 0.15 Padding may help detecting
                        large faces occupying the whole image with
                        OcvCnnFacedetector. Default padding values are 0.15
                        for OcvCnnFacedetector and 0 for LibFaceDetection
                        (default: None)

optional arguments to be used only with video materials (--type video):
  --ass_subtitle_export
                        export analyses into a rich ASS subtitle file which
                        can be displayed with VLC or ELAN (default: False)
  --mp4_export          export analyses into a a MP4 video with incrusted
                        bounding boxes and analysis estimates (default: False)
  --fps FPS             Amount of video frames to be processed per second.
                        Remaining frames will be skipped. If not provided, all
                        video frames will be processed (generally between 25
                        and 30 per seconds). Lower FPS values results in
                        faster processing time. Incompatible with the
                        --keyframes argument (default: None)
  --keyframes           Face detection and analysis from video limited to
                        video key frames. Allows fastest video analysis time
                        associated to a summary with non uniform frame
                        sampling rate. Incompatible with the --fps,
                        --ass_subtitle_export or --mp4_export arguments.
                        (default: False)
  --tracking FACE_DETECTION_PERIOD
                        Activate face tracking and define
                        FACE_DETECTION_PERIOD. Face detection (costly) will be
                        performed each FACE_DETECTION_PERIOD. Face tracking
                        (cheap) will be performed for the remaining
                        (FACE_DETECTION_PERIOD -1) frames. Tracked faces are
                        associated to a numeric identifier. Tracked faces
                        classification predictions are averaged, and more
                        robust than frame-isolated predictions. To obtain the
                        most robust result, --tracking 1 will perform face
                        detection for each frame and track the detected faces
                        (default: None)

optional arguments to be used only with image material (--type image):
  --preprocessed_faces  To be used when using a list of preprocessed images.
                        Preprocessed images are assument to be already
                        detected, cropped, centered, aligned and rescaled to
                        224*224 pixels. Result will be stored in a csv file
                        with 1 line per image with name provided in --o
                        argument (default: False)

If you are using inaFaceAnalyzer in your research-related documents, please
cite the current version number used (1.0.0+55.g918ac80.dirty) together with a
reference to the following paper: David Doukhan and Thomas Petit (2022).
inaFaceAnalyzer: a Python toolbox for large-scale face-based description of
gender representation in media with limited gender, racial and age biases.
Submitted to JOSS - The journal of Open Source Software (submission in
progress).
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