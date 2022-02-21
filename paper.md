---
title: 'inaFaceAnalyzer: a Python toolbox for large-scale face-based description of gender representation in media with limited gender, racial and age biases'
tags:
  - Python
  - gender detection
  - age prediction 
  - video
  - face
  - face analysis
  - face tracking
  - face detection
  - digital humanities
  - gender bias
  - racial bias
  - age bias
authors:
  - name: David Doukhan
    orcid: 0000-0002-1645-7334
    affiliation: "1" 
  - name: Thomas Petit
    orcid: 0000-0001-7289-9084
    affiliation: "1,2" # (Multiple affiliations must be quoted)
affiliations:
 - name: French National Institute of Audiovisual (INA)
   index: 1
 - name: Univ Lyon, INSA Lyon, LIRIS (UMR 5202 CNRS)
   index: 2
date: February 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`inaFaceAnalyzer` is a Python toolbox designed for large-scale analysis of faces in image or video streams.
It provides a modular processing pipeline allowing to predict age and gender from faces.
Results can be exported to tables, augmented video streams, or rich ASS subtitles.
`inaFaceAnalyzer` is designed with speed in mind to perform large-scale media monitoring campaigns.
The trained age and gender classification model provided is based on a `ResNet50` architecture and evaluated on 4 face databases.
Evaluation results are highly competitive with respect to the current state-of-the-art, and appear to reduce gender, age and racial biases.

# Statement of need

Automatic facial attribute analysis allows to extract information from images or video streams containing faces such as gender, age, emotion, identity or *race*.
This information can be used in a wide range of applications including biometrics, human-computer interaction, multimedia indexation, digital humanities and media monitoring - but also surveillance and security.

`inaFaceAnalyzer` is a Python framework aimed at extracting facial attribute information from massive video and image streams.
It was realized to meet the needs of French National Audiovisual Institute ([INA](https://www.ina.fr)), in charge of archiving and providing access to more than 22 million hours of TV and radio programs.
The emergence of computational digital humanities and data journalism has increased the need of INA's users to access meta-data obtained from automatic information extraction methods.

Since 2018, INA has realized several large-scale studies (up to 1 million hours of program analyzed) in the context of Gender Equality Monitor project, which aims at describing men and women representation differences in media.
INA's automatic softwares aimed at estimating men and women speech time [@doukhan2018describing] or decoding TV text incrustations [@doukhautorite2020] were used in digital humanity studies as well as in French public reports [@csa2021;@calvez20].
Earlier `inaFaceAnalyzer` prototypes were used to detect gender from face in TV programs, and compare these estimates to speech-time [@rezgui2019carthage;@doukhan2019estimer;@baudry20].


`inaFaceAnalyzer` has a modular design allowing easy integration of new components.
With respect to the high social impact associated to the studies using this software, it should provide high accuracy prediction models.
Being aimed at describing the representation of under-represented categories of people in media, it should minimize gender, age or racial biases that are known to also affect machine learning datasets and softwares.
It is highly configurable, allowing to define trade-offs between accuracy and processing time depending on the scale of the analyses to be performed and on the available computational resources.


# Image and Video processing pipeline

The face analysis pipeline is composed of several modules :

* **stream decoding** : image and video frames are converted to 8-bits RGB arrays. Trade-offs between processing time and result robustness can be defined by restricting analyses to a given amount of frames per second of video content (FPS), or to *key frames*.
* **face detection** : based on `libfacedetection` [@eiou] or OpenCV CNN, with customizable detection confidency and minimal size of faces to be returned. Table \ref{tab:resolutionimpact} show best predictions performances are obtained for face resolutions above 75 pixels.
<!-- * **face tracking (optional)** : a system based on DLib's correlation tracker [@danelljan2014accurate] is proposed to limit the amount of calls to time-consuming face detection methods using a user-defined *face detection period*. Tracked faces are associated to an identifier allowing to smooth prediction results and obtained more robust analysis estimates. -->
* **face tracking (optional)** : built on the top of DLib's correlation tracker [@danelljan2014accurate].  Allows to limit the amount of calls to time-consuming face detection methods, and to smooth tracked faces prediction results.
* **face preprocessing** : Facial landmarks are extracted using Dlib, and faces are rotated so that the eyes lie in a horizontal line.
A parametric scaling factor is used to extend face bounding boxes (we obtained best results on gender classification with a scaling factor of 1.1 [@rezgui2019carthage], while [@rothe2018deep] used a factor of 1.4).
Finally, faces are resized to the dimensions required by classifiers and stored in a FIFO structure, allowing to take advantage of GPU architectures by providing large batches of faces.
* **face classification** : 
Our default and best model is based on `ResNet50` architecture [@he2016deep] and implemented in `Keras` [@chollet2015keras] using input faces of size (224*224). It was trained on FairFace [@karkkainenfairface] using data augmentation strategies and a multi-objective loss allowing to predict simultaneously gender, age and race. This strategy allows to learn different concepts jointly, and optimize inference computation time.

Results are returned as `pandas` DataFrames that can be exported to tabular formats.
Export functions allows to generate videos with embed results, or ASS subtitles that can be displayed in VLC.
Lastly, command line scripts allow to process list of files with custom parameters.

Execution time and accuracy may vary according to input video resolution and analysis parameters.
Using a GeForce RTX 2080 Ti GPU, it can process a 10 minutes 960*540 video in 26 seconds, using a 1 FPS analysis rate without tracking.

# Evaluation of face gender and age classification models

Four publicly available databases with gender, age and racial annotations are used : FairFace [@karkkainenfairface], ColorFerret [@phillips2000feret], UTK Faces [@zhifei2017cvpr] and MAFA [@ge2017detecting]. MAFA is particularly challenging since it is composed of masked faces, which can be representative of audiovisual content during the Covid-19 pandemic. MAFA is mostly composed of asian faces and do not contain age annotations.

Tables \ref{tab:genderperf} and \ref{tab:ageperf} compare the results obtained with `inaFaceAnalyzer`, to those obtained with the open-source frameworks [DeepFace](https://github.com/serengil/deepface) [@serengil2020lightface] and [DEX](https://github.com/siriusdemon/pytorch-DEX) [@rothe2018deep]. DeepFace and DEX were evaluated using `inaFaceAnalyzer`'s face detection and preprocessing pipeline\footnote{DEX won ChaLearn LAP 2015 challenge on apparent age estimation and do not provide face detection. DeepFace's face detection systems do not detect all faces we're using in the evaluation. We report the best results obtained by these systems using 10 different bounding box scaling factors varying between 1 and 2.}. Gender classification is evaluated using accuracy and age prediction using Root Mean Square Error (RMSE).
In order to neutralize corpora biases (some categories of people may be more represented - impacting accuracy and RMSE), we present the results together with gender (male vs female), racial (white vs non white) and age biases (adults from 20 to 50 vs adults over 50 years old).
These estimates were obtained by calculating evaluation metric for each of the 8 (2\*2\*2) categories separately (ie: accuracy for non white females over 50 years old) and reporting their averaged difference (ex: positive accuracy age bias means the system is more accurate for people under 50).


inaFaceAnalyzer is associated to the best performances in gender and age prediction, with very low gender, age and racial biases compared to the other systems. Gender classification results are also higher than those reported in the literature for FairFace (+1 accuracy) and MAFA (+1.2 accuracy) [@karkkainenfairface;@islam2021gender]. The large negative gender bias obtained on MAFA indicates a tendency to associate masked faces to female faces, which should be improved in future releases.


\footnotesize

|System/Dataset | FairFace | ColorFerret | MAFA | UTK Faces|
|:-|-:|-:|-:|-:|
|inaFaceAnalyzer|**95.3**/**0.8**/**-1.5**/**-0.2**|**99.2**/**0.8**/**0.3**/**-0.2**|**84.9**/-8.9/**6.4**/?|**94.8**/**-1.5**/**0.4**/**0.2**|
|DeepFace|81.7/26.0/5.7/7.7|87.4/33.9/7.3/4.0|77.7/6.3/10.1/?|90.3/9.8/3.9/5.2|
|DEX|83.6/17.1/6.1/5.2|93.2/14.6/5.1/1.3|80.8/**6.2**/7.9/?|91.6/4.1/2.1/2.7|
: Gender prediction evaluation based on accuracy / gender bias  / racial bias / age bias \label{tab:genderperf}

\normalsize



| System/Dataset | FairFace | ColorFerret | UTK Faces|
|:-|-:|-:|-:|
|inaFaceAnalyzer|**6.8**/-0.8/**-0.6**/**-2.0**|**7.2**/-0.8/**-0.0**/**0.1**|**7.7**/**-2.7**/**0.1**/**-3.5**|
|DeepFace|11.8/**-0.4**/-1.5/-5.0|9.5/-1.0/**0.0**/-0.4|10.7/**-2.7**/-0.9/-5.1|
|DEX|8.7/-1.4/-1.0/-2.7|**7.2**/**-0.6**/-0.2/0.3|8.4/-2.9/-1.2/-4.1|
: Age prediction evaluation based on RMSE / gender bias / racial bias / age bias \label{tab:ageperf}



resolution (pixels) |  10|15|20|25|30|40|50|75|100|150|224  |
|:------|------:|------:|------:|------:|------:|-------:|------:|------:|------:|------:|------:|
| Gender accuracy | 63.7|76.9|83.5|88.5|90.9|93.3|94.3|95.0|95.4|95.4|95.3 |
| Age RMSE |  14.9|13.2|10.9|9.3|8.4|7.6|7.2|6.9|6.8|6.8|6.8|
: Fairface Gender and age prediction evaluation for face resolutions varying between (10\*10) and (224\*224) pixels. Lower resolution faces were obtained artificially by resizing detected faces to target dimensions using OpenCV resize function \label{tab:resolutionimpact}


# Distribution
`inaFaceAnalyzer` source code and trained models are available on [github](https://github.com/ina-foss/inaFaceAnalyzer), [PyPI](https://pypi.org/project/inaFaceAnalyzer/) and  [Dockerhub](https://hub.docker.com/r/inafoss/inafaceanalyzer);  and distributed under MIT license.
After consultation of French CNIL (French data protection authority) and DDD (French Rights Defender), racial classification layers of our models were removed from their public distribution in order to prevent their use for non ethical purposes. These models can however be provided for free after examination of each demand.


# Acknowledgements
This work has been partially funded by the French National Research Agency (project GEM: ANR-19-CE38-0012) and European Union (project MeMAD : H2020 grant agreement No 780069).
We acknowledge contributions from Zohra Rezgui wrote the first piece of code that lead to inaFaceAnalyzer during her internship at INA [@rezgui2019carthage].

# References
