# COMP213-2202628

## Abstract

This paper presents an evaluation of different face detection algorithms in Python with respect to light quality. Light quality is a vital factor in successful face detection and needs to be considered when developing or selecting algorithms to be used in real world applications. The main objective of this experiment is to determine which face detection algorithm has the greatest overall performance throughout different increments of light quality. The algorithms selected were implementations of two main libraries: OpenCV and Dlib. The algorithms are titled: Haar Cascades, CNN, HOG, Face\_recognition, MTCNN, and HOG MultiScale. Results demonstrate that Haar Cascades yielded the highest performance, showing the best average score, but had a smaller range of functionality with darker brightness levels compared to the other algorithms. These findings slightly add to the existing body of knowledge by further elaborating on a key functionality point that determines the accuracy of face detection algorithms, helping to display its importance and necessity.

## Copyright 

As stated within the Yale Face Database, "Without permission from Yale, images from within the database cannot be incorporated into a larger database which is then publicly distributed." Because of this, I am unable to post the pictures I used to github due to potential copyright issues.

The database can be found here: https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database/data

P. N. Belhumeur, J. P. Hespanha and D. J. Kriegman, "Eigenfaces vs. Fisherfaces: recognition using class specific linear projection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 19, no. 7, pp. 711-720, July 1997, doi: 10.1109/34.598228.
keywords: {Face recognition;Light scattering;Lighting;Face detection;Principal component analysis;Shadow mapping;Light sources;Pattern classification;Pixel;Error analysis},

## main.py setup:

1) install python (3.10.11)

py -m pip install dlib face_recognition matplotlib opencv-python csv imutils <br>
pip install mtcnn tensorflow
