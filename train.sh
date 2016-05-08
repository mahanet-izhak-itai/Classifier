#!/bin/bash

cd /workdir/haar-classifier
find ./positive_images -iname "*.jpg" > positives.txt
find ./negative_images -iname "*.jpg" > negatives.txt

perl bin/createsamples.pl positives.txt negatives.txt samples 1700 \
     "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 \
	-maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 24 -h 24"


cp src/mergevec.cpp /workdir/opencv/apps/haartraining
cd /workdir/opencv/apps/haartraining

g++ `pkg-config --libs --cflags opencv` -I. -o mergevec mergevec.cpp \
    cvboost.cpp cvcommon.cpp cvsamples.cpp cvhaarclassifier.cpp \
    cvhaartraining.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui  \
    -lopencv_objdetect

cd /workdir/haar-classifier
find ./samples -name '*.vec' > samples.txt
/workdir/opencv/apps/haartraining/mergevec samples.txt samples.vec

opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt \
		    -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1200 \
		    -numNeg 700 -w 24 -h 24 -mode ALL -precalcValBufSize 512 \
		    -precalcIdxBufSize 512

cp classifier/* /output
