From ubuntu
MAINTAINER itaimain

RUN apt-get update && apt-get install -y git build-essential python-opencv pkg-config libopencv-dev && apt-get autoremove

WORKDIR /workdir/
RUN git clone https://github.com/Itseez/opencv.git
RUN git clone https://github.com/mrnugget/opencv-haar-classifier-training.git

RUN cd opencv && git checkout 2.4.5
RUN mv opencv-haar-classifier-training haar-classifier

ADD train.sh ./
RUN chmod +x train.sh
ADD classifier.py ./haar-classifier/

CMD sh -c 'ln -s /dev/null /dev/raw1394'; ./train.sh

RUN mkdir /output
VOLUME /output
