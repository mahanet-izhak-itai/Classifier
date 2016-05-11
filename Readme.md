Classifier
=============
Classifier is a an environment using docker to train opencv haar-cascade

Requirements
-------------
* Docker
* Image sets (positives + negatives)

How to use?
------------
First you need to have the docker image "itaimain/haar-cascade-trainer" locally.
This step can be down using two different methods:

- docker pull itaimain/haar-cascade-trainer
- ```$ make build```

Now place your negative images in a directory named negative_images  
and your positive images in a directory named positive_images.

Now it's the time to run the training sequence:  
```bash
$ make run
```

Sit back and wait for training to be finished.  
When finished, your trained haar-cascade (xml file) will be placed in an "output" directory.
