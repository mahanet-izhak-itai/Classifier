

build:
	docker build -t itaimain/haar-cascade-trainer .

run:
	docker run --rm \
		-v `pwd`/positive_images:/workdir/haar-classifier/positive_images \
		-v `pwd`/negative_images:/workdir/haar-classifier/negative_images \
		-v `pwd`/output:/output \
		itaimain/haar-cascade-trainer
