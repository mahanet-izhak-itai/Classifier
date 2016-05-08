

build:
	docker build -t classifier .

run:
	docker run --rm \
		-v `pwd`/positive_images:/workdir/haar-classifier/positive_images \
		-v `pwd`/negative_images:/workdir/haar-classifier/negative_images \
		-v `pwd`/output:/output \
		classifier
