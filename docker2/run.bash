#docker run -v$(realpath ..):/HW1 -ti --rm --network=host -u $(id -u):$(id -g) csc275-tensorflow
docker run --gpus all -v$(realpath ..):/HW1 -ti --rm --network=host csc275-tensorflow-3
#docker run --gpus all -v$(realpath ..):/HW1 -ti --rm --network=host tensorflow/tensorflow:latest-gpu
