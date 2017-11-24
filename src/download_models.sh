#!/bin/bash

# Path to caffemodel files: http://dl.caffe.berkeleyvision.org


echo 'Downloading caffe models:'
if [ ! -f bvlc_googlenet.caffemodel ]; then
  echo 'Downloading bvlc_googlenet.caffemodel ...'
  wget  --quiet --continue http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
fi

if [ ! -f bvlc_alexnet.caffemodel ]; then
  echo 'Downloading bvlc_alexnet.caffemodel ...'
  wget -q -c http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
fi

#
# train_val.prototxt is used in training whereas 
# deploy.prototxt is used in inference.
#
if [ ! -f bvlc_alexnet.prototxt ]; then
  echo 'Downloading bvlc_alexnet.prototxt ...'
  wget -q -c https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt
  cp deploy.prototxt bvlc_alexnet.prototxt 
fi

if [ ! -f bvlc_googlenet.prototxt ]; then
  echo 'Downloading caffe bvlc_googlenet prototxt'
  wget -q -c https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/bvlc_googlenet.prototxt
fi
