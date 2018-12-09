#!/bin/bash

mkdir -p data/popularity_prediction_raw_audio;

wget https://www.dropbox.com/s/jytsd3zx8luc5bm/popularity_prediction_raw_audio.tar.gz?dl=1 -O data/popularity_prediction_raw_audio.tar.gz;
tar -xzf data/popularity_prediction_raw_audio.tar.gz -C data/popularity_prediction_raw_audio --verbose;