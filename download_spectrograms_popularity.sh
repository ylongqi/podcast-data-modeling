#!/bin/bash

mkdir -p data/popularity_prediction_spectrograms;

wget https://www.dropbox.com/s/lujjisw9iya7vem/popularity_prediction_spectrograms.tar.gz?dl=1 -O data/popularity_prediction_spectrograms.tar.gz;
tar -xzf data/popularity_prediction_spectrograms.tar.gz -C data/popularity_prediction_spectrograms --verbose;
