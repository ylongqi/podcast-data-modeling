#!/bin/bash

mkdir -p data/attributes_prediction_spectrograms;

wget https://www.dropbox.com/s/2v72tiwqd5h3ual/attributes_prediction_spectrograms.tar.gz?dl=1 -O data/attributes_prediction_spectrograms.tar.gz;
tar -xzf data/attributes_prediction_spectrograms.tar.gz -C data/attributes_prediction_spectrograms --verbose;
