#!/bin/bash

mkdir -a data/attributes_prediction_raw_audio;

wget https://www.dropbox.com/s/616w52hsp24roda/attributes_prediction_raw_audio.tar.gz?dl=1 -O data/attributes_prediction_raw_audio.tar.gz;
tar -xzf data/attributes_prediction_raw_audio.tar.gz -C data/attributes_prediction_raw_audio --verbose;