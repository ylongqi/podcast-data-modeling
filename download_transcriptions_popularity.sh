#!/bin/bash

mkdir -p data/popularity_prediction_transcriptions;

wget https://www.dropbox.com/s/32qaoz6vqkdgikr/popularity_prediction_transcriptions.tar.gz?dl=1 -O data/popularity_prediction_transcriptions.tar.gz;
tar -xzf data/popularity_prediction_transcriptions.tar.gz -C data/popularity_prediction_transcriptions --verbose;
