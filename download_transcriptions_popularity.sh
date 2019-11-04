#!/bin/bash

mkdir -p data/popularity_prediction_transcriptions;

wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/popularity_prediction_transcriptions.tar.gz -O data/popularity_prediction_transcriptions.tar.gz;
tar -xzf data/popularity_prediction_transcriptions.tar.gz -C data/popularity_prediction_transcriptions --verbose;
