#!/bin/bash

mkdir -p data/popularity_prediction_spectrograms;

wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/popularity_prediction_spectrograms.tar.gz -O data/popularity_prediction_spectrograms.tar.gz;
tar -xzf data/popularity_prediction_spectrograms.tar.gz -C data/popularity_prediction_spectrograms --verbose;
