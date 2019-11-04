#!/bin/bash

mkdir -p data/attributes_prediction_spectrograms;

wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/attributes_prediction_spectrograms.tar.gz -O data/attributes_prediction_spectrograms.tar.gz;
tar -xzf data/attributes_prediction_spectrograms.tar.gz -C data/attributes_prediction_spectrograms --verbose;
