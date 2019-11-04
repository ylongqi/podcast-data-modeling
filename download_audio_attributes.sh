#!/bin/bash

mkdir -a data/attributes_prediction_raw_audio;

wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/attributes_prediction_raw_audio.tar.gz -O data/attributes_prediction_raw_audio.tar.gz;
tar -xzf data/attributes_prediction_raw_audio.tar.gz -C data/attributes_prediction_raw_audio --verbose;
